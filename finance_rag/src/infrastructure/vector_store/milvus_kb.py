"""问答知识库管理模块（Agentic RAG）。

基于 Milvus 2.4+ 的 BM25 稀疏向量 + ONNX INT8 本地稠密向量实现 RRF 混合检索，
使用单一集合 ``finance_kb`` 管理问答知识库。

核心能力（委托给专门模块）：
* 层级父子结构感知切块 → :mod:`finance_rag.src.rag.ingestion.chunker`
* 稠密向量 + 稀疏向量 RRF 混合检索 + BGE 重排序 → :mod:`finance_rag.src.rag.retrieval.hybrid_retriever`
* 文档增量入库 / 删除 / 列表
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pymilvus import (
    DataType,
    Function,
    FunctionType,
    MilvusClient,
)


def get_milvus_client() -> MilvusClient:
    """创建配置好的 Milvus 客户端，供基础设施适配器及管理服务复用。"""
    kwargs: dict[str, Any] = {
        "uri": MILVUS_URI,
        "timeout": MILVUS_TIMEOUT_SECONDS,
    }
    if MILVUS_TOKEN:
        kwargs["token"] = MILVUS_TOKEN
    return MilvusClient(**kwargs)

from finance_rag.src.rag.ingestion.chunker import DoclingChunks, get_chunker
from finance_rag.src.rag.models.document_category import merge_category_into_metadata
from finance_rag.src.rag.models.document_version import extract_document_version
from finance_rag.src.core.config import (
    EMBEDDING_DIM,
    EMBED_BATCH_SIZE,
    EMBEDDING_MODEL,
    ENABLE_VERSIONING,
    KB_COLLECTION_NAME,
    MAX_VERSIONS_PER_DOC,
    MILVUS_FINGERPRINT_PATH,
    MILVUS_NLIST,
    MILVUS_TIMEOUT_SECONDS,
    MILVUS_TOKEN,
    MILVUS_URI,
    RRF_K,
    TENANT_ID,
)
from .onnx_embedder import OnnxEmbedder
from .fingerprint_store import FingerprintStore
from finance_rag.src.rag.retrieval.parent_store import ParentStore
from finance_rag.src.rag.retrieval.table_store import TableStore
from finance_rag.src.rag.retrieval.hybrid_retriever import HybridRetriever
from finance_rag.src.core.exceptions import (
    EmbeddingError,
    VectorStoreError,
    VectorStoreUnavailableError,
)

logger = logging.getLogger(__name__)


# 研报维度与证据字段：``(字段名, 类型, VARCHAR 长度 或 None)``。
# 单独抽出来是为了让「集合该有哪些字段」这件事可以直接被单测锁住，
# 而不是只能靠连一个真实 Milvus 才能验证。
_REPORT_FIELD_SPECS: tuple[tuple[str, Any, int | None], ...] = (
    ("security_code", DataType.VARCHAR, 32),
    ("security_name", DataType.VARCHAR, 64),
    ("industry_l1", DataType.VARCHAR, 32),
    ("industry_l2", DataType.VARCHAR, 32),
    ("report_type", DataType.VARCHAR, 16),
    ("broker", DataType.VARCHAR, 64),
    ("meta_source", DataType.VARCHAR, 16),
    ("block_type", DataType.VARCHAR, 16),
    ("start_page", DataType.INT64, None),
    ("end_page", DataType.INT64, None),
    ("image_key", DataType.VARCHAR, 256),
    ("needs_review", DataType.BOOL, None),
)

# 研报字段的写入开关探测字段：新旧集合要么全有、要么全无（一次 DDL 加的），
# 因此只需探测一个字段即可决定整组是否可写。
_REPORT_FIELD_SENTINEL = "block_type"

# 文档级研报字段：可在不重新嵌入的前提下原地更新。
# 其余研报字段（block_type / 页码 / image_key）是块级信息，文档级更新不得覆盖。
_DOCUMENT_LEVEL_REPORT_FIELDS = frozenset({
    "security_code", "security_name", "industry_l1", "industry_l2",
    "report_type", "broker", "meta_source", "needs_review",
})


def _add_report_fields(schema: Any) -> None:
    """把研报维度与证据字段加入集合 schema。"""
    for name, dtype, max_length in _REPORT_FIELD_SPECS:
        if max_length is None:
            schema.add_field(name, dtype)
        else:
            schema.add_field(name, dtype, max_length=max_length)


def _clip(value: Any, limit: int) -> str:
    """VARCHAR 超长会让整篇文档入库失败，这里一律截断而不是抛错。"""
    return str(value or "")[:limit]


def _report_scalars(meta: dict[str, Any], chunk_meta: dict[str, Any]) -> dict[str, Any]:
    """构造研报标量字段值。

    文档级元数据（security_code / industry_* / report_type / broker / meta_source）
    来自入库时的抽取结果；块级字段（block_type / 页码 / image_key）来自切块结果。
    Milvus 标量不接受 ``None``，缺失一律给非空默认值。
    """
    return {
        "security_code": _clip(meta.get("security_code"), 32),
        "security_name": _clip(meta.get("security_name"), 64),
        "industry_l1": _clip(meta.get("industry_l1"), 32),
        "industry_l2": _clip(meta.get("industry_l2"), 32),
        "report_type": _clip(meta.get("report_type"), 16),
        "broker": _clip(meta.get("broker"), 64),
        "meta_source": _clip(meta.get("meta_source"), 16),
        "block_type": _clip(chunk_meta.get("block_type") or "text", 16),
        "start_page": int(chunk_meta.get("start_page") or 0),
        "end_page": int(chunk_meta.get("end_page") or 0),
        "image_key": _clip(chunk_meta.get("image_key"), 256),
        "needs_review": bool(meta.get("needs_review", False)),
    }




@dataclass(frozen=True)
class ParsedDocument:
    """A parsed and chunked document ready for embedding and storage."""

    path: Path
    source: str
    title: str
    chunks: DoclingChunks
    metadata: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# 集合级写锁
# ---------------------------------------------------------------------------
# 「删旧行 + 插新行」、文档间 SimHash 去重（先查再插）、版本切换与删除都必须
# **按 collection 串行**：并发写同一集合会让近似重复文档同时通过去重检查，
# 或让同 source 的覆盖互相删除。
#
# 锁刻意放在**资源侧**（而不是让各调用方自行持锁）：此前入库流水线用
# asyncio 写锁、document_service 用模块级 threading 锁，两把锁互不排斥——
# 对象存储事件回调与流水线写入并发时会同时进入 add_parsed_document。
# 放在这里以后，流水线、事件回调、管理脚本等所有调用方自动受保护。
_COLLECTION_WRITE_LOCKS: dict[str, threading.RLock] = {}
_COLLECTION_WRITE_LOCKS_GUARD = threading.Lock()


def _collection_write_lock(collection_name: str) -> threading.RLock:
    """取得（并缓存）某 collection 的写锁。

    使用 ``RLock``：同线程内的嵌套调用（例如某写方法复用另一写方法）
    不会自锁死，跨线程仍然互斥。
    """
    lock = _COLLECTION_WRITE_LOCKS.get(collection_name)
    if lock is not None:
        return lock
    with _COLLECTION_WRITE_LOCKS_GUARD:
        lock = _COLLECTION_WRITE_LOCKS.get(collection_name)
        if lock is None:
            lock = threading.RLock()
            _COLLECTION_WRITE_LOCKS[collection_name] = lock
    return lock


class KnowledgeBase:
    """问答知识库：Docling Hybrid 切块 + BM25 稀疏 + 稠密向量混合检索 + BGE 重排序。

    使用单一集合 ``finance_kb``，支持文档上传增量管理。
    切块逻辑委托给 :class:`DoclingHybridChunker`，
    检索逻辑委托给 :class:`HybridRetriever`。
    """

    def __init__(
        self,
        collection_name: str = KB_COLLECTION_NAME,
        embedding_model: str = EMBEDDING_MODEL,
    ):
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self._client: MilvusClient | None = None
        self._embeddings: OnnxEmbedder | None = None
        self._parent_store: ParentStore | None = None
        self._table_store: TableStore | None = None
        self._chunker = get_chunker()
        self._retriever: HybridRetriever | None = None
        self._fingerprint_store: FingerprintStore | None = None
        self._schema_fields: set[str] | None = None
        # 惰性初始化锁必须是**可重入**的：_get_retriever() 在持锁状态下会再调用
        # _get_client() / _get_parent_store()，而这两个方法同样 `with self._init_lock`。
        # 用普通 threading.Lock 会自死锁——只在「先入库、后检索」的顺序下侥幸不触发，
        # 新进程直接检索（CLI / 评测脚本）会永久挂起。
        self._init_lock = threading.RLock()
        # 集合级写锁（同 collection 的写操作串行；见 _collection_write_lock）
        self._write_lock = _collection_write_lock(collection_name)

    # ------------------------------------------------------------------
    # 集合管理
    # ------------------------------------------------------------------

    def ensure_collection(self) -> None:
        """确保集合存在并已加载，不存在则创建空集合。"""
        client = self._get_client()
        try:
            if not client.has_collection(
                self.collection_name, timeout=MILVUS_TIMEOUT_SECONDS
            ):
                self._create_collection(client)
            else:
                self._warn_if_stale_schema(client)
            client.load_collection(
                self.collection_name, timeout=MILVUS_TIMEOUT_SECONDS
            )
        except VectorStoreUnavailableError:
            raise
        except Exception as exc:
            raise VectorStoreUnavailableError(
                f"集合 {self.collection_name} 检查/加载失败：{exc}"
            ) from exc

    def _collection_fields(self, client: MilvusClient) -> set[str]:
        """读取集合字段名（按实例缓存；读取失败返回空集合并允许下次重试）。"""
        if self._schema_fields is None:
            try:
                desc = client.describe_collection(self.collection_name)
                self._schema_fields = {
                    f.get("name") for f in desc.get("fields", [])
                }
            except Exception as exc:
                logger.warning("读取集合 schema 失败：%s", exc)
                return set()
        return self._schema_fields

    def _collection_has_field(self, client: MilvusClient, field: str) -> bool:
        """判断集合是否含指定字段：旧 schema 集合缺字段时跳过对应写入。"""
        return field in self._collection_fields(client)

    def _collection_has_version_fields(self, client: MilvusClient) -> bool:
        """判断集合是否包含版本相关字段（version/ingested_at/is_current）。"""
        return {"version", "ingested_at", "is_current"} <= self._collection_fields(client)

    def _warn_if_stale_schema(self, client: MilvusClient) -> None:
        """检测集合是否为旧 schema（缺少新字段），是则提示重建。"""
        try:
            desc = client.describe_collection(self.collection_name)
            fields = {f.get("name") for f in desc.get("fields", [])}
            missing = {
                "category", "date", "version", "ingested_at", "is_current",
                "heading_path", _REPORT_FIELD_SENTINEL,
            } - fields
            if missing:
                logger.warning(
                    "集合 %s 是旧 schema（缺少 %s 字段），请执行 rebuild_collection() 重建",
                    self.collection_name, ", ".join(sorted(missing)),
                )
        except Exception as exc:
            logger.warning("检测集合 schema 失败：%s", exc)

    def rebuild_collection(self) -> None:
        """删除并重建集合（用于 schema 变更后的迁移）。"""
        client = self._get_client()
        if client.has_collection(
            self.collection_name, timeout=MILVUS_TIMEOUT_SECONDS
        ):
            client.drop_collection(self.collection_name)
            logger.info("已删除旧集合 %s", self.collection_name)
        self._schema_fields = None  # 清空 schema 缓存
        self._create_collection(client)
        client.load_collection(
            self.collection_name, timeout=MILVUS_TIMEOUT_SECONDS
        )
        logger.info("集合 %s 重建完成（含 category/date/version/heading_path 字段）", self.collection_name)

    def _create_collection(self, client: MilvusClient) -> None:
        """创建支持 BM25 稀疏向量的集合。"""
        schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field("content", DataType.VARCHAR, max_length=8192, enable_analyzer=True)
        schema.add_field("source", DataType.VARCHAR, max_length=512)
        schema.add_field("title", DataType.VARCHAR, max_length=128)
        schema.add_field("chunk", DataType.INT64)
        schema.add_field("parent_id", DataType.VARCHAR, max_length=64)
        schema.add_field("chunk_key", DataType.VARCHAR, max_length=64)
        schema.add_field("content_hash", DataType.VARCHAR, max_length=64)
        # 顶层到本块的标题路径（层级切块骨架），无结构时为空串
        schema.add_field("heading_path", DataType.VARCHAR, max_length=1024)
        schema.add_field("tenant_id", DataType.VARCHAR, max_length=64)
        schema.add_field("category", DataType.VARCHAR, max_length=64)
        schema.add_field("date", DataType.VARCHAR, max_length=32)
        schema.add_field("version", DataType.VARCHAR, max_length=32)
        schema.add_field("ingested_at", DataType.INT64)
        schema.add_field("is_current", DataType.BOOL)
        schema.add_field("is_deleted", DataType.BOOL)
        # 研报维度与证据字段（个股/行业/宏观 + 块类型 + 页码）
        _add_report_fields(schema)
        schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)
        schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)

        # Milvus BM25 Function：自动从 content 生成 sparse_vector
        schema.add_function(Function(
            name="bm25_fn",
            function_type=FunctionType.BM25,
            input_field_names=["content"],
            output_field_names=["sparse_vector"],
        ))

        index_params = client.prepare_index_params()
        # 稠密向量索引
        index_params.add_index(
            field_name="dense_vector",
            index_type="IVF_FLAT",
            metric_type="COSINE",
            params={"nlist": MILVUS_NLIST},
        )
        # 稀疏向量索引（BM25）
        index_params.add_index(
            field_name="sparse_vector",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
        )

        client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
            consistency_level="Strong",
        )
        logger.info("知识库集合 %s 创建成功（含 BM25 稀疏向量）", self.collection_name)

    # ------------------------------------------------------------------
    # 文档入库
    # ------------------------------------------------------------------

    def check_unchanged(self, source: str, file_path: str | Path) -> bool:
        """检查文件是否未变更，可用于增量构建跳过。

        Returns:
            True 表示文件哈希与上次入库时一致，可跳过解析+嵌入。
        """
        try:
            file_hash = FingerprintStore.hash_file(file_path)
            return self._get_fingerprint_store().is_unchanged(source, file_hash)
        except (OSError, FileNotFoundError):
            return False

    def check_unchanged_by_hash(self, source: str, content_hash: str) -> bool:
        """基于内容哈希判断文件是否未变更（避免读取本地文件）。

        Returns:
            True 表示未变更，可跳过解析；False 表示已变更或首次入库。
        """
        return not self._get_fingerprint_store().is_changed_by_hash(source, content_hash)

    def parse_document(
        self,
        file_path: str | Path,
        *,
        source: str | None = None,
        title: str | None = None,
        category: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> ParsedDocument:
        """使用 Docling 解析并通过 HybridChunker 切分文档。"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在：{path}")

        source = source or path.name
        title = title or Path(source).stem
        chunks = self._chunker.parse_and_chunk(
            path,
            source=source,
            title=title,
            category=category,
        )

        # 分类作为 metadata 的强制字段统一合并
        merged_metadata = merge_category_into_metadata(metadata, category)

        return ParsedDocument(
            path=path,
            source=source,
            title=title,
            chunks=chunks,
            metadata=merged_metadata,
        )

    def add_parsed_document(
        self,
        parsed: ParsedDocument,
        path: str | Path | None = None,
        content_hash: str | None = None,
        *,
        existing_vectors: dict[str, list[float]] | None = None,
        dense_vectors: list[list[float]] | None = None,
    ) -> dict[str, Any]:
        """嵌入并存储已经完成解析和切块的文档（**同 collection 串行**）。

        本方法内部持有集合级写锁（:func:`_collection_write_lock`），
        因此**所有调用方都无需再自行加锁**：入库流水线的写入阶段、
        对象存储事件回调、管理脚本共用同一把锁。

        参数说明见 :meth:`_add_parsed_document_locked`。
        """
        with self._write_lock:
            return self._add_parsed_document_locked(
                parsed,
                path,
                content_hash,
                existing_vectors=existing_vectors,
                dense_vectors=dense_vectors,
            )

    def _add_parsed_document_locked(
        self,
        parsed: ParsedDocument,
        path: str | Path | None = None,
        content_hash: str | None = None,
        *,
        existing_vectors: dict[str, list[float]] | None = None,
        dense_vectors: list[list[float]] | None = None,
    ) -> dict[str, Any]:
        """嵌入并存储已经完成解析和切块的文档（写锁内的实现）。

        Args:
            parsed: 解析 + 切块结果。
            path: 原始文件路径（未传 ``content_hash`` 时用于计算文件指纹）。
            content_hash: 上层已知的内容哈希，避免重复读取文件。
            existing_vectors: 集合内该 source 已有块的 ``{content_hash: 稠密向量}``。
                入库流水线在 Embedding **之前**查询并以它跳过重复嵌入；
                为 None 时本方法自行查询（保持旧调用方语义不变）。
            dense_vectors: 与 ``parsed.chunks.chunks`` 等长且一一对应的稠密向量
                （入库流水线在 Embedding 线程池中预先算好）。
                为 None 时本方法内联补齐缺失块。

        Note:
            本方法为**同步**写操作且**不自行加锁**，调用方必须通过
            :meth:`add_parsed_document` 进入（由它保证同一 collection 串行）。
        """
        source = parsed.source
        title = parsed.title
        chunks = parsed.chunks

        # Hybrid chunks 可直接用于嵌入和检索
        texts = [chunk.page_content for chunk in chunks.chunks]

        # 文档间 SimHash 去重：与已入库文档比对，疑似重复则拒绝入库
        doc_simhash = self._check_duplicate_document(source, texts)

        if dense_vectors is not None:
            if len(dense_vectors) != len(chunks.chunks):
                raise ValueError(
                    f"稠密向量数量与切块数量不一致（source={source}）："
                    f"{len(dense_vectors)} != {len(chunks.chunks)}"
                )
            resolved_vectors = list(dense_vectors)
        else:
            # 先读取同 source 的已有块向量；相同 content_hash 的块无需重复 embedding。
            if existing_vectors is None:
                existing_vectors = self.fetch_existing_vectors(source)
            missing_indexes = [
                index for index, chunk in enumerate(chunks.chunks)
                if str(chunk.metadata.get("content_hash", "")) not in existing_vectors
            ]
            new_vectors = self._embed_documents(
                [texts[index] for index in missing_indexes]
            )
            resolved_vectors = [
                existing_vectors.get(str(chunk.metadata.get("content_hash", "")), [])
                for chunk in chunks.chunks
            ]
            for index, vector in zip(missing_indexes, new_vectors, strict=True):
                resolved_vectors[index] = vector
        dense_vectors = resolved_vectors

        # 从文档级元数据中拆出分类/日期/版本，写入独立标量字段
        meta = parsed.metadata or {}
        doc_category = str(meta.get("category", "") or "")
        doc_date = str(meta.get("date", "") or "")
        doc_version = extract_document_version(source, meta)
        ingested_at = int(time.time())

        client = self._get_client()
        self.ensure_collection()
        # 旧 schema 集合（未重建）无 version/ingested_at/is_current 字段
        has_version_fields = self._collection_has_version_fields(client)
        # 旧 schema 集合无 heading_path 字段：跳过该字段写入，避免插入报错
        has_heading_path = self._collection_has_field(client, "heading_path")
        # 旧 schema 集合无研报字段：同样跳过，让未重建的集合仍可写入
        has_report_fields = self._collection_has_field(client, _REPORT_FIELD_SENTINEL)

        keep_versions = ENABLE_VERSIONING and bool(doc_version)
        if keep_versions and not has_version_fields:
            raise RuntimeError(
                "已启用版本保留（ENABLE_VERSIONING=true）但集合缺少 version 字段，"
                "请调用 KnowledgeBase.rebuild_collection() 重建集合后重新上传文档"
            )

        # 构造插入行（sparse_vector 由 Milvus Function 自动生成，不传）
        rows = []
        for chunk, dense in zip(chunks.chunks, dense_vectors, strict=True):
            chunk_id = chunk.metadata["id"]
            parent_id = chunk.metadata.get("parent_id", "")
            row: dict[str, Any] = {
                "id": chunk_id,
                "content": chunk.page_content,
                "source": source,
                "title": title,
                "chunk": int(chunk.metadata.get("chunk", 0)),
                "parent_id": parent_id,
                "chunk_key": str(chunk.metadata.get("chunk_key", chunk_id)),
                "content_hash": str(chunk.metadata.get("content_hash", "")),
                "tenant_id": TENANT_ID,
                "category": doc_category,
                "date": doc_date,
                "dense_vector": dense,
            }
            if has_version_fields:
                row.update({
                    "version": doc_version,
                    "ingested_at": ingested_at,
                    "is_current": True,
                    "is_deleted": False,
                })
            if has_heading_path:
                # 字段上限 1024：超长标题路径截断，避免整篇文档入库失败
                row["heading_path"] = str(
                    chunk.metadata.get("heading_path", "")
                )[:1024]
            if has_report_fields:
                row.update(_report_scalars(meta, chunk.metadata))
            rows.append(row)

        if keep_versions:
            # 版本保留模式：旧版本标记为非当前（检索/列表只命中最新版本）
            self._mark_old_versions_inactive(client, source)
        else:
            # 覆盖模式：删除该 source 旧记录后插入（支持重复上传覆盖）
            self._delete_by_source(client, source)

        # 存储父块到 ParentStore
        parent_store = self._get_parent_store()
        if chunks.parents:
            parent_store.store_batch([
                {
                    "id": p.id,
                    "content": p.content,
                    "heading": p.heading,
                    "heading_path": p.heading_path,
                    "source": source,
                }
                for p in chunks.parents
            ])

        # 整表落 PostgreSQL：向量库里的表格子块只有「表头 + 首行」，
        # 表体存这里，检索侧按 table_id 展开成完整表格
        if getattr(chunks, "tables", None):
            self._get_table_store().store_batch([
                {
                    "id": t.id,
                    "parent_id": t.parent_id,
                    "markdown": t.markdown,
                    "payload": t.payload,
                    "row_count": t.row_count,
                    "source": source,
                    "heading_path": t.heading_path,
                    "start_page": t.start_page,
                }
                for t in chunks.tables
            ])

        try:
            client.insert(collection_name=self.collection_name, data=rows)
            client.flush(self.collection_name)
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError(f"向量库写入失败（source={source}）：{exc}") from exc

        # 版本数超限时清理最旧版本
        if keep_versions and MAX_VERSIONS_PER_DOC > 0:
            self._prune_old_versions(client, source)

        # 更新文件指纹，用于增量构建去重
        # 计算或使用传入的哈希
        if content_hash:
            file_hash = content_hash
        elif path:
            file_hash = FingerprintStore.hash_file(path)
        else:
            file_hash = "unknown"

        self._get_fingerprint_store().update(
            source=parsed.source,
            sha256=file_hash,
            file_path=str(path) if path else parsed.source,
            chunk_count=len(rows),
            simhash=doc_simhash,
            version=doc_version,
        )

        logger.info(
            "文档入库成功：%s（%d 个 Hybrid 块）",
            Path(parsed.source).name, len(rows),
        )

        return {
            "source": source,
            "title": title,
            "chunk_count": len(rows),
            "parent_count": len(chunks.parents),
        }

    def fetch_existing_vectors(self, source: str) -> dict[str, list[float]]:
        """读取指定 source 已有块的 ``{content_hash: 稠密向量}``。

        供入库流水线在 Embedding 之前调用：命中相同 content_hash 的块无需重新嵌入。
        查询失败时返回空字典（退化行为与旧实现一致：全部重新嵌入）。
        """
        try:
            client = self._get_client()
            self.ensure_collection()
            rows = client.query(
                collection_name=self.collection_name,
                filter=(
                    f'source == "{self._escape_expr(source)}" '
                    f'and tenant_id == "{TENANT_ID}"'
                ),
                output_fields=["content_hash", "dense_vector"],
                limit=16384,
            )
        except Exception as exc:
            logger.debug("读取旧块向量失败，全部重新嵌入：%s", exc)
            return {}
        return {
            str(row.get("content_hash")): row["dense_vector"]
            for row in rows
            if row.get("content_hash") and row.get("dense_vector")
        }

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量计算文档稠密向量（公开接口，供 Embedding 线程池调用）。

        内部按 ``EMBED_BATCH_SIZE`` 分批，失败统一抛 :class:`EmbeddingError`。
        """
        return self._embed_documents(texts)

    def _check_duplicate_document(self, source: str, texts: list[str]) -> int | None:
        """文档间 SimHash 去重检查（固定流程，不可关闭）。

        计算文档级 SimHash（拼接全部子块文本），与指纹存储中已入库文档比对；
        海明距离 <= ``DUPLICATE_HAMMING_THRESHOLD``（固化阈值 3）时拒绝入库
        （排除自身 source，支持同源覆盖更新）。返回文档 SimHash（无文本时为 None）。
        """
        from finance_rag.src.rag.ingestion.cleaner import (
            DUPLICATE_HAMMING_THRESHOLD,
            find_similar_documents,
            simhash,
        )

        if not texts:
            return None

        doc_simhash = simhash("\n".join(texts))
        similar = find_similar_documents(
            doc_simhash,
            self._get_fingerprint_store().get_all_simhashes(),
            DUPLICATE_HAMMING_THRESHOLD,
            exclude_source=source,
        )
        if similar:
            dup_source, dist = similar[0]
            raise ValueError(
                f"疑似重复文档：与已入库文档 {dup_source} 高度相似"
                f"（SimHash 海明距离 {dist}），已拒绝入库"
            )
        return doc_simhash

    def soft_delete_document(self, source: str) -> dict[str, Any]:
        """Mark all source rows deleted and non-current, preserving audit data.

        内部持有集合级写锁（同 collection 串行）。
        """
        with self._write_lock:
            client = self._get_client()
            self.ensure_collection()
            expr = f'source == "{self._escape_expr(source)}" and tenant_id == "{TENANT_ID}"'
            rows = client.query(
                collection_name=self.collection_name,
                filter=expr,
                output_fields=["*"],
                limit=16384,
            )
            for row in rows:
                row["is_current"] = False
                row["is_deleted"] = True
            if rows:
                client.upsert(collection_name=self.collection_name, data=rows)
                client.flush(self.collection_name)
            return {"source": source, "deleted_count": len(rows), "soft_deleted": True}

    def update_document_metadata(
        self, source: str, metadata: dict[str, Any]
    ) -> dict[str, Any]:
        """只更新文档级研报元数据（**不重新嵌入、不改向量、不动块级字段**）。

        标量可以原地更新：文本与向量都没变，因此没必要重跑嵌入，
        也不应触发指纹失效（否则下次上传会被判为「已变更」而全量重跑）。

        ``block_type`` / 页码 / ``image_key`` 是**块级**字段，文档级更新不能碰它们，
        否则一张表或一张图的定位信息会被整篇覆盖掉。

        内部持有集合级写锁（同 collection 串行）。
        """
        with self._write_lock:
            client = self._get_client()
            self.ensure_collection()
            if not self._collection_has_field(client, _REPORT_FIELD_SENTINEL):
                raise RuntimeError(
                    "集合缺少研报元数据字段，请先执行 rebuild_collection() 重建集合"
                )

            expr = f'source == "{self._escape_expr(source)}" and tenant_id == "{TENANT_ID}"'
            rows = client.query(
                collection_name=self.collection_name,
                filter=expr,
                output_fields=["*"],
                limit=16384,
            )
            if not rows:
                return {"source": source, "updated_count": 0, "missing": True}

            scalars = {
                key: value
                for key, value in _report_scalars(metadata, {}).items()
                if key in _DOCUMENT_LEVEL_REPORT_FIELDS
            }
            for row in rows:
                row.update(scalars)
            client.upsert(collection_name=self.collection_name, data=rows)
            client.flush(self.collection_name)
            logger.info("研报元数据更新：%s（%d 行）", source, len(rows))
            return {"source": source, "updated_count": len(rows), "missing": False}

    def remove_document(
        self, source: str, version: str | None = None
    ) -> dict[str, Any]:
        """按 source（可选指定版本）删除向量记录，保留本地原文件。

        * ``version=None``：删除该 source 的全部版本；
        * 指定版本：仅删除该版本的行，其余版本保留；
        * 仅当 source 已无任何记录时清理父块存储与指纹。

        内部持有集合级写锁（同 collection 串行）。
        """
        with self._write_lock:
            client = self._get_client()
            self.ensure_collection()
            deleted = self._delete_by_source(client, source, version=version)
            client.flush(self.collection_name)

            remaining = self._count_by_source(client, source)
            if remaining == 0:
                # 同步清理父块与整表存储及指纹（文档已完全删除）
                self._get_parent_store().delete_by_source(source)
                self._get_table_store().delete_by_source(source)
                self._get_fingerprint_store().remove(source)

            logger.info(
                "知识库向量删除：%s（version=%s，%d 条记录，本地文件保留）",
                source, version or "all", deleted,
            )
            return {"source": source, "version": version or "", "deleted_count": deleted}

    def _mark_old_versions_inactive(self, client: MilvusClient, source: str) -> None:
        """将该 source 现有的 is_current==true 行标记为 False（版本保留模式）。"""
        expr = (
            f'source == "{self._escape_expr(source)}" '
            f'and tenant_id == "{TENANT_ID}" and is_current == true'
        )
        rows: list[dict[str, Any]] = []
        iterator = client.query_iterator(
            collection_name=self.collection_name,
            filter=expr,
            output_fields=["*"],
            batch_size=1000,
        )
        try:
            while True:
                batch = iterator.next()
                if not batch:
                    break
                for row in batch:
                    row["is_current"] = False
                    rows.append(row)
        finally:
            iterator.close()

        if rows:
            client.upsert(collection_name=self.collection_name, data=rows)

    def _prune_old_versions(self, client: MilvusClient, source: str) -> None:
        """保留该 source 最新 ``MAX_VERSIONS_PER_DOC`` 个版本，删除更旧版本的行。"""
        expr = (
            f'source == "{self._escape_expr(source)}" '
            f'and tenant_id == "{TENANT_ID}"'
        )
        rows: list[dict[str, Any]] = []
        iterator = client.query_iterator(
            collection_name=self.collection_name,
            filter=expr,
            output_fields=["id", "version", "ingested_at"],
            batch_size=1000,
        )
        try:
            while True:
                batch = iterator.next()
                if not batch:
                    break
                rows.extend(batch)
        finally:
            iterator.close()

        # 每个版本取最大 ingested_at 作为版本时间
        version_time: dict[str, int] = {}
        for row in rows:
            v = row.get("version", "") or ""
            ts = int(row.get("ingested_at", 0) or 0)
            if ts > version_time.get(v, 0):
                version_time[v] = ts

        kept_versions = {
            v for v, _ in sorted(
                version_time.items(), key=lambda item: item[1], reverse=True
            )[: MAX_VERSIONS_PER_DOC]
        }
        stale_ids = [
            row["id"] for row in rows if (row.get("version", "") or "") not in kept_versions
        ]
        if stale_ids:
            self._delete_by_ids(client, stale_ids)
            logger.info(
                "版本清理：%s 删除 %d 行（保留最新 %d 个版本）",
                source, len(stale_ids), MAX_VERSIONS_PER_DOC,
            )

    def list_documents(self, include_versions: bool = False) -> list[dict[str, Any]]:
        """列出知识库中的文档（按 source 聚合）。

        ``include_versions=True`` 时返回全部版本明细（含历史版本），
        否则仅统计当前版本（``is_current == true``）并附带当前版本号。
        """
        client = self._get_client()
        if not client.has_collection(
            self.collection_name, timeout=MILVUS_TIMEOUT_SECONDS
        ):
            return []
        self.ensure_collection()

        filter_expr = f'tenant_id == "{TENANT_ID}"'
        if not include_versions and ENABLE_VERSIONING:
            filter_expr += " and is_current == true"

        try:
            results = client.query(
                collection_name=self.collection_name,
                filter=filter_expr,
                output_fields=[
                    "source", "title", "chunk", "category", "date",
                    "version", "ingested_at", "is_current",
                ],
                limit=16384,
            )
        except Exception as exc:
            logger.warning("列出文档失败：%s", exc)
            return []

        agg: dict[str, dict[str, Any]] = {}
        for r in results:
            src = r.get("source", "")
            if src not in agg:
                agg[src] = {
                    "source": src,
                    "title": r.get("title", ""),
                    "category": r.get("category", ""),
                    "date": r.get("date", ""),
                    "chunk_count": 0,
                    "version": r.get("version", ""),
                    "versions": [],
                }
            item = agg[src]
            item["chunk_count"] += 1
            if include_versions:
                item["versions"].append({
                    "version": r.get("version", "") or "",
                    "chunk_count": 0,  # 在下方汇总
                    "ingested_at": int(r.get("ingested_at", 0) or 0),
                    "is_current": bool(r.get("is_current", True)),
                })

        if include_versions:
            for item in agg.values():
                # 按版本汇总块数与最新入库时间
                merged: dict[str, dict[str, Any]] = {}
                for v in item["versions"]:
                    key = v["version"]
                    if key not in merged:
                        merged[key] = {"version": key, "chunk_count": 0,
                                       "ingested_at": 0, "is_current": False}
                    merged[key]["chunk_count"] += 1
                    merged[key]["ingested_at"] = max(
                        merged[key]["ingested_at"], v["ingested_at"]
                    )
                    merged[key]["is_current"] = (
                        merged[key]["is_current"] or v["is_current"]
                    )
                item["versions"] = sorted(
                    merged.values(), key=lambda v: v["ingested_at"], reverse=True
                )

        return list(agg.values())

    def get_stats(self) -> dict[str, Any]:
        """知识库统计信息。"""
        client = self._get_client()
        has_collection = client.has_collection(
            self.collection_name, timeout=MILVUS_TIMEOUT_SECONDS
        )
        if not has_collection:
            return {
                "collection": self.collection_name,
                "exists": False,
                "document_count": 0,
                "chunk_count": 0,
            }
        self.ensure_collection()
        docs = self.list_documents()
        return {
            "collection": self.collection_name,
            "exists": True,
            "document_count": len(docs),
            "chunk_count": sum(d["chunk_count"] for d in docs),
        }

    # ------------------------------------------------------------------
    # 混合检索（委托给 HybridRetriever）
    # ------------------------------------------------------------------

    def hybrid_search(
        self,
        query: str,
        k: int = 5,
        *,
        use_dense_only: bool = False,
        expand_parents: bool = True,
        use_rerank: bool = False,
        rerank_top_n: int = 3,
        filters: dict[str, Any] | None = None,
        rrf_k: int = RRF_K,
        keywords: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """稠密 + 稀疏（BM25）RRF 混合检索，可选 BGE 重排序 + 关键词增强。

        Args:
            query: 查询文本。
            k: 返回结果数（重排序关闭时的返回数量）。
            use_dense_only: True 时只走稠密向量检索，False 时走 RRF 混合检索。
            expand_parents: 是否按 parent_id 去重并替换为父块全文。
            use_rerank: 是否启用 BGE 本地重排序。
            rerank_top_n: 启用重排序时的最终返回数量。
            filters: 元数据过滤条件（需 ENABLE_METADATA_FILTER=true）。
            rrf_k: RRF 融合常数。
            keywords: 可选的关键词列表，用于过滤和重排序。

        Returns:
            检索结果列表，每项含
            ``id / chunk_key / content / source / title / chunk / score / parent_id``
            （父子扩展时另含 ``child_id`` / ``parent_key``）。
            重排序启用时额外含 ``rerank_score`` 字段。

        Note:
            重排序失败时自动回退到 Milvus 原始排序，保证检索链路可用。
        """
        self.ensure_collection()
        retriever = self._get_retriever()
        return retriever.search(
            query,
            k=k,
            use_dense_only=use_dense_only,
            expand_parents=expand_parents,
            use_rerank=use_rerank,
            rerank_top_n=rerank_top_n,
            filters=filters,
            rrf_k=rrf_k,
            keywords=keywords,
        )

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _get_retriever(self) -> HybridRetriever:
        """延迟创建混合检索器。"""
        if self._retriever is None:
            with self._init_lock:
                if self._retriever is None:
                    self._retriever = HybridRetriever(
                        client=self._get_client(),
                        collection_name=self.collection_name,
                        embed_query_fn=self._embed_query,
                        parent_store=self._get_parent_store(),
                    )
        return self._retriever

    @staticmethod
    def _escape_expr(value: str) -> str:
        """转义 Milvus 过滤表达式中的字符串字面量。"""
        return value.replace("\\", "\\\\").replace('"', '\\"')

    def _delete_by_source(
        self, client: MilvusClient, source: str, version: str | None = None
    ) -> int:
        """删除指定 source（可选指定版本）的所有记录，返回删除行数。"""
        expr = (
            f'source == "{self._escape_expr(source)}" '
            f'and tenant_id == "{TENANT_ID}"'
        )
        if version is not None:
            expr += f' and version == "{self._escape_expr(version)}"'
        try:
            # Milvus delete 不返回精确计数，删除前先统计匹配行数
            deleted = self._count_matching(client, expr)
            client.delete(
                collection_name=self.collection_name,
                filter=expr,
            )
            return deleted
        except Exception as exc:
            raise VectorStoreError(
                f"删除 source={source} version={version or 'all'} 失败：{exc}"
            ) from exc

    def _count_by_source(self, client: MilvusClient, source: str) -> int:
        """统计指定 source 的剩余记录行数（用于判断是否完全删除）。"""
        expr = (
            f'source == "{self._escape_expr(source)}" '
            f'and tenant_id == "{TENANT_ID}"'
        )
        return self._count_matching(client, expr)

    def _count_matching(self, client: MilvusClient, expr: str) -> int:
        """统计匹配过滤表达式的行数；失败抛 VectorStoreError。"""
        try:
            res = client.query(
                collection_name=self.collection_name,
                filter=expr,
                output_fields=["count(*)"],
            )
            return int(res[0].get("count(*)", 0)) if res else 0
        except Exception as exc:
            raise VectorStoreError(f"统计行数失败（expr={expr}）：{exc}") from exc

    def _delete_by_ids(self, client: MilvusClient, ids: list[str]) -> None:
        """按主键批量删除记录。"""
        if not ids:
            return
        # 分批拼接 id in [...] 过滤表达式，避免表达式过长
        for i in range(0, len(ids), 100):
            batch = ids[i : i + 100]
            quoted = ", ".join(f'"{self._escape_expr(rid)}"' for rid in batch)
            client.delete(
                collection_name=self.collection_name,
                filter=f"id in [{quoted}]",
            )

    def _get_client(self) -> MilvusClient:
        if self._client is None:
            with self._init_lock:
                if self._client is None:
                    try:
                        self._client = get_milvus_client()
                    except Exception as exc:
                        raise VectorStoreUnavailableError(
                            f"向量数据库客户端初始化失败（uri={MILVUS_URI}）：{exc}"
                        ) from exc
        return self._client

    def _get_embeddings(self) -> OnnxEmbedder:
        if self._embeddings is None:
            with self._init_lock:
                if self._embeddings is None:
                    self._embeddings = OnnxEmbedder(model_name=self.embedding_model)
        return self._embeddings

    def _get_parent_store(self) -> ParentStore:
        if self._parent_store is None:
            with self._init_lock:
                if self._parent_store is None:
                    # 父块全文持久化到 PostgreSQL，按集合名隔离
                    self._parent_store = ParentStore(collection=self.collection_name)
        return self._parent_store

    def _get_table_store(self) -> TableStore:
        if self._table_store is None:
            with self._init_lock:
                if self._table_store is None:
                    # 整表持久化到 PostgreSQL（向量库只留「表头 + 首行」索引）
                    self._table_store = TableStore(collection=self.collection_name)
        return self._table_store

    def _get_fingerprint_store(self) -> FingerprintStore:
        if self._fingerprint_store is None:
            with self._init_lock:
                if self._fingerprint_store is None:
                    # 非默认集合按集合名隔离指纹文件
                    if self.collection_name == KB_COLLECTION_NAME:
                        self._fingerprint_store = FingerprintStore()
                    else:
                        self._fingerprint_store = FingerprintStore(
                            MILVUS_FINGERPRINT_PATH + self._collection_suffix()
                        )
        return self._fingerprint_store

    def _collection_suffix(self) -> str:
        """存储路径隔离后缀（仅非默认集合使用）。"""
        import re as _re
        safe = _re.sub(r"[^a-zA-Z0-9_]", "_", self.collection_name)
        return f"_{safe}"

    def _embed_documents(self, texts: list[str]) -> list[list[float]]:
        """分批嵌入文档"""
        embeddings: list[list[float]] = []
        try:
            for i in range(0, len(texts), EMBED_BATCH_SIZE):
                batch = texts[i:i + EMBED_BATCH_SIZE]
                embeddings.extend(self._get_embeddings().embed_documents(batch))
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(f"文档向量化失败：{exc}") from exc
        return embeddings

    def _embed_query(self, text: str) -> list[float]:
        try:
            return self._get_embeddings().embed_query(text)
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(f"查询向量化失败：{exc}") from exc

    def embed_query(self, text: str) -> list[float]:
        """生成查询向量，供检索前的语义一致性校验使用。"""
        return self._embed_query(text)

    def get_embeddings(self) -> OnnxEmbedder:
        """公开获取嵌入器实例（供引用校验等跨模块使用）。"""
        return self._get_embeddings()

    def get_fingerprint_store(self) -> FingerprintStore:
        """公开获取指纹存储实例（供文档服务做增量检查）。"""
        return self._get_fingerprint_store()

    @staticmethod
    def cosine_similarity(left: list[float], right: list[float]) -> float:
        """计算两个向量的余弦相似度；零向量返回 0。"""
        if len(left) != len(right) or not left:
            raise ValueError("向量维度不一致或为空")
        dot_product = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        denominator = left_norm * right_norm
        return dot_product / denominator if denominator else 0.0


# ---------------------------------------------------------------------------
# 单例
# ---------------------------------------------------------------------------

_KB_INSTANCES: dict[str, KnowledgeBase] = {}
_KB_LOCK = threading.Lock()


def get_knowledge_base(collection_name: str = KB_COLLECTION_NAME) -> KnowledgeBase:
    """获取（并缓存）指定集合的 KnowledgeBase 实例（线程安全）。

    默认返回全局配置集合的实例；传入其他集合名则返回对应实例，
    实例按集合名缓存，供多知识库管理与跨库检索使用。
    """
    kb = _KB_INSTANCES.get(collection_name)
    if kb is None:
        with _KB_LOCK:
            kb = _KB_INSTANCES.get(collection_name)
            if kb is None:
                kb = KnowledgeBase(collection_name=collection_name)
                _KB_INSTANCES[collection_name] = kb
    return kb
