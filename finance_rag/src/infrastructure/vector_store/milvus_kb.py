"""问答知识库管理模块（Agentic RAG）。

基于 Milvus 2.4+ 的 BM25 稀疏向量 + ONNX INT8 本地稠密向量实现 RRF 混合检索，
使用单一集合 ``finance_kb`` 管理问答知识库。

核心能力（委托给专门模块）：
* 层级父子结构感知切块 → :mod:`finance_rag.chunking`
* 稠密向量 + 稀疏向量 RRF 混合检索 + BGE 重排序 → :mod:`finance_rag.retrieve`
* 文档增量入库 / 删除 / 列表
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
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

from .chunker import DoclingChunks, get_chunker
from config.settings import (
    EMBEDDING_DIM,
    EMBED_BATCH_SIZE,
    EMBEDDING_MODEL,
    KB_COLLECTION_NAME,
    MILVUS_FINGERPRINT_PATH,
    MILVUS_NLIST,
    MILVUS_TIMEOUT_SECONDS,
    MILVUS_TOKEN,
    MILVUS_URI,
    PARENT_STORE_PATH,
    RRF_K,
    TENANT_ID,
)
from .onnx_embedder import OnnxEmbedder
from finance_rag.src.utils.logger import agent_logger
from .parent_store import ParentStore
from .hybrid_retriever import HybridRetriever

logger = logging.getLogger(__name__)




@dataclass(frozen=True)
class ParsedDocument:
    """A parsed and chunked document ready for embedding and storage."""

    path: Path
    source: str
    title: str
    chunks: DoclingChunks


class FingerprintStore:
    """文件哈希指纹存储，用于增量构建：跳过未变更文件的重复解析+嵌入。

    以 JSON 文件持久化到 ``MILVUS_FINGERPRINT_PATH``。
    """

    def __init__(self, store_path: str | None = None):
        self._path = Path(store_path) if store_path else Path(MILVUS_FINGERPRINT_PATH)
        self._data: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                self._data = json.loads(self._path.read_text("utf-8"))
            except (json.JSONDecodeError, OSError):
                self._data = {}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2), "utf-8"
        )

    def get(self, source: str) -> dict[str, Any] | None:
        """获取指定 source 的指纹记录，不存在返回 None。"""
        return self._data.get(source)

    def is_unchanged(self, source: str, file_hash: str) -> bool:
        """检查文件是否未变更（哈希一致）。"""
        record = self._data.get(source)
        return record is not None and record.get("hash") == file_hash

    def update(self, source: str, file_hash: str, file_path: str, chunk_count: int) -> None:
        """更新（或创建）指纹记录。"""
        self._data[source] = {
            "hash": file_hash,
            "file_path": file_path,
            "chunk_count": chunk_count,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        }
        self._save()

    def remove(self, source: str) -> None:
        """删除指纹记录。"""
        self._data.pop(source, None)
        self._save()

    @staticmethod
    def hash_file(file_path: str | Path) -> str:
        """计算文件 SHA256 哈希。"""
        sha = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                sha.update(chunk)
        return sha.hexdigest()


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
        docs_dir: str | None = None,
    ):
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.docs_dir = Path(docs_dir) if docs_dir else Path(
            os.getenv("UPLOAD_DIR", "files")
        )
        self._client: MilvusClient | None = None
        self._embeddings: OnnxEmbedder | None = None
        self._parent_store: ParentStore | None = None
        self._chunker = get_chunker()
        self._retriever: HybridRetriever | None = None
        self._fingerprint_store: FingerprintStore | None = None

    # ------------------------------------------------------------------
    # 集合管理
    # ------------------------------------------------------------------

    def ensure_collection(self) -> None:
        """确保集合存在并已加载，不存在则创建空集合。"""
        client = self._get_client()
        if not client.has_collection(
            self.collection_name, timeout=MILVUS_TIMEOUT_SECONDS
        ):
            self._create_collection(client)
        client.load_collection(
            self.collection_name, timeout=MILVUS_TIMEOUT_SECONDS
        )

    def _create_collection(self, client: MilvusClient) -> None:
        """创建支持 BM25 稀疏向量的集合。"""
        schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field("content", DataType.VARCHAR, max_length=8192, enable_analyzer=True)
        schema.add_field("source", DataType.VARCHAR, max_length=512)
        schema.add_field("title", DataType.VARCHAR, max_length=128)
        schema.add_field("chunk", DataType.INT64)
        schema.add_field("parent_id", DataType.VARCHAR, max_length=64)
        schema.add_field("tenant_id", DataType.VARCHAR, max_length=64)
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

    def add_document(
        self,
        file_path: str | Path,
        *,
        source: str | None = None,
        title: str | None = None,
    ) -> dict[str, Any]:
        """使用 Docling 解析并进行 Hybrid 切块 → 嵌入 → 插入 Milvus。

        Returns:
            ``{"source", "title", "chunk_count", "parent_count"}``
        """
        parsed = self.parse_document(file_path, source=source, title=title)
        return self.add_parsed_document(parsed)

    def parse_document(
        self,
        file_path: str | Path,
        *,
        source: str | None = None,
        title: str | None = None,
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
        )

        return ParsedDocument(
            path=path,
            source=source,
            title=title,
            chunks=chunks,
        )

    def add_parsed_document(self, parsed: ParsedDocument) -> dict[str, Any]:
        """嵌入并存储已经完成解析和切块的文档。"""
        path = parsed.path
        source = parsed.source
        title = parsed.title
        chunks = parsed.chunks

        # Hybrid chunks 可直接用于嵌入和检索
        texts = [chunk.page_content for chunk in chunks.chunks]
        dense_vectors = self._embed_documents(texts)

        # 构造插入行（sparse_vector 由 Milvus Function 自动生成，不传）
        rows = []
        for chunk, dense in zip(chunks.chunks, dense_vectors):
            chunk_id = chunk.metadata["id"]
            parent_id = chunk.metadata.get("parent_id", "")
            rows.append({
                "id": chunk_id,
                "content": chunk.page_content,
                "source": source,
                "title": title,
                "chunk": int(chunk.metadata.get("chunk", 0)),
                "parent_id": parent_id,
                "tenant_id": TENANT_ID,
                "dense_vector": dense,
            })

        client = self._get_client()
        self.ensure_collection()

        # 先删除该 source 的旧记录（支持重复上传覆盖）
        self._delete_by_source(client, source)

        # 存储父块到 ParentStore
        parent_store = self._get_parent_store()
        if chunks.parents:
            parent_store.store_batch([
                {"id": p.id, "content": p.content, "heading": p.heading, "source": source}
                for p in chunks.parents
            ])

        client.insert(collection_name=self.collection_name, data=rows)
        client.flush(self.collection_name)

        # 更新文件指纹，用于增量构建去重
        file_hash = FingerprintStore.hash_file(path)
        self._get_fingerprint_store().update(source, file_hash, str(path), len(rows))

        logger.info(
            "文档入库成功：%s（%d 个 Hybrid 块）",
            path.name, len(rows),
        )

        return {
            "source": source,
            "title": title,
            "chunk_count": len(rows),
            "parent_count": len(chunks.parents),
        }

    def remove_document(self, source: str) -> dict[str, Any]:
        """按 source 删除文档的所有向量记录，保留本地原文件。"""
        client = self._get_client()
        self.ensure_collection()
        deleted = self._delete_by_source(client, source)
        client.flush(self.collection_name)

        # 同步清理父块存储
        self._get_parent_store().delete_by_source(source)

        # 同步清理指纹
        self._get_fingerprint_store().remove(source)

        logger.info("知识库向量删除：%s（%d 条记录，本地文件保留）", source, deleted)
        return {"source": source, "deleted_count": deleted}

    def list_documents(self) -> list[dict[str, Any]]:
        """列出知识库中的所有文档（按 source 聚合）。"""
        client = self._get_client()
        if not client.has_collection(
            self.collection_name, timeout=MILVUS_TIMEOUT_SECONDS
        ):
            return []
        self.ensure_collection()

        try:
            results = client.query(
                collection_name=self.collection_name,
                filter=f'tenant_id == "{TENANT_ID}"',
                output_fields=["source", "title", "chunk"],
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
                    "chunk_count": 0,
                }
            agg[src]["chunk_count"] += 1

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
            检索结果列表，每项含 content/source/title/chunk/score/parent_id。
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
            self._retriever = HybridRetriever(
                client=self._get_client(),
                collection_name=self.collection_name,
                embed_query_fn=self._embed_query,
                parent_store=self._get_parent_store(),
            )
        return self._retriever

    def _delete_by_source(self, client: MilvusClient, source: str) -> int:
        """删除指定 source 的所有记录，返回删除数。"""
        try:
            escaped = source.replace("\\", "\\\\").replace('"', '\\"')
            client.delete(
                collection_name=self.collection_name,
                filter=f'source == "{escaped}" and tenant_id == "{TENANT_ID}"',
            )
            return 1  # Milvus delete 不返回精确计数
        except Exception as exc:
            logger.warning("删除 source=%s 失败：%s", source, exc)
            return 0

    def _get_client(self) -> MilvusClient:
        if self._client is None:
            kwargs: dict[str, Any] = {
                "uri": MILVUS_URI,
                "timeout": MILVUS_TIMEOUT_SECONDS,
            }
            if MILVUS_TOKEN:
                kwargs["token"] = MILVUS_TOKEN
            self._client = MilvusClient(**kwargs)
        return self._client

    def _get_embeddings(self) -> OnnxEmbedder:
        if self._embeddings is None:
            self._embeddings = OnnxEmbedder(model_name=self.embedding_model)
        return self._embeddings

    def _get_parent_store(self) -> ParentStore:
        if self._parent_store is None:
            self._parent_store = ParentStore()
        return self._parent_store

    def _get_fingerprint_store(self) -> FingerprintStore:
        if self._fingerprint_store is None:
            self._fingerprint_store = FingerprintStore()
        return self._fingerprint_store

    def _embed_documents(self, texts: list[str]) -> list[list[float]]:
        """分批嵌入文档"""
        embeddings: list[list[float]] = []
        for i in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[i:i + EMBED_BATCH_SIZE]
            embeddings.extend(self._get_embeddings().embed_documents(batch))
        return embeddings

    def _embed_query(self, text: str) -> list[float]:
        return self._get_embeddings().embed_query(text)

    def embed_query(self, text: str) -> list[float]:
        """生成查询向量，供检索前的语义一致性校验使用。"""
        return self._embed_query(text)

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

_KB_INSTANCE: KnowledgeBase | None = None


def get_knowledge_base() -> KnowledgeBase:
    """获取全局 KnowledgeBase 单例。"""
    global _KB_INSTANCE
    if _KB_INSTANCE is None:
        _KB_INSTANCE = KnowledgeBase()
    return _KB_INSTANCE
