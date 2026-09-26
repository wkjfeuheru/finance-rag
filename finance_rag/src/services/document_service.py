"""文档上传/解析/管理模块（Agentic RAG）。

负责接收前端上传的文件，通过存储抽象层持久化文件，调用 :class:`KnowledgeBase`
完成切块/嵌入/入库，支持文档列表与删除。
上传统一走异步模式：立即返回 task_id，后台解析入库。
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import hashlib
import logging
import os
import re
import tempfile
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import UploadFile

from finance_rag.src.core.config import (
    DOCUMENT_PARSE_WORKERS,
    INGEST_PIPELINE_ENABLED,
    KB_COLLECTION_NAME,
    MAX_UPLOAD_SIZE_MB,
    MINERU_SUPPORTED_EXTENSIONS,
    TENANT_ID,
)
from finance_rag.src.infrastructure.storage import get_storage
from finance_rag.src.infrastructure.vector_store.milvus_kb import (
    KnowledgeBase,
    ParsedDocument,
    get_knowledge_base,
)
from finance_rag.src.rag.models.document_category import merge_category_into_metadata
from finance_rag.src.services.task_service import TaskStatus, get_task_manager

logger = logging.getLogger(__name__)

# 支持的文件格式
SUPPORTED_EXTENSIONS = set(MINERU_SUPPORTED_EXTENSIONS)

# 注：向量库写入的「同 collection 串行」由 KnowledgeBase 在资源侧自持写锁保证
# （见 milvus_kb._collection_write_lock），本模块不再维护自己的上传锁——
# 否则流水线的 asyncio 写锁与本模块的 threading 锁互不排斥，
# 对象存储事件回调会与流水线写入并发进入 add_parsed_document。


def _safe_delete(storage, key: str) -> None:
    """安全删除存储对象，忽略不存在的 key。"""
    try:
        storage.delete_sync(key)
    except Exception as exc:
        logger.warning("删除存储对象失败 key=%s: %s", key, exc)


def _unlink_temp_file(path: Path) -> None:
    """删除上传临时文件（不存在时静默忽略）。"""
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:  # pragma: no cover - 防御：清理失败不阻断入库
        logger.debug("上传临时文件清理失败 %s：%s", path, exc)


# ---------------------------------------------------------------------------
# 文档日期自动提取
# ---------------------------------------------------------------------------

# 关键词邻域日期：发布日期/报告期/日期 等关键词后紧跟的日期（优先级最高）
_DATE_KEYWORD_RE = re.compile(
    r"(?:发布日期|发布时间|报告日期|报告期|日期|公布日期)[:：]?\s*"
    r"(20\d{2})\s*[年\-/.]\s*(\d{1,2})\s*(?:[月\-/.]\s*(\d{1,2})\s*日?)?"
)
# 完整日期：2024年5月1日 / 2024-05-01 / 2024.5.1 / 2024/05/01
_DATE_FULL_RE = re.compile(
    r"(20\d{2})\s*[年\-/.]\s*(\d{1,2})\s*[月\-/.]\s*(\d{1,2})\s*日?"
)
# 年月：2024年5月 / 2024年05月（无具体日，补 01）
_DATE_YM_RE = re.compile(r"(20\d{2})\s*年\s*(\d{1,2})\s*月")

# 内容提取的扫描范围：文档头部（封面/标题区）
_DATE_SCAN_CHARS = 3000

# --- 文件名日期提取（自动提取的最高优先级） ---
# 8 位连续数字：20240501
_FN_YYYYMMDD_RE = re.compile(
    r"(?<!\d)(20\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])(?!\d)"
)
# 年月日：2024-05-01 / 2024.5.1 / 2024/05/01 / 2024年5月1日 / 2024_05_01
_FN_FULL_RE = re.compile(
    r"(20\d{2})\s*[年\-_.]\s*(\d{1,2})\s*[月\-_.]\s*(\d{1,2})\s*日?"
)
# 年月：2024-05 / 2024.5 / 2024年5月 / 2024_05
_FN_YM_RE = re.compile(r"(20\d{2})\s*[年\-_.]\s*(\d{1,2})\s*月?(?!\d)")
# 纯年份：2024 / 2024年（前后非数字，兼容〔2024〕等括号形式）
_FN_YEAR_RE = re.compile(r"(?<!\d)(20\d{2})\s*年?(?!\d)")


def _normalize_ymd(year: str, month: str, day: str | None) -> str | None:
    """将年/月/日分组规整为 YYYY-MM-DD；月日非法时返回 None（日缺省为 01）。"""
    try:
        y, m = int(year), int(month)
        if not 1 <= m <= 12:
            return None
        d = int(day) if day else 1
        if not 1 <= d <= 31:
            return None
        return f"{y:04d}-{m:02d}-{d:02d}"
    except (TypeError, ValueError):
        return None


def extract_date_from_text(text: str | None) -> str | None:
    """从文档头部提取日期，返回 YYYY-MM-DD；提取不到返回 None。

    提取优先级：关键词邻域日期 > 完整日期 > 年月（补 01）。
    """
    if not text:
        return None
    head = text[:_DATE_SCAN_CHARS]

    m = _DATE_KEYWORD_RE.search(head)
    if m:
        date = _normalize_ymd(m.group(1), m.group(2), m.group(3))
        if date:
            return date

    m = _DATE_FULL_RE.search(head)
    if m:
        date = _normalize_ymd(m.group(1), m.group(2), m.group(3))
        if date:
            return date

    m = _DATE_YM_RE.search(head)
    if m:
        return _normalize_ymd(m.group(1), m.group(2), None)

    return None


def extract_date_from_filename(filename: str | None) -> str | None:
    """从文件名提取日期，返回 YYYY-MM-DD；提取不到返回 None。

    依次降级匹配：8 位数字（20240501）> 年月日 > 年月（补 01）> 纯年份（补 01-01）。
    """
    if not filename:
        return None
    name = Path(filename).name

    m = _FN_YYYYMMDD_RE.search(name)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"

    m = _FN_FULL_RE.search(name)
    if m:
        date = _normalize_ymd(m.group(1), m.group(2), m.group(3))
        if date:
            return date

    m = _FN_YM_RE.search(name)
    if m:
        date = _normalize_ymd(m.group(1), m.group(2), None)
        if date:
            return date

    m = _FN_YEAR_RE.search(name)
    if m:
        return _normalize_ymd(m.group(1), "1", None)

    return None


def date_from_file_mtime(path: Path) -> str | None:
    """取文件修改时间作为兜底日期（YYYY-MM-DD）。"""
    try:
        return datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d")
    except OSError:
        return None


def resolve_document_date(parsed: ParsedDocument) -> str:
    """解析文档日期，优先级：显式 metadata['date'] > 文件名 > 内容提取 > 文件修改时间。

    Returns:
        YYYY-MM-DD 字符串；全部失败返回 ""。
    """
    explicit = (parsed.metadata or {}).get("date")
    if explicit:
        return str(explicit)

    # 1. 文件名提取（最高自动优先级，文件名通常包含发布日期且不易误匹配）
    fn_date = extract_date_from_filename(parsed.path.name if parsed.path else parsed.source)
    if fn_date:
        return fn_date

    # 2. 内容提取：优先用整篇 markdown；无 markdown 时用前几个切块拼接
    markdown = getattr(parsed.chunks, "markdown", None)
    if not markdown:
        markdown = "\n".join(
            chunk.page_content for chunk in parsed.chunks.chunks[:5]
        )

    extracted = extract_date_from_text(markdown)
    if extracted:
        return extracted

    # 3. 文件修改时间兜底
    return date_from_file_mtime(parsed.path) or ""


class DocumentManager:
    """知识库文档管理器：上传 / 删除 / 列表。

    collection_name 决定该管理器绑定的 Milvus 集合（知识库），
    为空时使用默认知识库。
    """

    def __init__(self, kb: KnowledgeBase | None = None, collection_name: str = ""):
        self._kb = kb
        self._collection_name = collection_name
        self._storage = get_storage()
        self._parse_executor = ThreadPoolExecutor(
            max_workers=DOCUMENT_PARSE_WORKERS,
            thread_name_prefix="document-parser",
        )
        # 后台上传任务的强引用集合：done 后由回调自动丢弃，避免无限增长
        self._bg_tasks: set = set()
        # 流水线开关：关闭时回退到逐文件串行入库
        self._pipeline_enabled = INGEST_PIPELINE_ENABLED

    @property
    def kb(self) -> KnowledgeBase:
        if self._kb is None:
            self._kb = get_knowledge_base(self._collection_name or KB_COLLECTION_NAME)
        return self._kb

    def _parse_file(
        self,
        temp_path: Path,
        filename: str,
        ext: str,
        category: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> ParsedDocument:
        """解析文件并返回 ParsedDocument（同步，供线程池调用）。

        MinerU 统一解析所有支持格式；文本格式由 adapter 做无损直读标准化。
        """
        kb = self.kb
        source = filename
        title = Path(filename).stem

        # 分类作为 metadata 的强制字段统一合并
        merged_metadata = merge_category_into_metadata(metadata, category)

        if ext not in MINERU_SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"不支持的文件格式 {ext}，仅支持 {', '.join(MINERU_SUPPORTED_EXTENSIONS)}"
            )
        # 所有格式统一进入 MinerU parser；文本格式由 adapter 做无损直读标准化。
        parsed = kb.parse_document(
            temp_path, source=source, title=title, category=category,
            metadata=merged_metadata,
        )

        # 文档日期自动补全：显式 metadata > 内容提取 > 文件修改时间
        doc_date = resolve_document_date(parsed)
        if doc_date:
            merged_metadata["date"] = doc_date
            # ParsedDocument 是 frozen dataclass，用 replace 重建实例
            parsed = replace(parsed, metadata=merged_metadata)
            logger.info("文档日期解析：%s -> %s", filename, doc_date)

        return parsed

    # -----------------------------------------------------------------------
    # 异步上传（后台解析入库）
    # -----------------------------------------------------------------------

    async def upload_document_async(
        self,
        file: UploadFile,
        category: str = "",
    ) -> dict[str, str]:
        """接收上传文件，保存后立即返回 task_id，后台异步解析入库。

        Returns:
            ``{"task_id": "...", "filename": "..."}``
        """
        raw_filename = file.filename or "untitled.md"
        filename = Path(raw_filename).name
        if filename in {"", ".", ".."}:
            raise ValueError("文件名无效")
        ext = Path(filename).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"不支持的文件格式 {ext}，仅支持 {', '.join(SUPPORTED_EXTENSIONS)}"
            )

        content = await file.read()
        size_mb = len(content) / (1024 * 1024)
        if size_mb > MAX_UPLOAD_SIZE_MB:
            raise ValueError(
                f"文件大小 {size_mb:.1f}MB 超过限制 {MAX_UPLOAD_SIZE_MB}MB"
            )

        # 保存到系统临时目录
        temp_path = Path(tempfile.gettempdir()) / f".{uuid.uuid4().hex}.uploading{ext}"
        await asyncio.to_thread(temp_path.write_bytes, content)

        # 预计算内容哈希，供后台任务增量检查使用
        content_hash = hashlib.sha256(content).hexdigest()

        # 创建任务并启动处理
        tm = get_task_manager()
        task = tm.create(filename=filename)
        tm.update(task.id, status=TaskStatus.PROCESSING)

        if self._pipeline_enabled:
            # 流式路径：**入队即返回**，不创建后台任务、不等同批其它文档。
            # 队列满 / 流水线已关闭时抛 IngestionRejectedError，由路由转换为 503。
            try:
                self._submit_to_pipeline(
                    task.id, temp_path, filename, ext, content, content_hash, category
                )
            except Exception as exc:
                from finance_rag.src.core.exceptions import friendly_message

                tm.update(task.id, status=TaskStatus.FAILED, error=friendly_message(exc))
                _unlink_temp_file(temp_path)
                logger.error("异步入库提交失败：%s (task_id=%s): %s", filename, task.id, exc)
                raise
        else:
            # 串行回退路径：结果 Future 由后台处理投递「成功统计」或「失败异常」
            loop = asyncio.get_running_loop()
            result_future: asyncio.Future = loop.create_future()
            bg_task = asyncio.create_task(
                self._process_upload_background(
                    task.id,
                    temp_path,
                    filename,
                    ext,
                    content,
                    content_hash,
                    category,
                    result_future,
                )
            )
            # 持有强引用，便于观察/清理；done 后由回调自动丢弃，避免无限增长
            self._bg_tasks.add(bg_task)
            bg_task.add_done_callback(self._bg_tasks.discard)

        return {"task_id": task.id, "filename": filename}

    async def upload_documents_async(
        self,
        files: list[UploadFile],
        category: str = "",
    ) -> list[dict[str, str]]:
        """批量异步上传：每个文件立即返回 task_id，后台解析入库。

        Returns:
            ``[{"task_id", "filename"}, ...]``
        """
        results: list[dict[str, str]] = []
        for file in files:
            results.append(await self.upload_document_async(file, category=category))
        return results

    def _submit_to_pipeline(
        self,
        task_id: str,
        temp_path: Path,
        filename: str,
        ext: str,
        content: bytes,
        content_hash: str,
        category: str = "",
        result_future: asyncio.Future | None = None,
    ) -> dict[str, Any]:
        """把一份文档投递进常驻流水线（**同步、非阻塞、入队即返回**）。

        与 :meth:`_ingest_via_pipeline` 的区别：本方法不创建后台任务、不等待
        入库结束，投递成功即返回，因此调用方可以立刻响应上传请求；队列满时
        抛 :class:`~finance_rag.src.core.exceptions.IngestionQueueFullError`
        （路由层转 503），而不是阻塞在 ``await queue.put()`` 上。

        两点关键语义：

        * **增量检查在入队前完成**：重复文档不占用流水线容量；
        * **``category`` 随文档走**：常驻流水线没有调用级参数可回退，分类通过
          ``functools.partial`` 绑定到解析回调，否则会被静默丢弃。

        写库注入**两阶段**回调（``prepare_fn`` 锁外可并行 / ``commit_fn`` 锁内串行），
        使多篇文档的存储上传能并行，而向量库原子区仍保持同 collection 串行。

        Returns:
            命中增量跳过时返回终态结果字典（``skipped=True``），
            否则返回 ``{"task_id", "filename"}`` 表示已入队。
        """
        from finance_rag.src.services.ingestion_pipeline import (
            PipelineItem,
            get_ingestion_pipeline,
        )

        tm = get_task_manager()

        # 增量构建：基于内容哈希检查是否已入库
        if self.kb.get_fingerprint_store().is_unchanged(filename, content_hash):
            result = {
                "filename": filename,
                "source": filename,
                "title": Path(filename).stem,
                "chunk_count": 0,
                "parent_count": 0,
                "skipped": True,
            }
            tm.update(task_id, status=TaskStatus.COMPLETED, result=result)
            _unlink_temp_file(temp_path)
            if result_future is not None and not result_future.done():
                result_future.set_result(result)
            logger.info("异步：文件未变更，跳过入库：%s (task_id=%s)", filename, task_id)
            return result

        def _finish(tid: str, res: dict[str, Any] | None, err: BaseException | None) -> None:
            if err is not None:
                from finance_rag.src.core.exceptions import friendly_message

                tm.update(tid, status=TaskStatus.FAILED, error=friendly_message(err))
                logger.error("异步入库失败：%s (task_id=%s): %s", filename, tid, err)
                return
            tm.update(tid, status=TaskStatus.COMPLETED, result=res)
            logger.info("异步入库完成：%s (task_id=%s)", filename, tid)

        def _on_progress(tid: str, progress: dict[str, Any]) -> None:
            tm.update(tid, progress=progress)

        item = PipelineItem(
            task_id=task_id,
            temp_path=temp_path,
            filename=filename,
            ext=ext,
            content=content,
            content_hash=content_hash,
            result_future=result_future,
            kb=self.kb,
            # 分类随文档走：常驻队列只能通过 item 传递路由上下文
            parse_fn=functools.partial(self._parse_file, category=category),
            # 两阶段写库：阶段 A（存储上传）锁外可并行，阶段 B（向量库原子区）锁内串行
            prepare_fn=self._prepare_parsed_document,
            commit_fn=self._commit_parsed_document,
            finish_fn=_finish,
            on_progress=_on_progress,
        )
        get_ingestion_pipeline().submit(item)
        return {"task_id": task_id, "filename": filename}

    async def _ingest_via_pipeline(
        self,
        task_id: str,
        temp_path: Path,
        filename: str,
        ext: str,
        content: bytes,
        content_hash: str,
        category: str,
        result_future: asyncio.Future,
    ) -> None:
        """流水线入库（**等待版**）：投递后等待本文件结算。

        生产上传路径走 :meth:`_submit_to_pipeline`「入队即返回」；
        本方法保留给「调用方需要等到入库结束」的场景与既有测试。
        """
        tm = get_task_manager()
        try:
            self._submit_to_pipeline(
                task_id,
                temp_path,
                filename,
                ext,
                content,
                content_hash,
                category,
                result_future,
            )
        except Exception as exc:
            from finance_rag.src.core.exceptions import friendly_message

            tm.update(task_id, status=TaskStatus.FAILED, error=friendly_message(exc))
            if not result_future.done():
                result_future.set_exception(exc)
            _unlink_temp_file(temp_path)
            logger.error("异步入库失败：%s (task_id=%s): %s", filename, task_id, exc)
            return

        # 失败已通过任务终态与 Future 双向上报，这里只负责「等到结算」
        if not result_future.done():
            with contextlib.suppress(Exception):
                await result_future

    async def _process_upload_background(
        self,
        task_id: str,
        temp_path: Path,
        filename: str,
        ext: str,
        content: bytes,
        content_hash: str,
        category: str = "",
        result_future: asyncio.Future | None = None,
    ) -> None:
        """逐文件串行入库（``INGEST_PIPELINE_ENABLED=false`` 时的回退路径）。

        处理流程与流水线一致（增量检查 → 解析 → 存储 → 写库），
        只是不与其他文档并行。
        """
        tm = get_task_manager()
        try:
            result = await self._process_upload_serially(
                task_id, temp_path, filename, ext, content, content_hash, category
            )
            if result_future is not None and not result_future.done():
                result_future.set_result(result)
            logger.info("异步入库完成：%s (task_id=%s)", filename, task_id)
        except Exception as exc:
            from finance_rag.src.core.exceptions import friendly_message

            tm.update(task_id, status=TaskStatus.FAILED, error=friendly_message(exc))
            if result_future is not None and not result_future.done():
                result_future.set_exception(exc)
            logger.error("异步入库失败：%s (task_id=%s): %s", filename, task_id, exc)
        finally:
            _unlink_temp_file(temp_path)

    async def _process_upload_serially(
        self,
        task_id: str,
        temp_path: Path,
        filename: str,
        ext: str,
        content: bytes,
        content_hash: str,
        category: str,
    ) -> dict[str, Any]:
        """回退路径的同步处理体：解析 + 持久化，返回任务结果字典。"""
        tm = get_task_manager()

        # 增量构建：基于内容哈希检查是否已入库
        if self.kb.get_fingerprint_store().is_unchanged(filename, content_hash):
            result = {
                "filename": filename,
                "source": filename,
                "title": Path(filename).stem,
                "chunk_count": 0,
                "parent_count": 0,
                "skipped": True,
            }
            tm.update(task_id, status=TaskStatus.COMPLETED, result=result)
            logger.info("异步：文件未变更，跳过入库：%s (task_id=%s)", filename, task_id)
            return result

        # 统一解析（MinerU 唯一路径）
        # 使用自定义 _parse_executor 以真正受 DOCUMENT_PARSE_WORKERS 限制并发
        loop = asyncio.get_running_loop()
        parsed = await loop.run_in_executor(
            self._parse_executor,
            self._parse_file,
            temp_path,
            filename,
            ext,
            category,
        )

        result = await self._persist_parsed_document(parsed, content_hash)
        tm.update(task_id, status=TaskStatus.COMPLETED, result=result)
        return result

    async def _persist_parsed_document(
        self,
        parsed: ParsedDocument,
        content_hash: str | None,
        dense_vectors: list[list[float]] | None = None,
    ) -> dict[str, Any]:
        """持久化已解析文档（**单回调兼容入口**）：阶段 A + 阶段 B 顺序执行。

        入库流水线的生产路径走**两阶段**（:meth:`_prepare_parsed_document` 锁外、
        :meth:`_commit_parsed_document` 锁内），以便存储上传与其他文档并行；
        本方法是二者的组合，供串行回退路径与既有调用方使用，语义与旧版一致。

        Args:
            parsed: 解析 + 切块结果。
            content_hash: 上传内容哈希（透传指纹存储，避免增量去重失效）。
            dense_vectors: 流水线在 Embedding 线程池中预计算的稠密向量；
                None 时由 :meth:`KnowledgeBase.add_parsed_document` 内联补齐。
        """
        context = await self._prepare_parsed_document(parsed, content_hash, dense_vectors)
        return await self._commit_parsed_document(context)

    async def _prepare_parsed_document(
        self,
        parsed: ParsedDocument,
        content_hash: str | None,
        dense_vectors: list[list[float]] | None = None,
    ) -> dict[str, Any]:
        """写库**阶段 A（锁外，可与其他文档并行）**：文档产物落存储后端。

        原文件、解析提取的嵌入图片、解析后的 .md 依次上传。这一整段是网络 IO
        （本地文件系统或 OSS/S3），不持有写锁，因此多篇文档可以并行上传。

        Returns:
            提交上下文，原样交给 :meth:`_commit_parsed_document`。
        """
        storage = self._storage
        filename = parsed.source

        # 上传原始文件到存储后端
        await storage.upload(f"docs/{filename}", parsed.path.read_bytes())

        # 上传解析提取的嵌入图片到存储后端，并清理临时目录
        await self._upload_extracted_images(parsed, storage)

        # 解析后的 .md 文件直接上传到存储后端
        if parsed.chunks.markdown is not None:
            md_key = f"docs/{Path(filename).stem}.md"
            await storage.upload(md_key, parsed.chunks.markdown.encode("utf-8"))

        return {
            "parsed": parsed,
            "content_hash": content_hash,
            "dense_vectors": dense_vectors,
        }

    async def _commit_parsed_document(self, context: dict[str, Any]) -> dict[str, Any]:
        """写库**阶段 B（写锁内，同 collection 串行）**：向量库原子区 + 指纹。

        由入库流水线在写锁内调用。这一段必须串行——
        :meth:`KnowledgeBase.add_parsed_document` 的文档间 SimHash 去重是
        "先查再插"，并发会让近似重复文档同时通过检查；同 source 的
        「删旧行 + 插新行」并发也会互相删除。

        Args:
            context: :meth:`_prepare_parsed_document` 的返回值。
        """
        parsed = context["parsed"]
        filename = parsed.source
        stored = await asyncio.to_thread(
            self.kb.add_parsed_document,
            parsed,
            None,
            content_hash=context.get("content_hash"),
            dense_vectors=context.get("dense_vectors"),
        )
        return {
            "filename": filename,
            "source": stored["source"],
            "title": stored["title"],
            "chunk_count": stored["chunk_count"],
            "parent_count": stored["parent_count"],
        }

    def _store_parsed_document(
        self,
        parsed: ParsedDocument,
        content_hash: str | None = None,
        dense_vectors: list[list[float]] | None = None,
    ) -> dict[str, Any]:
        """同步写库（对象存储事件路径）：直接调用 ``add_parsed_document``。

        ``content_hash`` 传入时透传给指纹存储，避免增量去重失效。
        ``dense_vectors`` 由调用方预计算时传入，否则内联嵌入。

        无需在此加锁：``KnowledgeBase.add_parsed_document`` 内部持有
        集合级写锁，与入库流水线的写入阶段共用同一把锁。
        """
        return self.kb.add_parsed_document(
            parsed, content_hash=content_hash, dense_vectors=dense_vectors
        )

    @staticmethod
    async def _upload_extracted_images(
        parsed: ParsedDocument, storage: Any
    ) -> None:
        """把解析阶段提取的嵌入图片上传到存储后端。

        图片来自 ``parsed.chunks.images``（``ImageAsset``，已在内存中，无临时文件）；
        单张失败仅告警不阻断入库。整体文本描述由视觉模型在入库链路上生成，
        这里只负责把原始图片落到对象存储，供页码溯源与「查看原图」使用。
        """
        images = getattr(parsed.chunks, "images", None) or []
        if not images:
            return

        from finance_rag.src.rag.ingestion.pdf_assets import image_object_key

        uploaded = 0
        for asset in images:
            key = image_object_key(parsed.source, asset)
            try:
                await storage.upload(key, asset.data)
                uploaded += 1
            except Exception as exc:
                logger.warning("嵌入图片上传失败 %s: %s", key, exc)

        if uploaded:
            logger.info("嵌入图片上传完成：%s（%d 张）", parsed.source, uploaded)

    async def process_object_event(
        self, temp_path: Path, filename: str, version_id: str
    ) -> dict[str, Any]:
        """Process an already downloaded object and persist its version metadata."""
        try:
            parsed = await asyncio.to_thread(
                self._parse_file, temp_path, filename, temp_path.suffix.lower()
            )
            result = await asyncio.to_thread(self._store_parsed_document, parsed, None)
            from finance_rag.src.infrastructure.relational_db.document_manifest import DocumentManifestRepository
            repo = DocumentManifestRepository()
            repo.upsert_document({
                "document_id": f"{TENANT_ID}:{filename}", "tenant_id": TENANT_ID,
                "source": filename, "title": parsed.title,
                "current_version_id": version_id, "status": "active",
                "category": str((parsed.metadata or {}).get("category", "")),
                "date": str((parsed.metadata or {}).get("date", "")),
                "object_key": f"docs/{filename}",
            })
            repo.create_version({
                "version_id": version_id, "document_id": f"{TENANT_ID}:{filename}",
                "file_sha256": hashlib.sha256(temp_path.read_bytes()).hexdigest(),
                "status": "current", "parser_version": "mineru",
                "chunk_count": result["chunk_count"],
            })
            return result
        finally:
            temp_path.unlink(missing_ok=True)

    def soft_delete_source(self, source: str) -> dict[str, Any]:
        """Mark document metadata deleted and exclude its vectors from normal use."""
        from finance_rag.src.infrastructure.relational_db.document_manifest import DocumentManifestRepository
        repo = DocumentManifestRepository()
        document_id = f"{TENANT_ID}:{source}"
        existing = repo.get_document(document_id)
        if existing:
            repo.upsert_document({"document_id": document_id, "status": "deleted", "deleted_at": datetime.now()})
        return self.kb.soft_delete_document(source)

    def delete_document(self, source: str, version: str | None = None) -> dict[str, Any]:
        """删除文档的 Milvus 向量记录和存储中的文件。

        ``version=None`` 时删除该文档的全部版本；指定版本时仅删除该版本。
        """
        storage = self._storage

        # 删除存储中的原始文件
        doc_key = f"docs/{source}"
        _safe_delete(storage, doc_key)

        # 删除解析后的 .md 文件（仅 PDF：上传时生成了 .stem.md）
        md_key = f"docs/{Path(source).stem}.md"
        if md_key != doc_key:
            _safe_delete(storage, md_key)

        return self.kb.remove_document(source, version=version)

    def list_documents(self, include_versions: bool = False) -> list[dict[str, Any]]:
        """列出知识库所有文档；``include_versions=True`` 时含历史版本明细。"""
        return self.kb.list_documents(include_versions=include_versions)

    def get_stats(self) -> dict[str, Any]:
        """知识库统计。"""
        return self.kb.get_stats()


# ---------------------------------------------------------------------------
# 单例（按知识库集合缓存）
# ---------------------------------------------------------------------------

_DM_INSTANCES: dict[str, DocumentManager] = {}
_DM_LOCK = threading.Lock()


def get_document_manager(collection: str = "") -> DocumentManager:
    """获取（并缓存）指定知识库的 DocumentManager 单例（线程安全）。

    collection 为空时返回默认知识库的管理器。
    """
    name = collection or KB_COLLECTION_NAME
    dm = _DM_INSTANCES.get(name)
    if dm is None:
        with _DM_LOCK:
            dm = _DM_INSTANCES.get(name)
            if dm is None:
                dm = DocumentManager(collection_name=name)
                _DM_INSTANCES[name] = dm
    return dm
