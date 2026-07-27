"""文档上传/解析/管理模块（Agentic RAG）。

负责接收前端上传的文件，保存到知识库目录，调用 :class:`KnowledgeBase`
完成切块/嵌入/入库，支持文档列表与删除。
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from fastapi import UploadFile

from .config import DOCUMENT_PARSE_WORKERS, MAX_UPLOAD_SIZE_MB, UPLOAD_DIR
from .knowledge_base import KnowledgeBase, ParsedDocument, get_knowledge_base

logger = logging.getLogger(__name__)

# 支持的文件格式
SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf"}

# 上传并发锁（避免同时写 Milvus 导致冲突）
_upload_lock = threading.Lock()


class DocumentManager:
    """知识库文档管理器：上传 / 删除 / 列表。"""

    def __init__(
        self,
        kb: KnowledgeBase | None = None,
        upload_dir: str | None = None,
    ):
        self._kb = kb
        self._upload_dir = Path(upload_dir) if upload_dir else Path(UPLOAD_DIR)
        self._upload_dir.mkdir(parents=True, exist_ok=True)
        self._parse_executor = ThreadPoolExecutor(
            max_workers=DOCUMENT_PARSE_WORKERS,
            thread_name_prefix="document-parser",
        )

    @property
    def kb(self) -> KnowledgeBase:
        if self._kb is None:
            self._kb = get_knowledge_base()
        return self._kb

    async def upload_document(self, file: UploadFile) -> dict[str, Any]:
        """接收上传文件 → 保存 → 入库。

        Returns:
            ``{"filename", "source", "title", "chunk_count", "parent_count"}``
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

        # 读取并校验大小
        content = await file.read()
        size_mb = len(content) / (1024 * 1024)
        if size_mb > MAX_UPLOAD_SIZE_MB:
            raise ValueError(
                f"文件大小 {size_mb:.1f}MB 超过限制 {MAX_UPLOAD_SIZE_MB}MB"
            )

        # 先写入同目录临时文件，入库成功后再原子替换正式文件。
        save_path = self._upload_dir / filename
        temp_path = self._upload_dir / f".{uuid.uuid4().hex}.uploading{ext}"
        loop = asyncio.get_running_loop()
        kb = self.kb
        await loop.run_in_executor(self._parse_executor, temp_path.write_bytes, content)

        try:
            parsed = await loop.run_in_executor(
                self._parse_executor,
                lambda: kb.parse_document(
                    temp_path,
                    source=filename,
                    title=Path(filename).stem,
                ),
            )
            markdown_temp_path: Path | None = None
            markdown_save_path: Path | None = None
            if parsed.chunks.markdown is not None:
                markdown_temp_path = (
                    self._upload_dir / f".{uuid.uuid4().hex}.md.uploading"
                )
                markdown_save_path = save_path.with_suffix(".md")
                await loop.run_in_executor(
                    self._parse_executor,
                    markdown_temp_path.write_text,
                    parsed.chunks.markdown,
                    "utf-8",
                )
            result = await asyncio.to_thread(
                self._store_parsed_document,
                parsed,
                temp_path,
                save_path,
                markdown_temp_path,
                markdown_save_path,
            )
        except Exception:
            temp_path.unlink(missing_ok=True)
            if "markdown_temp_path" in locals() and markdown_temp_path is not None:
                markdown_temp_path.unlink(missing_ok=True)
            raise

        logger.info("文件已保存并入库：%s（%.2f MB）", save_path, size_mb)

        return {
            "filename": filename,
            "source": result["source"],
            "title": result["title"],
            "chunk_count": result["chunk_count"],
            "parent_count": result["parent_count"],
            "size_mb": round(size_mb, 2),
        }

    async def upload_documents(
        self,
        files: list[UploadFile],
    ) -> list[dict[str, Any] | BaseException]:
        """在线程池中并行解析一批文档，并收集每个文件的结果。"""
        tasks = [self.upload_document(file) for file in files]
        return await asyncio.gather(*tasks, return_exceptions=True)

    def _store_parsed_document(
        self,
        parsed: ParsedDocument,
        temp_path: Path,
        save_path: Path,
        markdown_temp_path: Path | None = None,
        markdown_save_path: Path | None = None,
    ) -> dict[str, Any]:
        """串行执行嵌入、Milvus 写入和正式文件原子替换。"""
        with _upload_lock:
            result = self.kb.add_parsed_document(parsed)
            os.replace(temp_path, save_path)
            if markdown_temp_path is not None and markdown_save_path is not None:
                os.replace(markdown_temp_path, markdown_save_path)
            return result

    def delete_document(self, source: str) -> dict[str, Any]:
        """删除文档的 Milvus 向量记录，保留本地原文件。"""
        with _upload_lock:
            return self.kb.remove_document(source)

    def list_documents(self) -> list[dict[str, Any]]:
        """列出知识库所有文档。"""
        return self.kb.list_documents()

    def get_stats(self) -> dict[str, Any]:
        """知识库统计。"""
        return self.kb.get_stats()


# ---------------------------------------------------------------------------
# 单例
# ---------------------------------------------------------------------------

_DM_INSTANCE: DocumentManager | None = None


def get_document_manager() -> DocumentManager:
    """获取全局 DocumentManager 单例。"""
    global _DM_INSTANCE
    if _DM_INSTANCE is None:
        _DM_INSTANCE = DocumentManager()
    return _DM_INSTANCE
