"""PDF 解析模块（子进程隔离）。

将 PDF 解析放在独立子进程中运行，避免解析库内存泄漏影响 API 进程。
支持两种模式：
- 作为模块导入：使用 ``parse_pdf_in_subprocess()``
- 作为子进程入口：``python -m finance_rag.src.rag.ingestion.pdf_parser INPUT OUTPUT SOURCE TITLE``
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import traceback
import uuid
from pathlib import Path
from typing import Any, Sequence

from langchain_core.documents import Document

from finance_rag.src.rag.ingestion.chunker import DoclingChunks, ParentChunk

logger = logging.getLogger(__name__)


# ===========================================================================
# 异常定义
# ===========================================================================

class PdfParseProcessError(ValueError):
    """The PDF parser process failed or returned an invalid result."""


class PdfParseTimeoutError(PdfParseProcessError):
    """The PDF parser exceeded its hard deadline and was terminated."""


# ===========================================================================
# 结果加载
# ===========================================================================

def _load_result(result_path: Path, source: str) -> DoclingChunks:
    try:
        payload: Any = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PdfParseProcessError(f"PDF 解析子进程未返回有效结果：{source}") from exc

    if not isinstance(payload, dict) or payload.get("version") not in (1, 2):
        raise PdfParseProcessError(f"PDF 解析子进程结果格式无效：{source}")
    markdown = payload.get("markdown")
    raw_chunks = payload.get("chunks")
    if markdown is not None and not isinstance(markdown, str):
        raise PdfParseProcessError(f"PDF 解析子进程结果格式无效：{source}")
    if not isinstance(raw_chunks, list) or not raw_chunks:
        raise PdfParseProcessError(f"PDF 解析子进程未生成有效分块：{source}")

    documents: list[Document] = []
    for raw_chunk in raw_chunks:
        if not isinstance(raw_chunk, dict):
            raise PdfParseProcessError(f"PDF 解析子进程结果格式无效：{source}")
        content = raw_chunk.get("page_content")
        metadata = raw_chunk.get("metadata")
        if not isinstance(content, str) or not content.strip() or not isinstance(metadata, dict):
            raise PdfParseProcessError(f"PDF 解析子进程结果格式无效：{source}")
        documents.append(Document(page_content=content, metadata=metadata))

    parents: list[ParentChunk] = []
    raw_parents = payload.get("parents", [])
    if isinstance(raw_parents, list):
        for rp in raw_parents:
            if isinstance(rp, dict):
                parents.append(
                    ParentChunk(
                        id=rp.get("id", ""),
                        heading=rp.get("heading", ""),
                        content=rp.get("content", ""),
                        source=rp.get("source", ""),
                        title=rp.get("title", ""),
                    )
                )

    # 嵌入图片元数据（可选字段，旧 worker 无此字段时为空列表）
    images: list[dict] = []
    raw_images = payload.get("images")
    if isinstance(raw_images, list):
        images = [img for img in raw_images if isinstance(img, dict)]

    return DoclingChunks(chunks=documents, markdown=markdown, parents=parents, images=images)


# ===========================================================================
# 子进程启动
# ===========================================================================

def run_parser_process(
    command: Sequence[str],
    *,
    result_path: Path,
    source: str,
    timeout_seconds: float,
) -> DoclingChunks:
    """Execute a parser command and reconstruct its schema-validated result."""
    try:
        try:
            completed = subprocess.run(
                list(command),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_seconds,
                check=False,
                shell=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise PdfParseTimeoutError(
                f"PDF 解析超过 {timeout_seconds:g} 秒，已终止解析子进程：{source}"
            ) from exc
        if completed.returncode != 0:
            stderr_tail = (completed.stderr or "").strip()[-2000:]
            if stderr_tail:
                logger.error("PDF parser stderr (%s): %s", source, stderr_tail)
            # 提取 traceback 最后一行（即真实异常），并入错误消息便于直接定位
            detail = ""
            for line in reversed((completed.stderr or "").splitlines()):
                line = line.strip()
                if line and not line.startswith("File "):
                    detail = line[:300]
                    break
            msg = f"PDF 解析子进程异常退出（退出码 {completed.returncode}）：{source}"
            if detail:
                msg += f"；原因：{detail}"
            raise PdfParseProcessError(msg)
        return _load_result(result_path, source)
    finally:
        result_path.unlink(missing_ok=True)
        result_path.with_suffix(f"{result_path.suffix}.tmp").unlink(missing_ok=True)


def parse_pdf_in_subprocess(
    path: Path,
    *,
    source: str,
    title: str,
    timeout_seconds: float,
) -> DoclingChunks:
    """Parse one PDF with the current interpreter in a disposable process."""
    result_path = path.with_name(f".{uuid.uuid4().hex}.pdf-result.json")
    command = [
        sys.executable,
        "-m",
        "finance_rag.src.rag.ingestion.pdf_parser",
        str(path),
        str(result_path),
        source,
        title,
    ]
    return run_parser_process(
        command,
        result_path=result_path,
        source=source,
        timeout_seconds=timeout_seconds,
    )


# ===========================================================================
# 子进程 worker 入口
# ===========================================================================

def _serialize_chunks(chunks: DoclingChunks) -> dict:
    return {
        "version": 2,
        "markdown": chunks.markdown,
        "chunks": [
            {
                "page_content": chunk.page_content,
                "metadata": dict(chunk.metadata),
            }
            for chunk in chunks.chunks
        ],
        "parents": [
            {
                "id": p.id,
                "heading": p.heading,
                "content": p.content,
                "source": p.source,
                "title": p.title,
            }
            for p in chunks.parents
        ],
        "images": chunks.images,
    }


def _worker_main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 4:
        print("usage: python -m finance_rag.src.rag.ingestion.pdf_parser INPUT OUTPUT SOURCE TITLE", file=sys.stderr)
        return 2

    input_path, output_path, source, title = args
    output = Path(output_path)
    temporary_output = output.with_suffix(f"{output.suffix}.tmp")
    try:
        os.environ["DEEPSEEK_API_KEY"] = ""
        from finance_rag.src.rag.ingestion.chunker import get_chunker

        chunks = get_chunker().parse_and_chunk(
            Path(input_path),
            source=source,
            title=title,
        )
        temporary_output.write_text(
            json.dumps(_serialize_chunks(chunks), ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(temporary_output, output)
        return 0
    except Exception:
        traceback.print_exc(file=sys.stderr)
        return 1
    finally:
        temporary_output.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(_worker_main())
