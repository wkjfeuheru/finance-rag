"""切块阶段的进程池 worker（纯 Python CPU 密集型工作）。

背景
----
MinerU 解析之后的所有文本处理——规则清洗、段落级 SimHash 去重、标题结构注入、
父子层级切分——都是**纯 Python** 实现，受 GIL 限制无法用线程池并行。
实测 ``HierarchicalChunker._build_chunks`` 处理 21 万字符文档耗时约 3.4s，
是入库链路上可量化的真实瓶颈，因此单独放进 **进程池**。

边界与约束
----------
* MinerU 解析本身**不在此模块**：它加载 GB 级 ONNX/GPU 权重，进程池化会造成
  内存成倍增长，仍留在父进程的解析线程池中执行。
* 本模块的入口函数与返回值必须可 pickle：入参为纯字符串元组，
  返回值为 :class:`~finance_rag.src.rag.ingestion.chunker.DoclingChunks`
  （其 ``chunks`` 为 ``langchain_core.documents.Document``，已验证可序列化）。
* 进程池使用 ``spawn`` 上下文：规避 fork 与 CUDA/ONNXRuntime 的不兼容。
* 每个子进程独立持有自己的切块器实例（语义分块时即独立加载句向量模型）。
"""

from __future__ import annotations

import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

from finance_rag.src.core import config
from finance_rag.src.core.config import resolve_chunk_workers
from finance_rag.src.rag.ingestion.chunker import DoclingChunks

logger = logging.getLogger(__name__)

# 进程池单例（惰性创建：ProcessPoolExecutor 首次 submit 时才真正起子进程）
_chunk_pool: ProcessPoolExecutor | None = None
# 池不可用标记：受限环境（禁止创建子进程/管道）下避免每次入库都重试创建
_pool_unavailable = False

# 切块任务载荷：(markdown, source, title, category, is_pdf, blocks)
# blocks 为解析层的分页文本块元组（ContentBlock 可 pickle），
# 用于把切块结果归属到页码；图片字节**不进**载荷，避免大对象过 IPC。
ChunkPayload = tuple[str, str, str, str, bool, tuple]

# 模块级切块器（每进程一份，惰性创建）
_chunker = None


def _get_chunker():
    """获取进程内的切块器实例（按配置开关返回层级 / 语义切块器）。"""
    global _chunker
    if _chunker is None:
        from finance_rag.src.rag.ingestion.chunker import get_chunker

        _chunker = get_chunker()
    return _chunker


def chunk_markdown_inline(
    markdown: str,
    *,
    source: str,
    title: str = "",
    category: str = "",
    is_pdf: bool = False,
    blocks: tuple = (),
) -> DoclingChunks:
    """在当前进程内完成切块（小文档与进程池不可用时的降级路径）。

    与 :func:`chunk_document` 共用同一 ``_build_chunks`` 实现，
    因此两条路径的切块结果逐字节一致。
    """
    return _get_chunker()._build_chunks(
        markdown,
        source,
        title or source,
        is_pdf=is_pdf,
        category=category,
        blocks=tuple(blocks or ()),
    )


def chunk_document(payload: ChunkPayload) -> DoclingChunks:
    """进程池入口：对已解析出的 markdown 做清洗 + 层级切块。

    Args:
        payload: ``(markdown, source, title, category, is_pdf, blocks)``；
            兼容历史 5 元组（无 blocks，页码归属退化为 0）。

    Returns:
        切块结果；进程池路径与 :func:`chunk_markdown_inline` 结果一致。
    """
    markdown, source, title, category, is_pdf = payload[:5]
    blocks = payload[5] if len(payload) > 5 else ()
    return chunk_markdown_inline(
        markdown,
        source=source,
        title=title,
        category=category,
        is_pdf=is_pdf,
        blocks=blocks,
    )


def use_process_pool(markdown: str) -> bool:
    """判断该文档是否值得走进程池（过短文档的进程往返开销大于收益）。"""
    return config.INGEST_CHUNK_MIN_CHARS <= len(markdown)


def get_chunk_pool() -> ProcessPoolExecutor:
    """获取（并缓存）切块进程池单例。

    ``spawn`` 上下文与 Windows 默认一致，在 Linux/Docker 下同样可用且更安全
    （不会把父进程已加载的 CUDA/ORT 状态复制进子进程）。

    Raises:
        RuntimeError: 进程池已被标记为不可用（见 :func:`mark_chunk_pool_unavailable`）。
    """
    global _chunk_pool
    if _pool_unavailable:
        raise RuntimeError("切块进程池在当前环境不可用，已降级为内联切块")
    if _chunk_pool is None:
        workers = resolve_chunk_workers()
        ctx = multiprocessing.get_context("spawn")
        _chunk_pool = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=ctx,
        )
        logger.info("切块进程池已创建：workers=%d（spawn）", workers)
    return _chunk_pool


def mark_chunk_pool_unavailable() -> None:
    """标记进程池不可用并丢弃当前池（受限环境降级为内联切块，不再重试）。"""
    global _pool_unavailable, _chunk_pool
    _pool_unavailable = True
    pool, _chunk_pool = _chunk_pool, None
    if pool is not None:
        try:
            pool.shutdown(wait=False, cancel_futures=True)
        except Exception:  # pragma: no cover - 防御：失效池关闭可忽略
            pass
    logger.warning("切块进程池已标记为不可用，后续文档统一走内联切块")


def is_chunk_pool_unavailable() -> bool:
    """进程池是否已被标记为不可用（供测试与诊断使用）。"""
    return _pool_unavailable


def shutdown_chunk_pool(wait: bool = False) -> None:
    """关闭切块进程池。

    必须在服务生命周期收尾时显式调用：``ProcessPoolExecutor`` 的 atexit/GC
    清理会 ``join`` 子进程，把清理留到解释器退出阶段可能直接卡死进程。
    """
    global _chunk_pool
    pool, _chunk_pool = _chunk_pool, None
    if pool is None:
        return
    try:
        pool.shutdown(wait=wait, cancel_futures=True)
        logger.info("切块进程池已关闭")
    except Exception as exc:  # pragma: no cover - 防御：关闭失败不阻断退出
        logger.warning("切块进程池关闭失败：%s", exc)


__all__ = [
    "ChunkPayload",
    "chunk_document",
    "chunk_markdown_inline",
    "get_chunk_pool",
    "is_chunk_pool_unavailable",
    "mark_chunk_pool_unavailable",
    "shutdown_chunk_pool",
    "use_process_pool",
]
