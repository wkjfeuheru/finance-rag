"""入库流水线（进程单例常驻队列 + 分阶段 Executor）单元测试。

覆盖：
* 切块进程池与内联切块结果一致（核心不变量）；
* 阶段重叠（首个文档的 Embedding 早于最后一个文档解析结束）；
* 流式提交（``submit`` 入队即返回，不等同批其它文档）；
* 背压（队列有界，满时 ``submit`` 快速失败而非阻塞调用方）；
* 单文件失败隔离与临时文件清理；
* 阶段 worker 自愈（异常不终止 worker）与 supervisor 兜底重启；
* 关闭时结算全部在途文档，不让调用方永久等待；
* 随文档走的路由上下文（``kb`` 按文档路由，不串集合）；
* 回退路径（``INGEST_PIPELINE_ENABLED=false``）；
* 切块进程池不可用时的内联降级；
* ``KnowledgeBase`` 预计算向量复用与数量校验。
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from finance_rag.src.core import config
from finance_rag.src.core.exceptions import IngestionQueueFullError
from finance_rag.src.infrastructure.vector_store.milvus_kb import (
    KnowledgeBase,
    ParsedDocument,
)
from finance_rag.src.rag.ingestion import chunk_worker
from finance_rag.src.rag.ingestion.chunker import DoclingChunks
from finance_rag.src.services import ingestion_pipeline as pipeline_module
from finance_rag.src.services.ingestion_pipeline import (
    IngestionPipeline,
    PipelineItem,
)


@pytest.fixture(autouse=True)
def _shutdown_pipelines_after_test():
    """用例收尾：结算并关闭测试期间创建的常驻流水线实例。

    常驻 worker 在用例结束后仍会阻塞在 ``queue.get()`` 上，
    不关闭会跨用例累积任务与线程。
    """
    yield
    pipeline_module.shutdown_all_pipelines()

# ---------------------------------------------------------------------------
# 测试替身
# ---------------------------------------------------------------------------

# 上传内容样例（避免在源码中直接写非 ASCII 字节字面量）
_CONTENT_BYTES = "示例文档内容".encode()


def _big_markdown(marker: str, sections: int = 60) -> str:
    """生成指定规模的 markdown（sections 段），用于覆盖切块与嵌入链路。"""
    block = (
        f"## {marker} 小节标题\n"
        f"{marker} 正文内容，用于构造足够长的文档以覆盖切块与嵌入链路。\n\n"
    )
    return f"# {marker} 文档\n\n" + block * sections


def _huge_markdown(marker: str = "长文档") -> str:
    """生成超过 ``INGEST_CHUNK_MIN_CHARS`` 的 markdown（触发进程池路径）。"""
    from finance_rag.src.core import config

    return _big_markdown(marker, sections=config.INGEST_CHUNK_MIN_CHARS // 20 + 400)


def _make_parsed(source: str, markdown: str) -> ParsedDocument:
    """构造「仅解析、未切块」的 ParsedDocument（markdown 挂在 chunks 上）。"""
    return ParsedDocument(
        path=Path(source),
        source=source,
        title=Path(source).stem,
        chunks=DoclingChunks(markdown=markdown),
        metadata={"category": ""},
    )


@dataclass
class FakeKnowledgeBase:
    """最小 KnowledgeBase 替身：只实现流水线用到的三个方法。"""

    chunk_count: int = 4
    embed_delay: float = 0.0
    fetch_delay: float = 0.0
    existing: dict[str, list[float]] = field(default_factory=dict)
    embed_calls: list[list[str]] = field(default_factory=list)
    embedded_count: int = 0

    def fetch_existing_vectors(self, source: str) -> dict[str, list[float]]:
        if self.fetch_delay:
            time.sleep(self.fetch_delay)
        return dict(self.existing)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self.embed_delay:
            time.sleep(self.embed_delay)
        self.embed_calls.append(list(texts))
        self.embedded_count += len(texts)
        return [[0.1, 0.2, 0.3] for _ in texts]


def _workdir() -> Path:
    """仓库内的工作目录（受限环境下系统临时目录可能不可写）。"""
    path = Path(__file__).resolve().parent / ".tmp"
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass
class StageRecorder:
    """记录各阶段时间点，用于验证阶段重叠与背压。"""

    timeline: list[tuple[str, str, float]] = field(default_factory=list)
    chunk_running: int = 0
    max_concurrent_chunk: int = 0
    persist_failures: set[str] = field(default_factory=set)

    def record(self, stage: str, name: str) -> None:
        self.timeline.append((stage, name, time.perf_counter()))

    def first(self, stage: str) -> float:
        return min(t for s, _, t in self.timeline if s == stage)

    def last(self, stage: str) -> float:
        return max(t for s, _, t in self.timeline if s == stage)


def _raise_broken_process_pool():
    """模拟受限环境：创建切块进程池失败。"""
    from concurrent.futures.process import BrokenProcessPool

    raise BrokenProcessPool("模拟进程池不可用")


def _finish_recorder(results: dict[str, Any], errors: dict[str, BaseException]):
    def _finish(task_id: str, result: dict[str, Any] | None, err: BaseException | None) -> None:
        if err is not None:
            errors[task_id] = err
        else:
            results[task_id] = result

    return _finish


def _make_items(recorder: StageRecorder, count: int, prefix: str = "doc") -> list[PipelineItem]:
    items: list[PipelineItem] = []
    for index in range(count):
        name = f"{prefix}-{index}.md"
        path = Path(f"unused-{name}")
        items.append(
            PipelineItem(
                task_id=f"task-{prefix}-{index}",
                temp_path=path,
                filename=name,
                ext=".md",
                content=b"",
                content_hash=f"hash-{prefix}-{index}",
                result_future=asyncio.get_running_loop().create_future(),
            )
        )
    return items


# ---------------------------------------------------------------------------
# 1. 切块一致性（进程池 == 内联）
# ---------------------------------------------------------------------------


def test_process_pool_chunking_matches_inline(monkeypatch):
    """进程池切块与内联切块的结果必须逐块一致（id / 正文 / 标题路径）。"""
    markdown = _big_markdown("一致性")
    inline = chunk_worker.chunk_markdown_inline(
        markdown, source="a.md", title="a", category="", is_pdf=False
    )

    class _InlineExecutor:
        """替代真实进程池：在独立线程中执行同一 worker 函数。"""

        def submit(self, fn, payload):
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(fn, payload)

    monkeypatch.setattr(pipeline_module, "get_chunk_pool", lambda: _InlineExecutor())
    pool_result = chunk_worker.chunk_document((markdown, "a.md", "a", "", False))

    assert [c.metadata["id"] for c in pool_result.chunks] == [
        c.metadata["id"] for c in inline.chunks
    ]
    assert [c.page_content for c in pool_result.chunks] == [
        c.page_content for c in inline.chunks
    ]
    assert [p.id for p in pool_result.parents] == [p.id for p in inline.parents]


def test_use_process_pool_threshold():
    """短文档不进程池化，长文档（超过 INGEST_CHUNK_MIN_CHARS）进程池化。"""
    from finance_rag.src.core import config

    assert chunk_worker.use_process_pool("短文本") is False
    long_markdown = _huge_markdown("阈值")
    assert len(long_markdown) >= config.INGEST_CHUNK_MIN_CHARS
    assert chunk_worker.use_process_pool(long_markdown) is True


# ---------------------------------------------------------------------------
# 2. 阶段重叠：切块完立即进入 Embedding，不等整批解析结束
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_embedding_starts_before_all_parsing_finishes(monkeypatch):
    """核心收益验证：首个文档的 Embedding 应早于最后一个文档解析结束。"""
    recorder = StageRecorder()
    parse_delays = {"doc-0.md": 0.30, "doc-1.md": 0.40, "doc-2.md": 0.50}

    def parse_fn(temp_path: Path, filename: str, ext: str) -> ParsedDocument:
        time.sleep(parse_delays[filename])
        recorder.record("parse", filename)
        return _make_parsed(filename, _big_markdown(filename, sections=5))

    def chunk_fn(markdown, *, source, title, category, is_pdf, blocks=()):
        recorder.chunk_running += 1
        recorder.max_concurrent_chunk = max(
            recorder.max_concurrent_chunk, recorder.chunk_running
        )
        try:
            time.sleep(0.05)
            recorder.record("chunk", source)
            return chunk_worker.chunk_markdown_inline(
                markdown, source=source, title=title, category=category, is_pdf=is_pdf,
                blocks=blocks,
            )
        finally:
            recorder.chunk_running -= 1

    monkeypatch.setattr(pipeline_module, "chunk_markdown_inline", chunk_fn)

    fake_kb = FakeKnowledgeBase()
    original_embed = fake_kb.embed_documents

    def embed_documents(texts: list[str]) -> list[list[float]]:
        recorder.record("embed", texts[0][:12] if texts else "")
        return original_embed(texts)

    monkeypatch.setattr(fake_kb, "embed_documents", embed_documents)

    async def persist_fn(parsed, content_hash, dense_vectors):
        recorder.record("write", parsed.source)
        return {"source": parsed.source, "title": parsed.title, "chunk_count": 3, "parent_count": 1}

    items = _make_items(recorder, 3)
    results: dict[str, Any] = {}
    errors: dict[str, BaseException] = {}
    await IngestionPipeline().ingest(
        items,
        kb=fake_kb,
        parse_fn=parse_fn,
        persist_fn=persist_fn,
        finish_fn=_finish_recorder(results, errors),
    )

    assert not errors, errors
    assert len(results) == 3, (results, errors)
    # 首个 Embedding 早于最后一个解析结束 —— 这正是「不等整批解析完」的证据
    assert recorder.first("embed") < recorder.last("parse")
    # 且严格晚于首个解析结束（确实在切块之后）
    assert recorder.first("embed") > recorder.first("parse")


# ---------------------------------------------------------------------------
# 3. 流式提交与背压
# ---------------------------------------------------------------------------


def _bind_context(
    items: list[PipelineItem],
    *,
    kb: Any,
    parse_fn: Any,
    persist_fn: Any = None,
    prepare_fn: Any = None,
    commit_fn: Any = None,
    results: dict[str, Any],
    errors: dict[str, BaseException],
) -> None:
    """把随文档走的路由上下文挂到 item 上（流式 submit 的调用方职责）。

    写库可给两阶段（``prepare_fn`` + ``commit_fn``）或单回调（``persist_fn``）。
    """
    finish_fn = _finish_recorder(results, errors)
    for item in items:
        # 流式路径不等待结算，无需 Future（避免无人取回异常）
        item.result_future = None
        item.kb = kb
        item.parse_fn = parse_fn
        item.persist_fn = persist_fn
        item.prepare_fn = prepare_fn
        item.commit_fn = commit_fn
        item.finish_fn = finish_fn


@pytest.mark.asyncio
async def test_submit_enqueues_without_waiting_for_the_batch():
    """流式语义：submit 入队即返回，不等待本批其它文档处理完成。"""
    gate = threading.Event()

    def parse_fn(temp_path: Path, filename: str, ext: str) -> ParsedDocument:
        gate.wait(timeout=5)
        return _make_parsed(filename, _big_markdown(filename, sections=5))

    async def persist_fn(parsed, content_hash, dense_vectors):
        return {"source": parsed.source, "title": parsed.title, "chunk_count": 1, "parent_count": 1}

    results: dict[str, Any] = {}
    errors: dict[str, BaseException] = {}
    pipeline = IngestionPipeline()
    items = _make_items(StageRecorder(), 3, prefix="stream")
    _bind_context(
        items,
        kb=FakeKnowledgeBase(),
        parse_fn=parse_fn,
        persist_fn=persist_fn,
        results=results,
        errors=errors,
    )

    for item in items:
        pipeline.submit(item)

    # 三份文档均已入队且解析仍被 gate 挡住：submit 没有等待处理完成
    assert pipeline.pending_count == 3
    assert not results and not errors

    gate.set()
    assert await pipeline.drain(timeout=10)
    assert not errors and len(results) == 3
    assert pipeline.pending_count == 0


@pytest.mark.asyncio
async def test_queue_full_rejects_instead_of_blocking(monkeypatch):
    """背压：队列满时 submit 抛 IngestionQueueFullError，不阻塞调用方。"""
    monkeypatch.setattr(config, "INGEST_QUEUE_MAXSIZE", 1)
    monkeypatch.setattr(config, "resolve_parse_concurrency", lambda: 1)

    gate = threading.Event()

    def parse_fn(temp_path: Path, filename: str, ext: str) -> ParsedDocument:
        gate.wait(timeout=5)
        return _make_parsed(filename, _big_markdown(filename, sections=5))

    async def persist_fn(parsed, content_hash, dense_vectors):
        return {"source": parsed.source, "title": parsed.title, "chunk_count": 1, "parent_count": 1}

    results: dict[str, Any] = {}
    errors: dict[str, BaseException] = {}
    pipeline = IngestionPipeline()
    items = _make_items(StageRecorder(), 3, prefix="bp")
    _bind_context(
        items,
        kb=FakeKnowledgeBase(),
        parse_fn=parse_fn,
        persist_fn=persist_fn,
        results=results,
        errors=errors,
    )

    pipeline.submit(items[0])
    # 解析 worker 取走第一份文档并阻塞在解析中，提交队列被清空
    await asyncio.sleep(0.05)
    pipeline.submit(items[1])
    # 容量为 1 的队列已满：第三个文档被快速拒绝，而不是挂起等待
    with pytest.raises(IngestionQueueFullError):
        pipeline.submit(items[2])

    gate.set()
    assert await pipeline.drain(timeout=10)
    assert not errors and len(results) == 2


# ---------------------------------------------------------------------------
# 3-B. worker 自愈与关闭结算
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_worker_stage_body_exception_does_not_kill_worker():
    """阶段体异常不终止常驻 worker：包装层自动重启阶段体。"""
    pipeline = IngestionPipeline()
    calls = {"n": 0}

    async def body() -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("模拟阶段体异常")
        # 第二次正常返回（等价于阶段正常收敛退出）

    await asyncio.wait_for(pipeline._resilient("parse", body), timeout=5)
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_supervisor_restarts_dead_worker():
    """supervisor 兜底：已退出的阶段 worker 被重新拉起。"""
    pipeline = IngestionPipeline()
    pipeline.start()

    victim = pipeline._workers[0]
    assert victim is not None
    victim.cancel()
    await asyncio.sleep(0)

    assert victim.done()
    pipeline._restart_dead_workers()
    assert not pipeline._workers[0].done()


@pytest.mark.asyncio
async def test_shutdown_settles_pending_items():
    """关闭时结算全部在途文档，调用方不会永久等待。"""
    gate = threading.Event()

    def parse_fn(temp_path: Path, filename: str, ext: str) -> ParsedDocument:
        gate.wait(timeout=5)
        return _make_parsed(filename, _big_markdown(filename, sections=5))

    async def persist_fn(parsed, content_hash, dense_vectors):
        return {"source": parsed.source, "title": parsed.title, "chunk_count": 1, "parent_count": 1}

    results: dict[str, Any] = {}
    errors: dict[str, BaseException] = {}
    pipeline = IngestionPipeline()
    items = _make_items(StageRecorder(), 2, prefix="sd")
    _bind_context(
        items,
        kb=FakeKnowledgeBase(),
        parse_fn=parse_fn,
        persist_fn=persist_fn,
        results=results,
        errors=errors,
    )

    for item in items:
        pipeline.submit(item)
    await asyncio.sleep(0.05)
    assert pipeline.pending_count == 2

    pipeline.shutdown()
    gate.set()  # 释放被阻塞的解析线程，避免解释器退出时 join 子线程

    assert pipeline.pending_count == 0
    assert set(errors) == {"task-sd-0", "task-sd-1"}
    assert not results
    # 幂等：重复关闭无副作用
    pipeline.shutdown()


@pytest.mark.asyncio
async def test_write_stage_parallelizes_prepare_but_serializes_commit(monkeypatch):
    """两阶段写库：阶段 A（存储上传）可并行，阶段 B（向量库原子区）必须串行。

    这正是「拆写锁」的正确姿势——把网络 IO 移出锁，而不是取消锁：
    ``add_parsed_document`` 的文档间 SimHash 去重是"先查再插"，
    并发提交会让近似重复文档同时通过检查。
    """
    monkeypatch.setattr(config, "INGEST_WRITE_WORKERS", 2)
    state = {"prepare": 0, "max_prepare": 0, "commit": 0, "max_commit": 0}

    def parse_fn(temp_path: Path, filename: str, ext: str) -> ParsedDocument:
        return _make_parsed(filename, _big_markdown(filename, sections=5))

    async def prepare_fn(parsed, content_hash, dense_vectors):
        state["prepare"] += 1
        state["max_prepare"] = max(state["max_prepare"], state["prepare"])
        try:
            await asyncio.sleep(0.15)  # 模拟存储上传（网络 IO）
            return {"source": parsed.source}
        finally:
            state["prepare"] -= 1

    async def commit_fn(context):
        state["commit"] += 1
        state["max_commit"] = max(state["max_commit"], state["commit"])
        try:
            await asyncio.sleep(0.05)  # 模拟向量库「删旧行 + 插新行」
            return {
                "source": context["source"],
                "title": "",
                "chunk_count": 1,
                "parent_count": 1,
            }
        finally:
            state["commit"] -= 1

    results: dict[str, Any] = {}
    errors: dict[str, BaseException] = {}
    pipeline = IngestionPipeline()
    items = _make_items(StageRecorder(), 3, prefix="two")
    _bind_context(
        items,
        kb=FakeKnowledgeBase(),
        parse_fn=parse_fn,
        prepare_fn=prepare_fn,
        commit_fn=commit_fn,
        results=results,
        errors=errors,
    )
    for item in items:
        pipeline.submit(item)

    assert await pipeline.drain(timeout=10)
    assert not errors and len(results) == 3
    # 阶段 A：两个写消费者可并行上传
    assert state["max_prepare"] > 1, "存储上传应能并行（否则写锁未拆开）"
    # 阶段 B：向量库原子区任何时刻只有一个文档在提交
    assert state["max_commit"] == 1, "向量库原子区必须保持串行"


@pytest.mark.asyncio
async def test_write_stage_single_callback_runs_inside_lock():
    """只给 ``persist_fn`` 时退化为整段串行（兼容旧语义）。"""
    state = {"running": 0, "max_running": 0}

    def parse_fn(temp_path: Path, filename: str, ext: str) -> ParsedDocument:
        return _make_parsed(filename, _big_markdown(filename, sections=5))

    async def persist_fn(parsed, content_hash, dense_vectors):
        state["running"] += 1
        state["max_running"] = max(state["max_running"], state["running"])
        try:
            await asyncio.sleep(0.05)
            return {"source": parsed.source, "title": "", "chunk_count": 1, "parent_count": 1}
        finally:
            state["running"] -= 1

    results: dict[str, Any] = {}
    errors: dict[str, BaseException] = {}
    pipeline = IngestionPipeline()
    items = _make_items(StageRecorder(), 3, prefix="single")
    _bind_context(
        items,
        kb=FakeKnowledgeBase(),
        parse_fn=parse_fn,
        persist_fn=persist_fn,
        results=results,
        errors=errors,
    )
    for item in items:
        pipeline.submit(item)

    assert await pipeline.drain(timeout=10)
    assert not errors and len(results) == 3
    assert state["max_running"] == 1


@pytest.mark.asyncio
async def test_shared_pipeline_routes_embeddings_by_item_kb():
    """常驻共享队列按 item 的 kb 路由：不同 collection 的文档不会串。"""
    kb_a = FakeKnowledgeBase()
    kb_b = FakeKnowledgeBase()

    def parse_fn(temp_path: Path, filename: str, ext: str) -> ParsedDocument:
        return _make_parsed(filename, _big_markdown(filename, sections=5))

    async def persist_fn(parsed, content_hash, dense_vectors):
        return {"source": parsed.source, "title": parsed.title, "chunk_count": 1, "parent_count": 1}

    results: dict[str, Any] = {}
    errors: dict[str, BaseException] = {}
    pipeline = IngestionPipeline()
    items = _make_items(StageRecorder(), 2, prefix="kb")

    for item, kb in zip(items, (kb_a, kb_b), strict=True):
        _bind_context(
            [item],
            kb=kb,
            parse_fn=parse_fn,
            persist_fn=persist_fn,
            results=results,
            errors=errors,
        )
        pipeline.submit(item)

    assert await pipeline.drain(timeout=10)
    assert not errors and len(results) == 2
    # 每个知识库只收到属于自己那份文档的嵌入请求（共享队列未串集合）
    assert len(kb_a.embed_calls) == 1
    assert len(kb_b.embed_calls) == 1
    assert kb_a.embed_calls[0] != kb_b.embed_calls[0]


# ---------------------------------------------------------------------------
# 4. 失败隔离
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_single_document_failure_does_not_abort_batch(monkeypatch):
    """中间文档切块失败时，该任务失败、其余任务照常完成。"""
    failing = "doc-1.md"

    def parse_fn(temp_path: Path, filename: str, ext: str) -> ParsedDocument:
        return _make_parsed(filename, _big_markdown(filename, sections=5))

    def chunk_fn(markdown, *, source, title, category, is_pdf, blocks=()):
        if source == failing:
            raise ValueError(f"模拟切块失败：{source}")
        return chunk_worker.chunk_markdown_inline(
            markdown, source=source, title=title, category=category, is_pdf=is_pdf,
            blocks=blocks,
        )

    monkeypatch.setattr(pipeline_module, "chunk_markdown_inline", chunk_fn)

    async def persist_fn(parsed, content_hash, dense_vectors):
        return {"source": parsed.source, "title": parsed.title, "chunk_count": 1, "parent_count": 1}

    items = _make_items(StageRecorder(), 3)
    results: dict[str, Any] = {}
    errors: dict[str, BaseException] = {}
    await IngestionPipeline().ingest(
        items,
        kb=FakeKnowledgeBase(),
        parse_fn=parse_fn,
        persist_fn=persist_fn,
        finish_fn=_finish_recorder(results, errors),
    )

    assert set(results) == {"task-doc-0", "task-doc-2"}
    assert set(errors) == {"task-doc-1"}
    assert "模拟切块失败" in str(errors["task-doc-1"])
    # 所有 Future 均已结算，不会让调用方永久等待
    assert all(item.result_future.done() for item in items)


@pytest.mark.asyncio
async def test_parse_failure_isolated_and_temp_file_removed():
    """解析阶段失败：任务置错且临时文件被清理。"""
    recorder = StageRecorder()
    items = _make_items(recorder, 2)
    temp_files = []
    for item in items:
        path = _workdir() / f"parse-fail-{item.filename}"
        path.write_text("内容", encoding="utf-8")
        item.temp_path = path
        temp_files.append(path)

    def parse_fn(temp_path: Path, filename: str, ext: str) -> ParsedDocument:
        if filename == "doc-0.md":
            raise RuntimeError("模拟解析失败")
        return _make_parsed(filename, _big_markdown(filename, sections=5))

    async def persist_fn(parsed, content_hash, dense_vectors):
        return {"source": parsed.source, "title": parsed.title, "chunk_count": 1, "parent_count": 1}

    results: dict[str, Any] = {}
    errors: dict[str, BaseException] = {}
    await IngestionPipeline().ingest(
        items,
        kb=FakeKnowledgeBase(),
        parse_fn=parse_fn,
        persist_fn=persist_fn,
        finish_fn=_finish_recorder(results, errors),
    )

    assert set(errors) == {"task-doc-0"}
    assert set(results) == {"task-doc-1"}
    # 失败与成功路径都应清理临时文件
    assert all(not path.exists() for path in temp_files)


# ---------------------------------------------------------------------------
# 5. 进程池降级
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_broken_chunk_pool_falls_back_inline(monkeypatch):
    """进程池不可用时自动降级为内联切块，入库不失败。"""
    from finance_rag.src.core import config

    monkeypatch.setattr(
        pipeline_module,
        "get_chunk_pool",
        _raise_broken_process_pool,
    )
    marked = {"called": False}
    monkeypatch.setattr(
        pipeline_module,
        "mark_chunk_pool_unavailable",
        lambda: marked.__setitem__("called", True),
    )
    # 阈值降到 0：让示例文档走进程池分支，从而触发降级逻辑
    monkeypatch.setattr(config, "INGEST_CHUNK_MIN_CHARS", 0)

    recorder = StageRecorder()

    def parse_fn(temp_path: Path, filename: str, ext: str) -> ParsedDocument:
        return _make_parsed(filename, _big_markdown(filename, sections=5))

    async def persist_fn(parsed, content_hash, dense_vectors):
        return {"source": parsed.source, "title": parsed.title, "chunk_count": 1, "parent_count": 1}

    items = _make_items(recorder, 2)
    results: dict[str, Any] = {}
    errors: dict[str, BaseException] = {}
    await IngestionPipeline().ingest(
        items,
        kb=FakeKnowledgeBase(),
        parse_fn=parse_fn,
        persist_fn=persist_fn,
        finish_fn=_finish_recorder(results, errors),
    )

    assert not errors
    assert len(results) == 2
    assert marked["called"] is True


# ---------------------------------------------------------------------------
# 6. 回退路径（DocumentManager 串行入库）
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sequential_fallback_persists_every_document(monkeypatch):
    """``INGEST_PIPELINE_ENABLED=false`` 时走串行路径且结果一致。"""
    from finance_rag.src.services import document_service as ds
    from finance_rag.src.services.task_service import TaskStatus, get_task_manager

    persisted: list[str] = []

    class _FakeStorage:
        async def upload(self, key: str, data: bytes) -> None:
            persisted.append(key)

    class _FakeFingerprint:
        def is_unchanged(self, source: str, content_hash: str) -> bool:
            return False

    class _FakeKB:
        def get_fingerprint_store(self):
            return _FakeFingerprint()

    manager = ds.DocumentManager.__new__(ds.DocumentManager)
    manager._kb = _FakeKB()
    manager._collection_name = ""
    manager._storage = _FakeStorage()
    manager._parse_executor = None
    manager._bg_tasks = set()
    manager._pipeline_enabled = False

    monkeypatch.setattr(
        manager,
        "_parse_file",
        lambda temp_path, filename, ext, category="", metadata=None: _make_parsed(
            filename, _big_markdown(filename, sections=3)
        ),
    )

    async def fake_persist(parsed, content_hash, dense_vectors=None):
        return {
            "filename": parsed.source,
            "source": parsed.source,
            "title": parsed.title,
            "chunk_count": 2,
            "parent_count": 1,
        }

    monkeypatch.setattr(manager, "_persist_parsed_document", fake_persist)

    task = get_task_manager().create(filename="doc.md")
    tmp_file = Path("unused-doc.md")
    loop = asyncio.get_running_loop()
    future: asyncio.Future = loop.create_future()
    await manager._process_upload_background(
        task.id, tmp_file, "doc.md", ".md", _CONTENT_BYTES, "hash-x", "", future
    )

    result = await asyncio.wait_for(future, timeout=5)
    assert result["chunk_count"] == 2
    assert get_task_manager().get(task.id).status is TaskStatus.COMPLETED


@pytest.mark.asyncio
async def test_unchanged_document_is_skipped(monkeypatch):
    """内容哈希未变更时跳过入库，任务仍标记完成（skipped=True）。"""
    from finance_rag.src.services import document_service as ds

    class _FakeFingerprint:
        def is_unchanged(self, source: str, content_hash: str) -> bool:
            return True

    class _FakeKB:
        def get_fingerprint_store(self):
            return _FakeFingerprint()

    manager = ds.DocumentManager.__new__(ds.DocumentManager)
    manager._kb = _FakeKB()
    manager._pipeline_enabled = True
    manager._bg_tasks = set()

    tmp_file = Path("unused-skip.md")
    loop = asyncio.get_running_loop()
    future: asyncio.Future = loop.create_future()
    await manager._ingest_via_pipeline(
        "task-skip", tmp_file, "skip.md", ".md", _CONTENT_BYTES, "hash-same", "", future
    )

    result = await asyncio.wait_for(future, timeout=5)
    assert result["skipped"] is True
    assert result["chunk_count"] == 0


@pytest.mark.asyncio
async def test_submit_binds_category_into_parse_fn(monkeypatch):
    """分类必须随文档走：常驻流水线没有调用级参数，``category`` 经 partial 绑定。

    回归点：旧实现 ``_ingest_via_pipeline`` 收到 ``category`` 后既没写进
    ``PipelineItem`` 也没传给 ``parse_fn``，导致流水线（默认）路径下
    文档分类被静默丢弃。
    """
    from finance_rag.src.services import document_service as ds

    captured: dict[str, Any] = {}

    class _FakePipeline:
        def submit(self, item) -> None:
            captured["item"] = item

    monkeypatch.setattr(pipeline_module, "get_ingestion_pipeline", lambda: _FakePipeline())

    class _FakeFingerprint:
        def is_unchanged(self, source: str, content_hash: str) -> bool:
            return False

    class _FakeKB:
        def get_fingerprint_store(self):
            return _FakeFingerprint()

    manager = ds.DocumentManager.__new__(ds.DocumentManager)
    manager._kb = _FakeKB()

    seen_categories: list[str] = []

    def fake_parse(temp_path, filename, ext, category="", metadata=None):
        seen_categories.append(category)
        return _make_parsed(filename, _big_markdown(filename, sections=3))

    monkeypatch.setattr(manager, "_parse_file", fake_parse)
    monkeypatch.setattr(manager, "_prepare_parsed_document", lambda *a, **k: {})
    monkeypatch.setattr(manager, "_commit_parsed_document", lambda *a, **k: {})

    manager._submit_to_pipeline(
        "task-cat",
        Path("unused-cat.md"),
        "cat.md",
        ".md",
        _CONTENT_BYTES,
        "hash-cat",
        "compliance_risk",
    )

    item = captured["item"]
    # 路由上下文随文档走
    assert item.kb is manager.kb
    # 写库注入的是两阶段回调（锁外准备 + 锁内提交），而非单回调
    assert item.prepare_fn is not None and item.commit_fn is not None
    # 调用解析回调时分类确实传到了 _parse_file
    assert item.parse_fn(Path("p"), "cat.md", ".md") is not None
    assert seen_categories == ["compliance_risk"]


@pytest.mark.asyncio
async def test_upload_document_async_wires_into_resident_pipeline(monkeypatch):
    """生产装配链路：上传 → 流式入队 → 立即返回 task_id → 后台跑完并更新任务状态。"""
    from finance_rag.src.services import document_service as ds
    from finance_rag.src.services.task_service import TaskStatus, get_task_manager

    class _FakeFingerprint:
        def is_unchanged(self, source: str, content_hash: str) -> bool:
            return False

    class _WiredKB(FakeKnowledgeBase):
        """带指纹存储的替身：覆盖流水线用到的全部 kb 接口。"""

        def get_fingerprint_store(self):
            return _FakeFingerprint()

    manager = ds.DocumentManager.__new__(ds.DocumentManager)
    manager._kb = _WiredKB()
    manager._collection_name = ""
    manager._storage = None
    manager._parse_executor = None
    manager._bg_tasks = set()
    manager._pipeline_enabled = True

    # 临时目录重定向到工作区：沙箱禁止写系统临时目录
    monkeypatch.setattr(ds.tempfile, "gettempdir", lambda: str(_workdir()))

    def fake_parse(temp_path, filename, ext, category="", metadata=None):
        return _make_parsed(filename, _big_markdown(filename, sections=5))

    async def fake_prepare(parsed, content_hash, dense_vectors=None):
        return {"parsed": parsed}

    async def fake_commit(context):
        parsed = context["parsed"]
        return {
            "filename": parsed.source,
            "source": parsed.source,
            "title": parsed.title,
            "chunk_count": 1,
            "parent_count": 1,
        }

    monkeypatch.setattr(manager, "_parse_file", fake_parse)
    monkeypatch.setattr(manager, "_prepare_parsed_document", fake_prepare)
    monkeypatch.setattr(manager, "_commit_parsed_document", fake_commit)

    class _FakeUpload:
        filename = "wired.md"

        async def read(self) -> bytes:
            return _CONTENT_BYTES

        async def close(self) -> None:
            return None

    # 入队即返回：不创建后台任务、不等待入库结束
    submitted = await manager.upload_document_async(_FakeUpload(), category="management")
    task_id = submitted["task_id"]
    assert submitted["filename"] == "wired.md"
    assert not manager._bg_tasks, "流式路径不应创建后台任务"

    # 后台由常驻流水线推进，最终任务标记完成
    assert await pipeline_module.drain_ingestion_pipeline(timeout=10)
    task = get_task_manager().get(task_id)
    assert task is not None
    assert task.status is TaskStatus.COMPLETED, (task.status, task.error, task.result)
    assert task.result["chunk_count"] == 1


# ---------------------------------------------------------------------------
# 7. KnowledgeBase：预计算向量复用与校验
# ---------------------------------------------------------------------------


class _FakeMilvusClient:
    def __init__(self, existing_hashes: set[str] | None = None):
        self.inserted: list[dict[str, Any]] = []
        self.flushed = 0
        self._existing_hashes = existing_hashes or set()

    def query(self, **kwargs):
        return [
            {"content_hash": h, "dense_vector": [0.5, 0.5, 0.5]} for h in self._existing_hashes
        ]

    def insert(self, collection_name: str, data: list[dict[str, Any]]) -> None:
        self.inserted.extend(data)

    def flush(self, collection_name: str) -> None:
        self.flushed += 1

    def delete(self, **kwargs) -> None:
        return None


def _prepare_kb(monkeypatch, client: _FakeMilvusClient) -> KnowledgeBase:
    kb = KnowledgeBase(collection_name="test_kb")
    monkeypatch.setattr(kb, "_get_client", lambda: client)
    monkeypatch.setattr(kb, "ensure_collection", lambda: None)
    monkeypatch.setattr(kb, "_collection_has_version_fields", lambda c: True)
    monkeypatch.setattr(kb, "_collection_has_field", lambda c, f: True)
    monkeypatch.setattr(kb, "_mark_old_versions_inactive", lambda c, s: None)
    monkeypatch.setattr(kb, "_delete_by_source", lambda c, s, version=None: 0)
    monkeypatch.setattr(kb, "_check_duplicate_document", lambda s, t: None)
    monkeypatch.setattr(kb, "_get_parent_store", lambda: type("PS", (), {"store_batch": lambda self, x: None})())
    monkeypatch.setattr(kb, "_get_fingerprint_store", lambda: type("FS", (), {"update": lambda self, **kw: None})())
    return kb


def _chunked_document(source: str, texts: list[str]) -> ParsedDocument:
    import hashlib

    from langchain_core.documents import Document

    docs = []
    for index, text in enumerate(texts):
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "id": f"{source}-{index}",
                    "chunk_key": f"{source}-{index}",
                    "content_hash": content_hash,
                    "source": source,
                    "title": source,
                    "chunk": index,
                    "parent_id": f"p-{index}",
                    "heading_path": "",
                },
            )
        )
    return ParsedDocument(
        path=Path(source),
        source=source,
        title=source,
        chunks=DoclingChunks(chunks=docs, markdown=None, parents=[]),
        metadata={"category": ""},
    )


def test_precomputed_vectors_skip_inline_embedding(monkeypatch):
    """传入 dense_vectors 时不再触发内联嵌入，且写入行使用传入向量。"""
    client = _FakeMilvusClient()
    kb = _prepare_kb(monkeypatch, client)

    def _boom(texts):  # pragma: no cover - 不应被调用
        raise AssertionError("传入 dense_vectors 时不应再触发内联嵌入")

    monkeypatch.setattr(kb, "_embed_documents", _boom)

    parsed = _chunked_document("a.md", ["第一段", "第二段"])
    result = kb.add_parsed_document(
        parsed,
        content_hash="hash-a",
        dense_vectors=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    )

    assert result["chunk_count"] == 2
    assert [row["dense_vector"] for row in client.inserted] == [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
    assert client.flushed == 1


def test_dense_vector_count_mismatch_raises(monkeypatch):
    """向量数量与切块数量不一致时必须快速失败，避免错位写入。"""
    client = _FakeMilvusClient()
    kb = _prepare_kb(monkeypatch, client)
    parsed = _chunked_document("b.md", ["第一段", "第二段"])

    with pytest.raises(ValueError, match="稠密向量数量与切块数量不一致"):
        kb.add_parsed_document(parsed, content_hash="hash-b", dense_vectors=[[1.0, 0.0, 0.0]])
    assert client.inserted == []


def test_existing_vectors_reused_for_unchanged_chunks(monkeypatch):
    """同 content_hash 的块复用集合内已有向量，只对缺失块做嵌入。"""
    import hashlib

    text_a, text_b = "已入库段落", "新增段落"
    hash_a = hashlib.sha256(text_a.encode("utf-8")).hexdigest()
    client = _FakeMilvusClient(existing_hashes={hash_a})
    kb = _prepare_kb(monkeypatch, client)

    embedded: list[list[str]] = []

    def _embed(texts):
        embedded.append(list(texts))
        return [[9.0, 9.0, 9.0] for _ in texts]

    monkeypatch.setattr(kb, "_embed_documents", _embed)
    parsed = _chunked_document("c.md", [text_a, text_b])

    kb.add_parsed_document(parsed, content_hash="hash-c")

    assert embedded == [[text_b]]
    rows = {row["content_hash"]: row["dense_vector"] for row in client.inserted}
    assert rows[hash_a] == [0.5, 0.5, 0.5]
    assert rows[hashlib.sha256(text_b.encode("utf-8")).hexdigest()] == [9.0, 9.0, 9.0]


# ---------------------------------------------------------------------------
# 8. 资源释放
# ---------------------------------------------------------------------------


def test_pipeline_shutdown_is_idempotent(monkeypatch):
    """shutdown 可重复调用且会释放线程池与进程池。"""
    closed: list[str] = []
    monkeypatch.setattr(pipeline_module, "shutdown_chunk_pool", lambda wait=False: closed.append("chunk"))
    pipeline = IngestionPipeline()
    pipeline.shutdown()
    pipeline.shutdown()
    assert closed == ["chunk"]
