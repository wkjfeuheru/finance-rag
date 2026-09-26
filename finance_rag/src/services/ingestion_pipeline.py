"""入库流水线：进程单例的常驻流式流水线（asyncio.Queue + 分阶段 Executor）。

设计
----
队列与阶段 worker **进程单例常驻**，文档上传完成即入队（:meth:`IngestionPipeline.submit`），
入队即返回，既不等待同批其它文档，也不为每个请求新建任务与队列。

::

    提交 ──► in_queue ──► [解析 worker ×P] ──► parse_queue ──► [切块 worker ×C]
                                                                      │
       [写入 worker ×1] ◄── write_queue ◄── [Embedding worker ×E] ◄── chunk_queue

阶段重叠覆盖**整个服务会话**而非单一批次：A 文档在 Embedding 的同时 B 文档已在解析，
与它们来自哪个 HTTP 请求无关。

各阶段执行体与理由
------------------
=========== ========================================================== ============================================
阶段        执行体                                                     理由
=========== ========================================================== ============================================
解析        ``ThreadPoolExecutor``，worker 数 == 解析并发上限           MinerU 加载 GB 级 ONNX/GPU 权重，进程池化会成倍放大内存
切块        ``ProcessPoolExecutor``（spawn），消费者数 == 池 worker 数   清洗/SimHash/切分是纯 Python，受 GIL 限制，进程池才是正解
Embedding   ``ThreadPoolExecutor``（专用）                              ONNX 为 C 层；注意 ``OnnxEmbedder`` 内置类级推理锁，
                                                                       实际 ``session.run`` 仍串行，此处仅重叠分批提交
写入        ``await``（两阶段：锁外准备 + 锁内提交）                    存储上传可并行；向量库「删旧行 + 插新行」必须串行
=========== ========================================================== ============================================

关键不变量
----------
* **上传即入队**：``submit()`` 非阻塞（``put_nowait``），队列满时快速失败抛
  :class:`IngestionQueueFullError`，由接口转换为 503，**不阻塞调用方**。
* **背压**：四个队列均有限容量，压力通过「入队失败」回传上游，内存上界约为
  ``INGEST_QUEUE_MAXSIZE × MAX_UPLOAD_SIZE_MB``。
* **解析并发上限 = worker 数量**：解析 worker 持有文档直到成功入队，
  因此下游背压会自然传导到解析阶段，无需额外的信号量机制。
* **失败隔离**：任一阶段异常只让该文档失败，不拖垮整批。
* **worker 自愈**：常驻 worker 一旦死亡，其上游队列会填满并导致**全量上传阻塞**，
  因此阶段体异常一律在 worker 内兜住重试，并由 supervisor 兜底重启已退出的 worker。
* **随文档走的路由上下文**：共享队列里混有不同 collection 的文档，``kb`` 与
  ``parse_fn`` / ``persist_fn`` / ``finish_fn`` / ``on_progress`` 一律挂在
  :class:`PipelineItem` 上，不再作为调用级参数。
* **写序化（两阶段）**：写库拆成「锁外准备」与「锁内提交」两段。
  阶段 A（原文件/图片/解析 md 落存储）是网络 IO，与其他文档并行；
  阶段 B（向量库「删旧行 + 插新行」+ 文档间 SimHash 去重 + 指纹）必须同
  collection 串行——``add_parsed_document`` 的去重是"先查再插"，并发会让
  近似重复文档同时通过检查，且同 source 并发覆盖会互相删除。
  两阶段由**流水线强制**（而非依赖回调自觉），因此写入阶段可以有多个消费者。
  此外 ``KnowledgeBase`` 自身按 collection 持有写锁，因此**流水线之外的调用方**
  （对象存储事件回调、管理脚本）与流水线写入也共享同一串行点；
  本流水线的写锁则额外保证"任一时刻只有一个 commit 在跑"这一回调契约。
* **完成语义**：每个文档用独立的 ``asyncio.Future``（可选）传递结果；
  关闭时统一结算全部未完成文档，调用方不会永久等待。
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import logging
import threading
import time
import weakref
from collections.abc import Awaitable, Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from finance_rag.src.core import config as _config
from finance_rag.src.core.config import (
    DOCUMENT_PARSE_WORKERS,
    INGEST_EMBED_WORKERS,
)
from finance_rag.src.core.exceptions import (
    IngestionPipelineClosedError,
    IngestionQueueFullError,
)
from finance_rag.src.infrastructure.vector_store.milvus_kb import (
    KnowledgeBase,
    ParsedDocument,
)
from finance_rag.src.rag.ingestion.chunk_worker import (
    chunk_document,
    chunk_markdown_inline,
    get_chunk_pool,
    mark_chunk_pool_unavailable,
    shutdown_chunk_pool,
    use_process_pool,
)
from finance_rag.src.utils.metrics import (
    ingest_queue_depth,
    ingest_queue_rejected,
    ingest_stage_seconds,
    ingest_worker_restarts,
)

logger = logging.getLogger(__name__)


async def enrich_chunks(
    *,
    source: str,
    title: str,
    markdown: str,
    chunks: Any,
    metadata: dict[str, Any] | None,
    enable_llm: bool | None = None,
) -> tuple[Any, dict[str, Any]]:
    """切块之后、嵌入之前的增强：研报元数据抽取 + 图片描述成块。

    为什么必须卡在这个位置（而不是推迟到写阶段）：

    * 图片描述会**新增子块**，而稠密向量是按子块顺序严格一一对应的
      （写阶段 ``zip(chunks.chunks, dense_vectors, strict=True)``），
      嵌入之后再追加块会让向量与块错位；
    * 元数据要写进 Milvus 标量，越早合并越不容易漏掉某条写入路径。

    外部调用（LLM / 视觉模型）都放到线程执行器里，不阻塞事件循环。
    「抽取失败不阻断入库」由 :func:`extract_metadata` 内部保证（模型异常一律
    降级为正则结果）；因此这里允许异常向上冒泡——那说明是代码缺陷，
    应当由流水线的单文件失败隔离暴露出来，而不是静默入库一篇空元数据文档。
    """
    from finance_rag.src.rag.ingestion.image_captioner import build_image_chunks
    from finance_rag.src.rag.ingestion.metadata_extractor import (
        apply_metadata,
        extract_metadata,
    )

    loop = asyncio.get_running_loop()
    if enable_llm is None:
        enable_llm = bool(_config.ENABLE_METADATA_LLM)
    extracted = await loop.run_in_executor(
        None,
        functools.partial(
            extract_metadata,
            source,
            title,
            markdown or "",
            enable_llm=enable_llm,
        ),
    )
    merged = apply_metadata(metadata, extracted)

    images = list(getattr(chunks, "images", None) or ())
    if images:
        image_chunks = await loop.run_in_executor(
            None,
            functools.partial(
                build_image_chunks,
                images,
                source=source,
                title=title,
                start_index=len(chunks.chunks),
            ),
        )
        chunks.chunks.extend(image_chunks)
    return chunks, merged

# 阶段体重启前的退避（避免异常风暴时忙循环）
_RESTART_BACKOFF_SECONDS = 0.05

# 队列深度指标：队列名 -> (Prometheus label, 消费该队列的阶段名)
_QUEUE_METRIC_SPECS: tuple[tuple[str, str, str], ...] = (
    ("_in_queue", "pending", "解析"),
    ("_parse_queue", "chunk", "切块"),
    ("_chunk_queue", "embed", "嵌入"),
    ("_write_queue", "write", "写入"),
)

# 写库阶段 A 回调：在**写锁外**执行，可与其他文档并行（存储上传等网络 IO），
# 返回交给提交阶段的上下文对象。
PrepareFn = Callable[[ParsedDocument, str, list[list[float]] | None], Awaitable[Any]]
# 写库阶段 B 回调：在**写锁内**执行（向量库「删旧行 + 插新行」等必须串行的部分），
# 入参为阶段 A 返回的上下文，返回入库统计。
CommitFn = Callable[[Any], Awaitable[dict[str, Any]]]
# 单回调写法（向后兼容）：整段都在写锁内执行。
# 由 DocumentManager 注入，避免流水线反向依赖服务层。
PersistFn = Callable[[ParsedDocument, str, list[list[float]] | None], Awaitable[dict[str, Any]]]
# 解析阶段回调：MinerU 解析 + 日期补全，返回 ParsedDocument。
ParseFn = Callable[[Path, str, str], ParsedDocument]
# 任务终态回调：成功统计 / 失败异常
FinishFn = Callable[[str, dict[str, Any] | None, BaseException | None], None]
# 阶段进度回调
ProgressFn = Callable[[str, dict[str, Any]], None]

# 所有存活实例（含测试直接构造的实例），供 shutdown_all_pipelines 统一收尾
_live_pipelines: weakref.WeakSet[IngestionPipeline] = weakref.WeakSet()


@dataclass
class PipelineItem:
    """流水线中流转的单个文档。

    ``kb`` 与三个回调属于**随文档走的路由上下文**：常驻队列里可能同时混有
    不同 collection 的文档，因此这些参数必须挂在文档上，而不是作为
    ``ingest()`` 的调用级参数（否则会串集合）。
    """

    task_id: str
    temp_path: Path
    filename: str
    ext: str
    content: bytes
    content_hash: str
    parsed: ParsedDocument | None = None
    result_future: asyncio.Future | None = None
    # 路由上下文（submit 前必须填好）
    kb: KnowledgeBase | None = None
    parse_fn: ParseFn | None = None
    # 写库两阶段：``prepare_fn``（锁外、可并行）+ ``commit_fn``（锁内、串行）。
    # 二者齐备时走两阶段；只给 ``persist_fn`` 时整段在写锁内串行（兼容旧语义）。
    prepare_fn: PrepareFn | None = None
    commit_fn: CommitFn | None = None
    persist_fn: PersistFn | None = None
    finish_fn: FinishFn | None = None
    on_progress: ProgressFn | None = None


def _fail(item: PipelineItem, exc: BaseException) -> None:
    """把失败结果交付给等待方；已完成则忽略（每个 Future 只结算一次）。"""
    future = item.result_future
    if future is not None and not future.done():
        future.set_exception(exc)


def _complete(item: PipelineItem, result: dict[str, Any]) -> None:
    """把成功结果交付给等待方；已完成则忽略。"""
    future = item.result_future
    if future is not None and not future.done():
        future.set_result(result)


def _unlink_temp(path: Path) -> None:
    """清理临时文件（同步、幂等；失败仅记录，不阻断入库）。"""
    try:
        path.unlink(missing_ok=True)
    except Exception as exc:  # pragma: no cover - 防御
        logger.debug("临时文件清理失败 %s：%s", path, exc)


@contextlib.contextmanager
def _stage_timer(stage: str):
    """记录单个文档在某一阶段的耗时（定位入库链路瓶颈）。"""
    started = time.perf_counter()
    try:
        yield
    finally:
        ingest_stage_seconds.labels(stage=stage).observe(time.perf_counter() - started)


class IngestionPipeline:
    """入库流水线运行时（**进程单例**，可服务多个 collection）。

    线程池、进程池、队列与阶段 worker 全部跨批次复用；``start()`` 在首次
    ``submit()`` 时惰性启动 worker（``asyncio.create_task`` 需要运行中的事件循环）。
    """

    def __init__(self) -> None:
        self._parse_executor = ThreadPoolExecutor(
            max_workers=max(1, DOCUMENT_PARSE_WORKERS),
            thread_name_prefix="document-parser",
        )
        self._embed_executor = ThreadPoolExecutor(
            max_workers=max(1, INGEST_EMBED_WORKERS),
            thread_name_prefix="ingest-embed",
        )

        # 常驻状态：队列/worker/写锁均在 start() 中创建（需要运行中的事件循环）
        self._started = False
        self._closed = False
        self._in_queue: asyncio.Queue | None = None
        self._parse_queue: asyncio.Queue | None = None
        self._chunk_queue: asyncio.Queue | None = None
        self._write_queue: asyncio.Queue | None = None
        self._write_lock: asyncio.Lock | None = None
        self._specs: list[tuple[str, Callable[[], Awaitable[None]]]] = []
        self._workers: list[asyncio.Task | None] = []
        self._supervisor: asyncio.Task | None = None
        self._sampler: asyncio.Task | None = None
        # 已提交但尚未结算的文档（关闭时统一结算，避免调用方永久等待）
        self._pending: dict[str, PipelineItem] = {}

        _live_pipelines.add(self)

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    @property
    def started(self) -> bool:
        return self._started

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def pending_count(self) -> int:
        """在途（已提交未结算）文档数。"""
        return len(self._pending)

    def start(self) -> None:
        """启动常驻队列与阶段 worker（幂等；需在事件循环内调用）。"""
        if self._started:
            return
        if self._closed:
            raise IngestionPipelineClosedError("入库流水线已关闭，无法重新启动")

        maxsize = max(1, _config.INGEST_QUEUE_MAXSIZE)
        self._in_queue = asyncio.Queue(maxsize=maxsize)
        self._parse_queue = asyncio.Queue(maxsize=maxsize)
        self._chunk_queue = asyncio.Queue(maxsize=maxsize)
        self._write_queue = asyncio.Queue(maxsize=maxsize)
        self._write_lock = asyncio.Lock()

        self._specs = self._build_specs()
        self._workers = [None] * len(self._specs)
        for index in range(len(self._specs)):
            self._spawn(index)

        self._supervisor = asyncio.create_task(self._supervise(), name="ingest-supervisor")
        self._sampler = asyncio.create_task(self._sample_metrics(), name="ingest-metrics")
        self._started = True
        logger.info(
            "入库流水线已启动：解析×%d 切块×%d 嵌入×%d 写入×%d，队列上限 %d",
            _config.resolve_parse_concurrency(),
            _config.resolve_chunk_concurrency(),
            max(1, _config.INGEST_EMBED_WORKERS),
            max(1, _config.resolve_write_concurrency()),
            maxsize,
        )

    def _build_specs(self) -> list[tuple[str, Callable[[], Awaitable[None]]]]:
        """构造阶段 worker 规格：``(阶段名, 阶段体工厂)``。"""
        specs: list[tuple[str, Callable[[], Awaitable[None]]]] = []
        for _ in range(max(1, _config.resolve_parse_concurrency())):
            specs.append(("parse", self._parse_worker))
        for _ in range(max(1, _config.resolve_chunk_concurrency())):
            specs.append(("chunk", self._chunk_worker))
        for _ in range(max(1, _config.INGEST_EMBED_WORKERS)):
            specs.append(("embed", self._embed_worker))
        for _ in range(max(1, _config.resolve_write_concurrency())):
            specs.append(("write", self._write_worker))
        return specs

    def _spawn(self, index: int) -> None:
        """启动（或重启）第 ``index`` 个阶段 worker。"""
        stage, factory = self._specs[index]
        self._workers[index] = asyncio.create_task(
            self._resilient(stage, factory), name=f"ingest-{stage}-{index}"
        )

    async def _resilient(
        self, stage: str, factory: Callable[[], Awaitable[None]]
    ) -> None:
        """阶段体包装：**任何异常都不得终止常驻 worker**。

        常驻 worker 一旦死亡，其上游队列会填满并导致全量上传阻塞，
        因此这里在阶段体之外再兜一层异常并自动重启阶段体。
        """
        while True:
            try:
                await factory()
                return
            except asyncio.CancelledError:
                raise
            except BaseException:  # noqa: BLE001 - worker 绝不允许因单次异常死亡
                ingest_worker_restarts.labels(stage=stage).inc()
                if self._closed:
                    logger.warning("入库流水线 %s 阶段 worker 在关闭过程中退出", stage)
                    return
                logger.exception(
                    "入库流水线 %s 阶段 worker 异常退出，%.2fs 后重启阶段体",
                    stage,
                    _RESTART_BACKOFF_SECONDS,
                )
                await asyncio.sleep(_RESTART_BACKOFF_SECONDS)

    async def _supervise(self) -> None:
        """兜底监督：阶段 worker 任务整体意外结束（自愈层未能覆盖的路径）时重启。"""
        while not self._closed:
            await asyncio.sleep(max(0.1, _config.INGEST_SUPERVISOR_INTERVAL_SECONDS))
            if self._closed:
                return
            self._restart_dead_workers()

    def _restart_dead_workers(self) -> None:
        """重启所有已退出的阶段 worker（supervisor 兜底路径）。"""
        for index, task in enumerate(list(self._workers)):
            if task is not None and not task.done():
                continue
            stage = self._specs[index][0]
            logger.warning("入库流水线 %s 阶段 worker 已退出，由 supervisor 重启", stage)
            ingest_worker_restarts.labels(stage=stage).inc()
            self._spawn(index)

    async def _sample_metrics(self) -> None:
        """周期性上报各阶段队列深度（定位瓶颈段）。"""
        while not self._closed:
            for attr, label, _ in _QUEUE_METRIC_SPECS:
                queue = getattr(self, attr, None)
                if queue is not None:
                    ingest_queue_depth.labels(stage=label).set(queue.qsize())
            await asyncio.sleep(max(0.5, _config.INGEST_METRICS_INTERVAL_SECONDS))

    async def drain(self, timeout: float | None = None) -> bool:
        """等待在途文档全部结算（不接收新文档的语义由调用方保证）。

        Returns:
            ``True`` 表示在超时前全部结算；``False`` 表示仍有在途文档。
        """
        limit = _config.INGEST_DRAIN_TIMEOUT_SECONDS if timeout is None else timeout
        deadline = time.monotonic() + max(0.0, limit)
        while self._pending and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        return not self._pending

    def shutdown(self, wait: bool = False) -> None:
        """关闭流水线：停止调度 → 结算全部在途文档 → 释放执行器。

        必须在事件循环线程内调用（需要 cancel 阶段 worker 任务）；
        幂等，重复调用无副作用。

        需要在途文档跑完时应先 ``await drain()`` 再调用本方法。
        """
        if self._closed:
            return
        self._closed = True

        # 1) 停止调度与监控（取消即收敛：常驻队列不需要哨兵机制）
        for task in [*self._workers, self._supervisor, self._sampler]:
            if task is None or task.done():
                continue
            try:
                task.cancel()
            except RuntimeError:  # pragma: no cover - 事件循环已关闭：任务随循环丢弃
                pass
        self._workers = []
        self._supervisor = None
        self._sampler = None

        # 2) 结算全部未完成文档，避免调用方永久等待
        for item in list(self._pending.values()):
            self._settle_failure(item, RuntimeError("入库流水线已关闭，该文档未完成入库"))
        self._pending.clear()

        # 3) 释放线程池与切块进程池
        for executor, name in (
            (self._embed_executor, "Embedding 线程池"),
            (self._parse_executor, "解析线程池"),
        ):
            try:
                executor.shutdown(wait=wait, cancel_futures=True)
            except Exception as exc:  # pragma: no cover - 防御
                logger.warning("%s 关闭失败：%s", name, exc)
        shutdown_chunk_pool(wait=wait)

        _live_pipelines.discard(self)
        self._started = False
        logger.info("入库流水线已关闭")

    # ------------------------------------------------------------------
    # 提交入口
    # ------------------------------------------------------------------

    def submit(self, item: PipelineItem) -> None:
        """把一份文档投递进流水线并立即返回（非阻塞）。

        Raises:
            IngestionPipelineClosedError: 流水线已关闭。
            IngestionQueueFullError: 提交队列已满（背压，调用方应返回 503）。
            ValueError: 路由上下文缺失（``kb`` / ``parse_fn`` / ``persist_fn``）。
        """
        if self._closed:
            raise IngestionPipelineClosedError("入库流水线已关闭，暂时无法接收新的文档")
        self.start()
        self._validate(item)

        self._pending[item.task_id] = item
        try:
            assert self._in_queue is not None  # start() 已保证
            self._in_queue.put_nowait(item)
        except asyncio.QueueFull as exc:
            self._pending.pop(item.task_id, None)
            ingest_queue_rejected.inc()
            raise IngestionQueueFullError(
                f"入库队列已满（上限 {max(1, _config.INGEST_QUEUE_MAXSIZE)}），请稍后重试"
            ) from exc

    @staticmethod
    def _validate(item: PipelineItem) -> None:
        """校验随文档走的路由上下文是否齐备（共享队列无法回退到调用级参数）。"""
        if item.kb is None:
            raise ValueError(f"PipelineItem 缺少 kb（task_id={item.task_id}）")
        if item.parse_fn is None:
            raise ValueError(f"PipelineItem 缺少 parse_fn（task_id={item.task_id}）")
        two_phase = item.prepare_fn is not None and item.commit_fn is not None
        if not two_phase and item.persist_fn is None:
            raise ValueError(
                "PipelineItem 需要 prepare_fn + commit_fn（两阶段写库），"
                f"或退化为 persist_fn 单回调（task_id={item.task_id}）"
            )

    async def ingest(
        self,
        items: list[PipelineItem],
        *,
        kb: KnowledgeBase,
        parse_fn: ParseFn,
        persist_fn: PersistFn | None = None,
        finish_fn: FinishFn,
        on_progress: ProgressFn | None = None,
        prepare_fn: PrepareFn | None = None,
        commit_fn: CommitFn | None = None,
    ) -> None:
        """批量投递并等待全部文档结算（保留旧语义：单个文档失败不抛出）。

        面向「调用方希望整批跑完」的场景；上传接口走的是
        :meth:`submit` 的流式路径（入队即返回，不等整批）。
        已自带路由上下文的 item 优先使用自身字段，否则回填调用级参数。

        写库可选两阶段（``prepare_fn`` + ``commit_fn``）或单回调（``persist_fn``）；
        都不提供时 item 自身必须已带其一。
        """
        if not items:
            return

        loop = asyncio.get_running_loop()
        futures: list[asyncio.Future] = []
        for item in items:
            if item.kb is None:
                item.kb = kb
            if item.parse_fn is None:
                item.parse_fn = parse_fn
            if item.persist_fn is None and item.commit_fn is None:
                item.persist_fn = persist_fn
            if item.prepare_fn is None:
                item.prepare_fn = prepare_fn
            if item.commit_fn is None:
                item.commit_fn = commit_fn
            if item.finish_fn is None:
                item.finish_fn = finish_fn
            if item.on_progress is None:
                item.on_progress = on_progress
            if item.result_future is None:
                item.result_future = loop.create_future()
            futures.append(item.result_future)
            self.submit(item)

        await asyncio.gather(*futures, return_exceptions=True)

    # ------------------------------------------------------------------
    # 结算
    # ------------------------------------------------------------------

    def _progress(self, item: PipelineItem, payload: dict[str, Any]) -> None:
        """上报阶段进度；回调异常不影响入库。"""
        if item.on_progress is None:
            return
        try:
            item.on_progress(item.task_id, payload)
        except Exception as exc:  # pragma: no cover - 防御
            logger.warning("进度回调失败（task_id=%s）：%s", item.task_id, exc)

    def _settle_success(self, item: PipelineItem, result: dict[str, Any]) -> None:
        """结算成功：交付 Future、上报终态、清理临时文件。"""
        self._pending.pop(item.task_id, None)
        _complete(item, result)
        if item.finish_fn is not None:
            try:
                item.finish_fn(item.task_id, result, None)
            except Exception as exc:  # pragma: no cover - 防御
                logger.warning("终态回调失败（task_id=%s）：%s", item.task_id, exc)
        _unlink_temp(item.temp_path)

    def _settle_failure(self, item: PipelineItem, exc: BaseException) -> None:
        """结算失败：交付异常、上报终态、清理临时文件（幂等）。"""
        self._pending.pop(item.task_id, None)
        _fail(item, exc)
        if item.finish_fn is not None:
            try:
                item.finish_fn(item.task_id, None, exc)
            except Exception as callback_exc:  # pragma: no cover - 防御
                logger.warning("终态回调失败（task_id=%s）：%s", item.task_id, callback_exc)
        _unlink_temp(item.temp_path)

    # ------------------------------------------------------------------
    # 各阶段 worker
    # ------------------------------------------------------------------

    async def _parse_worker(self) -> None:
        """阶段 1：MinerU 解析（解析线程池）→ 入队切块。

        并发上限由 worker 数量（``resolve_parse_concurrency``）结构性保证；
        worker 持有文档直到成功入队，因此 ``parse_queue`` 满时不会再启动新解析。
        """
        assert self._in_queue is not None
        while True:
            item = await self._in_queue.get()
            await self._run_parse(item)

    async def _run_parse(self, item: PipelineItem) -> None:
        """解析单个文档（单文件失败隔离）。"""
        parse_fn = item.parse_fn
        if parse_fn is None:  # submit() 已校验，此处仅防御
            self._settle_failure(item, ValueError(f"解析回调缺失（task_id={item.task_id}）"))
            return
        try:
            loop = asyncio.get_running_loop()
            with _stage_timer("parse"):
                parsed = await loop.run_in_executor(
                    self._parse_executor, parse_fn, item.temp_path, item.filename, item.ext
                )
            item.parsed = parsed
            self._progress(item, {"stage": "parsed"})
            assert self._parse_queue is not None
            await self._parse_queue.put(item)
        except asyncio.CancelledError:
            self._settle_failure(item, RuntimeError("入库任务已取消"))
            raise
        except BaseException as exc:  # noqa: BLE001 - 单文件失败隔离
            logger.warning("解析失败（task_id=%s）：%s", item.task_id, exc)
            self._settle_failure(item, exc)

    async def _chunk_worker(self) -> None:
        """阶段 2：清洗 + 层级切块（进程池）→ 入队 Embedding。

        必须与切块进程池 worker 数等量，否则单消费者会 ``await`` 逐个等待，
        让进程池退化为串行执行。
        """
        assert self._parse_queue is not None
        while True:
            item = await self._parse_queue.get()
            await self._run_chunk(item)

    async def _run_chunk(self, item: PipelineItem) -> None:
        """切块单个文档（单文件失败隔离）。"""
        try:
            with _stage_timer("chunk"):
                await self._chunk_stage(item)
            self._progress(item, {"stage": "chunked"})
            assert self._chunk_queue is not None
            await self._chunk_queue.put(item)
        except asyncio.CancelledError:
            self._settle_failure(item, RuntimeError("入库任务已取消"))
            raise
        except BaseException as exc:  # noqa: BLE001 - 单文件失败隔离
            logger.warning("切块失败（task_id=%s）：%s", item.task_id, exc)
            self._settle_failure(item, exc)

    async def _embed_worker(self) -> None:
        """阶段 3-a：计算缺失 chunk 的稠密向量 → 入队写入。"""
        assert self._chunk_queue is not None
        while True:
            item = await self._chunk_queue.get()
            await self._run_embed(item)

    async def _run_embed(self, item: PipelineItem) -> None:
        """计算单个文档的稠密向量（单文件失败隔离）。"""
        try:
            with _stage_timer("embed"):
                dense_vectors = await self._embed_stage(item)
            assert self._write_queue is not None
            await self._write_queue.put((item, dense_vectors))
        except asyncio.CancelledError:
            self._settle_failure(item, RuntimeError("入库任务已取消"))
            raise
        except BaseException as exc:  # noqa: BLE001 - 单文件失败隔离
            logger.warning("Embedding 失败（task_id=%s）：%s", item.task_id, exc)
            self._settle_failure(item, exc)

    async def _write_worker(self) -> None:
        """阶段 4：两阶段写库（锁外存储上传 + 锁内向量库提交）。"""
        assert self._write_queue is not None
        while True:
            item, dense_vectors = await self._write_queue.get()
            await self._run_write(item, dense_vectors)

    async def _run_write(self, item: PipelineItem, dense_vectors: list[list[float]] | None) -> None:
        """写入单个文档（单文件失败隔离）。"""
        try:
            with _stage_timer("write"):
                result = await self._write_stage(item, dense_vectors)
            self._settle_success(item, result)
            self._progress(item, {"stage": "written"})
        except asyncio.CancelledError:
            self._settle_failure(item, RuntimeError("入库任务已取消"))
            raise
        except BaseException as exc:  # noqa: BLE001 - 单文件失败隔离
            logger.warning("写入失败（task_id=%s）：%s", item.task_id, exc)
            self._settle_failure(item, exc)

    # ------------------------------------------------------------------
    # 阶段实现
    # ------------------------------------------------------------------

    async def _chunk_stage(self, item: PipelineItem) -> None:
        """阶段 2 实现：对解析出的 markdown 做清洗 + 层级切块。"""
        parsed = item.parsed
        if parsed is None:
            raise RuntimeError(f"解析结果缺失（task_id={item.task_id}）")

        chunks = parsed.chunks
        markdown = chunks.markdown if chunks is not None else None
        if markdown is None:
            raise RuntimeError(f"解析未产出 markdown（task_id={item.task_id}）")

        new_chunks = await self._chunk_markdown(
            markdown,
            source=parsed.source,
            title=parsed.title,
            category=str((parsed.metadata or {}).get("category", "") or ""),
            is_pdf=Path(parsed.source).suffix.lower() == ".pdf",
            blocks=tuple(getattr(chunks, "blocks", None) or ()),
        )
        # 图片字节不进程池化：切块进程只做纯文本工作，图片在这里原样接回来
        new_chunks.images = list(getattr(chunks, "images", None) or ())
        new_chunks, metadata = await enrich_chunks(
            source=parsed.source,
            title=parsed.title,
            markdown=markdown,
            chunks=new_chunks,
            metadata=parsed.metadata,
        )
        item.parsed = ParsedDocument(
            path=parsed.path,
            source=parsed.source,
            title=parsed.title,
            chunks=new_chunks,
            metadata=metadata,
        )

    async def _embed_stage(self, item: PipelineItem) -> list[list[float]] | None:
        """阶段 3-a 实现：计算该文档缺失 chunk 的稠密向量。

        使用 ``item.kb`` 而非调用级参数：共享队列里可能混有不同 collection 的文档。

        Returns:
            与全部子块一一对应的向量列表；全部命中旧向量时返回 None
            （写阶段会基于旧的 ``content_hash`` 复用集合内已有向量）。
        """
        parsed = item.parsed
        if parsed is None:
            raise RuntimeError(f"切块结果缺失（task_id={item.task_id}）")
        kb = item.kb
        if kb is None:
            raise RuntimeError(f"知识库上下文缺失（task_id={item.task_id}）")

        chunks = parsed.chunks
        texts = [chunk.page_content for chunk in chunks.chunks]
        loop = asyncio.get_running_loop()
        existing_vectors = await loop.run_in_executor(
            self._embed_executor, kb.fetch_existing_vectors, parsed.source
        )
        missing_indexes = [
            index
            for index, chunk in enumerate(chunks.chunks)
            if str(chunk.metadata.get("content_hash", "")) not in existing_vectors
        ]
        if not missing_indexes:
            return None

        new_vectors = await loop.run_in_executor(
            self._embed_executor,
            kb.embed_documents,
            [texts[index] for index in missing_indexes],
        )
        dense_vectors: list[list[float]] = [
            existing_vectors.get(str(chunk.metadata.get("content_hash", "")), [])
            for chunk in chunks.chunks
        ]
        for index, vector in zip(missing_indexes, new_vectors, strict=True):
            dense_vectors[index] = vector
        return dense_vectors

    async def _write_stage(
        self, item: PipelineItem, dense_vectors: list[list[float]] | None
    ) -> dict[str, Any]:
        """阶段 4 实现：两阶段写库。

        * **阶段 A（锁外）**：``prepare_fn`` —— 原文件 / 图片 / 解析 md 落存储，
          纯网络 IO，多个文档可并行；
        * **阶段 B（写锁内）**：``commit_fn`` —— 向量库「删旧行 + 插新行」、
          文档间 SimHash 去重与指纹写入。这一段必须同 collection 串行：去重是
          "先查再插"，并发会让近似重复文档同时通过检查；同 source 并发覆盖也会
          互相删除。

        未提供两阶段回调时退化为 ``persist_fn`` 单回调，整段在写锁内执行。
        锁由流水线强制，不依赖回调自觉，因此写入阶段可以有多个消费者。

        Raises:
            RuntimeError: 待写文档或持久化回调缺失。
        """
        parsed = item.parsed
        if parsed is None:
            raise RuntimeError(f"待写文档缺失（task_id={item.task_id}）")
        if self._write_lock is None:
            raise RuntimeError("入库流水线尚未启动")

        if item.commit_fn is not None:
            prepare_fn = item.prepare_fn
            if prepare_fn is None:
                raise RuntimeError(f"缺少阶段 A 回调（task_id={item.task_id}）")
            # 阶段 A：锁外，可与其他文档并行
            context = await prepare_fn(parsed, item.content_hash, dense_vectors)
            # 阶段 B：锁内，同 collection 串行
            async with self._write_lock:
                return await item.commit_fn(context)

        persist_fn = item.persist_fn
        if persist_fn is None:
            raise RuntimeError(f"持久化回调缺失（task_id={item.task_id}）")
        async with self._write_lock:
            return await persist_fn(parsed, item.content_hash, dense_vectors)

    async def _chunk_markdown(
        self,
        markdown: str,
        *,
        source: str,
        title: str,
        category: str,
        is_pdf: bool,
        blocks: tuple = (),
    ):
        """切块：满足条件走进程池，否则内联；进程池崩溃时自动降级。"""
        kwargs = {
            "source": source,
            "title": title,
            "category": category,
            "is_pdf": is_pdf,
            "blocks": tuple(blocks or ()),
        }
        if not use_process_pool(markdown):
            return chunk_markdown_inline(markdown, **kwargs)

        try:
            pool: ProcessPoolExecutor = get_chunk_pool()
        except Exception as exc:
            # 进程池不可用（受限环境）：降级内联
            logger.warning("切块进程池不可用，降级为内联切块：%s", exc)
            mark_chunk_pool_unavailable()
            return chunk_markdown_inline(markdown, **kwargs)

        loop = asyncio.get_running_loop()
        payload = (markdown, source, title, category, is_pdf, kwargs["blocks"])
        try:
            return await loop.run_in_executor(pool, chunk_document, payload)
        except (BrokenProcessPool, OSError, RuntimeError) as exc:
            # worker 崩溃或子进程无法创建（受限容器/沙箱禁止管道）：
            # 本次降级内联切块，并标记池不可用，避免后续文档反复重试
            logger.warning("切块进程池异常（%s），本次降级为内联切块", exc)
            mark_chunk_pool_unavailable()
            return chunk_markdown_inline(markdown, **kwargs)


# ---------------------------------------------------------------------------
# 进程单例
# ---------------------------------------------------------------------------

_pipeline: IngestionPipeline | None = None
_pipeline_lock = threading.Lock()


def get_ingestion_pipeline() -> IngestionPipeline:
    """获取（并缓存）入库流水线单例。"""
    global _pipeline
    if _pipeline is None:
        with _pipeline_lock:
            if _pipeline is None:
                _pipeline = IngestionPipeline()
    return _pipeline


async def drain_ingestion_pipeline(timeout: float | None = None) -> bool:
    """等待单例流水线的在途文档结算（服务收尾前的优雅排空）。"""
    pipeline = _pipeline
    if pipeline is None:
        return True
    return await pipeline.drain(timeout)


def shutdown_ingestion_pipeline(wait: bool = False) -> None:
    """关闭入库流水线单例（服务生命周期收尾时调用）。"""
    global _pipeline
    pipeline, _pipeline = _pipeline, None
    if pipeline is not None:
        pipeline.shutdown(wait=wait)


def shutdown_all_pipelines() -> None:
    """结算并关闭所有存活流水线实例（测试收尾与诊断用）。

    同步实现：关闭本身不需要 ``await``（取消任务 + 释放执行器），
    因此同步测试的收尾 fixture 也能安全调用。
    """
    global _pipeline
    _pipeline = None
    for pipeline in list(_live_pipelines):
        pipeline.shutdown()


__all__ = [
    "IngestionPipeline",
    "PipelineItem",
    "drain_ingestion_pipeline",
    "get_ingestion_pipeline",
    "shutdown_all_pipelines",
    "shutdown_ingestion_pipeline",
]
