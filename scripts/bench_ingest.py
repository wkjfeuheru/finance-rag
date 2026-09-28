"""入库链路端到端耗时基准：单文档平均总耗时（分块 / 向量化 / 入库）。

测量口径
--------
* **提交耗时**：``upload_document_async`` 自身耗时（临时落盘 + 增量指纹检查 + 入队）。
  常驻流水线路径下入队即返回，因此这段应当很短。
* **入库耗时**：从入队返回（任务置 PROCESSING）到任务进入终态的墙钟耗时，
  即**解析 → 分块 → 向量化 → 写库**全链路。
* **端到端**：提交耗时 + 入库耗时。
* **分段耗时**：从 Prometheus 直方图 ``finance_rag_ingest_stage_seconds`` 读取各阶段
  累计值的增量，除以文档数得到单文档平均阶段耗时，用于定位瓶颈落在哪一段。

默认生成合成文档并**逐份**测量（得到"单文档平均总耗时"）。
加 ``--concurrent`` 则一次性全部提交，额外给出批量吞吐口径（总耗时 / 文档数）。

依赖
----
需要 Milvus 与对象存储可用（脚本会先做连通性预检）；``.md`` 文档由解析适配器直读，
不触发 MinerU 重模型，因此本基准可稳定复现。指向真实 PDF 目录（``--dir``）即可
把 MinerU 解析成本一并计入。

用法
----
    # 生成 5 份约 8 万字的合成文档并逐份测量
    python scripts/bench_ingest.py

    # 自定义规模
    python scripts/bench_ingest.py --count 8 --chars 120000

    # 用真实文档目录
    python scripts/bench_ingest.py --dir data/docs --ext pdf

    # 一次性提交（批量吞吐口径）
    python scripts/bench_ingest.py --count 10 --concurrent

    # 只生成文档不测量（离线自检）
    python scripts/bench_ingest.py --generate-only
"""

from __future__ import annotations

import argparse
import asyncio
import io
import socket
import statistics
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

# 允许从仓库根目录直接运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prometheus_client import REGISTRY  # noqa: E402
from starlette.datastructures import UploadFile  # noqa: E402

from finance_rag.src.core import config  # noqa: E402
from finance_rag.src.services.document_service import get_document_manager  # noqa: E402
from finance_rag.src.services.ingestion_pipeline import (  # noqa: E402
    shutdown_ingestion_pipeline,
)
from finance_rag.src.services.task_service import TaskStatus, get_task_manager  # noqa: E402

STAGES = ("parse", "chunk", "embed", "write")
STAGE_LABELS = {
    "parse": "解析",
    "chunk": "分块",
    "embed": "向量化",
    "write": "入库写库",
}

DEFAULT_OUT_DIR = Path("assets") / "bench"


# ---------------------------------------------------------------------------
# 合成文档生成
# ---------------------------------------------------------------------------

_CHAPTER_TEMPLATE = """\
## 第{chapter}章 {chapter_title}

### 第{section}节 适用范围与基本要求

本章适用于{subject}开展{object}相关业务的全过程管理。
{subject}负责人对本条线合规执行情况负首要责任，并应当按照每{months}个月一次的频度向合规管理部门提交执行情况报告。
与{object}相关的业务档案应当自业务终止之日起至少保存{days}个月备查。
{subject}在{object}业务中形成的客户材料，应当按{action}要求分类归档，不得与一般业务材料混放。

#### 第{article}条

{body}

#### 第{article_next}条

{body_two}

| 项目 | 限额 | 审批层级 | 报备要求 |
| --- | --- | --- | --- |
| 单笔交易 | {limit} 万元 | {role} | 无需报备 |
| 单日累计 | {limit2} 万元 | 分管领导 | {deadline}内报备 |
| 单月累计 | {limit3} 万元 | 风险管理委员会 | 月度报备 |

上述限额自本条第{article}条施行之日起执行。
{subject}可以根据市场环境变化提出调整方案，调整方案应当经风险管理委员会审议通过后{action}。
调整前后的限额应当在同一份文件中留档，便于事后审计追溯。

"""

# 正文生成：片段组合式（多槽位独立取词 × 多句式模板）。
#
# 为什么不能"固定句式循环拼接"：那样各段之间字符 3-gram 高度重合，会被入库流水线
# 自己的段落级 SimHash 去重判定为近似重复并删除——实测 189 段中删掉 119 段，
# 只剩约 30% 的文档体积进入向量化，基准结果随之失真（去重本身是正确的）。
# 这里让「主体 / 事项 / 动作 / 时限 / 记录 / 角色」六个槽位连同数值一起变化，
# 单句内即有约一半字符互不相同，段落之间不再近似重复；同时保留少量真实样板文字
# （页眉页脚、版权行）供清洗链路发挥作用。
_SUBJECTS = (
    "业务部门", "分支机构", "客户经理岗", "运营管理部门",
    "资金交易岗", "授信审批岗", "风险管理部", "内控合规岗",
)
_OBJECTS = (
    "大额资金划转", "关联交易审批", "客户身份核验", "产品适当性评估",
    "可疑交易报告", "信用风险敞口监测", "抵质押物估值", "跨境收支申报",
    "代销产品准入", "账户开立与变更",
)
_ACTIONS = (
    "履行审批手续", "完成双人复核", "提交书面备案", "开展现场检查",
    "组织专项审计", "报送监管数据", "留存操作痕迹", "更新风险台账",
)
_DEADLINES = (
    "三个工作日", "五个工作日", "十个工作日", "十五个自然日",
    "二十个工作日", "当日", "次日", "月末前",
)
_RECORDS = ("业务档案", "风险台账", "客户资料", "交易凭证", "审批记录", "检查底稿")
_ROLES = ("部门负责人", "分管领导", "风险管理委员会", "合规管理部门负责人")

_BODY_TEMPLATES = (
    "{subject}办理{object}时，应当{action}，并在{deadline}内完成，相关{record}由{role}留存{months}个月备查。",
    "{subject}发现{object}存在异常情形的，应当在{deadline}内{action}，并向合规管理部门报告，报告应当包含{n}项要素。",
    "{subject}每{months}个月对{object}开展一次专项检查，检查完成后应当{action}，相关{record}留存{days}个月。",
    "对{object}涉及金额超过人民币{amount}万元的情形，{subject}应当{action}，并由{role}在{deadline}内复核。",
    "{subject}就{object}制定实施细则的，不得低于本办法规定的标准，细则应当由{role}在{deadline}内完成{action}。",
    "{role}在对{object}进行审查时，可以要求{subject}补充{record}，补充期限一般不超过{deadline}。",
    "{subject}未按规定就{object}{action}的，由{role}责令限期整改，并可处人民币{amount2}万元以下的内部罚款。",
    "本办法所称{object}重大事项，是指涉及金额超过人民币{amount2}万元或者需要{role}知悉并{action}的事项。",
    "{subject}应当于每{months}个月末就{object}的{record}进行汇总核对，核对结果由{role}签字确认。",
    "涉及{n}户以上客户的{object}业务，{subject}应当提前{deadline}向{role}报备，并{action}。",
)


def _render_sentence(article: int, step: int) -> str:
    """按 (条号, 句序) 组合出一个句子；六个槽位各自独立取词，保证段间不近似重复。"""
    template = _BODY_TEMPLATES[(article * 7 + step) % len(_BODY_TEMPLATES)]
    return template.format(
        subject=_SUBJECTS[(article * 3 + step) % len(_SUBJECTS)],
        object=_OBJECTS[(article * 5 + step * 2) % len(_OBJECTS)],
        action=_ACTIONS[(article * 2 + step * 3) % len(_ACTIONS)],
        deadline=_DEADLINES[(article + step * 5) % len(_DEADLINES)],
        record=_RECORDS[(article * 4 + step) % len(_RECORDS)],
        role=_ROLES[(article + step * 7) % len(_ROLES)],
        amount=50 + (article * 7 + step * 13) % 950,
        amount2=100 + (article * 11 + step * 17) % 900,
        days=12 + (article * 3 + step * 5) % 48,
        n=5 + (article * 2 + step * 3) % 95,
        months=3 + (article + step) % 10,
    )


def _compose_body(article: int, target_chars: int) -> str:
    """把若干互异句子拼成条款正文，凑到目标长度（模拟真实条款体量）。"""
    parts: list[str] = []
    written = 0
    step = 0
    while written < target_chars:
        sentence = _render_sentence(article, step)
        parts.append(sentence)
        written += len(sentence)
        step += 1
    return "".join(parts)


def _generate_document(index: int, target_chars: int) -> str:
    """生成一份结构化的中文金融制度类文档（内容确定，便于复现）。

    正文体量刻意贴近真实制度文件（每条 600~900 字），使父块按标题层级自然形成、
    子块按 ``DOCLING_CHUNK_MAX_TOKENS`` 正常细分；若正文由固定句式循环拼接，
    段落之间会近似重复而被 SimHash 去重删除，基准就只能测到文档的一部分。
    """
    parts = [
        f"# 金融业务合规管理办法（测试文档 {index}）\n\n",
        "本文档用于入库链路耗时基准测试，内容由脚本合成，不具有业务含义。\n\n",
    ]
    chapter = 0
    written = sum(len(part) for part in parts)
    while written < target_chars:
        chapter += 1
        article = chapter * 3
        block = _CHAPTER_TEMPLATE.format(
            chapter=chapter,
            chapter_title="业务办理与风险控制",
            section=chapter,
            article=article,
            article_next=article + 1,
            subject=_SUBJECTS[(chapter + index) % len(_SUBJECTS)],
            object=_OBJECTS[(chapter * 2 + index) % len(_OBJECTS)],
            action=_ACTIONS[(chapter + index) % len(_ACTIONS)],
            role=_ROLES[(chapter + index) % len(_ROLES)],
            deadline=_DEADLINES[(chapter * 3 + index) % len(_DEADLINES)],
            record=_RECORDS[(chapter + index) % len(_RECORDS)],
            months=3 + chapter % 10,
            days=12 + (chapter * 5) % 48,
            body=_compose_body(article, 700),
            body_two=_compose_body(article + 1, 600),
            limit=chapter * 100,
            limit2=chapter * 500,
            limit3=chapter * 2000,
        )
        parts.append(block)
        written += len(block)

    # 页眉页脚噪声：清洗流水线应当将其剔除（验证清洗链路真实生效）
    parts.append("\n\n版权所有 · 内部资料，未经许可不得外传\n\n第 1 页\n\n")
    return "".join(parts)


def _prepare_documents(args: argparse.Namespace) -> list[Path]:
    """准备待入库文档：使用 --dir 下的真实文件，或生成合成文档。"""
    if args.dir:
        source_dir = Path(args.dir)
        if not source_dir.is_dir():
            raise SystemExit(f"目录不存在：{source_dir}")
        files = sorted(
            path
            for path in source_dir.iterdir()
            if path.is_file() and path.suffix.lower() == f".{args.ext}"
        )
        if not files:
            raise SystemExit(f"目录中未找到 .{args.ext} 文件：{source_dir}")
        selected = files[: args.count] if args.count else files
        print(f"使用真实文档 {len(selected)} 份（来自 {source_dir}）：")
        for path in selected:
            print(f"  - {path.name}（{path.stat().st_size / 1024:.0f} KB）")
        return selected

    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    char_counts: list[int] = []
    for index in range(1, args.count + 1):
        path = out_dir / f"bench-doc-{index}.md"
        text = _generate_document(index, args.chars)
        path.write_text(text, encoding="utf-8")
        paths.append(path)
        # 进程池阈值比较的是**字符数**（use_process_pool 用 len(markdown)），
        # 不能用字节数——中文 UTF-8 一个字符 3 字节，会判断错
        char_counts.append(len(text))
    avg_chars = statistics.mean(char_counts)
    avg_kb = statistics.mean(p.stat().st_size for p in paths) / 1024
    pool_note = (
        "走进程池切块"
        if avg_chars >= config.INGEST_CHUNK_MIN_CHARS
        else f"低于 INGEST_CHUNK_MIN_CHARS={config.INGEST_CHUNK_MIN_CHARS}，走内联切块"
    )
    print(f"已生成合成文档 {len(paths)} 份，平均 {avg_chars:.0f} 字符 / {avg_kb:.0f} KB（{pool_note}）")
    print(f"输出目录：{out_dir}")
    return paths


# ---------------------------------------------------------------------------
# 预检与配置回显
# ---------------------------------------------------------------------------

def _preflight() -> None:
    """检查 Milvus 连通性，并回显本次生效的并发配置。"""
    uri = config.MILVUS_URI
    parsed = urlparse(uri if "//" in uri else f"http://{uri}")
    host = parsed.hostname or "localhost"
    port = parsed.port or 19530
    try:
        with socket.create_connection((host, port), timeout=3):
            milvus_ok = True
    except OSError as exc:
        milvus_ok = False
        print(f"✗ Milvus 不可达（{host}:{port}）：{exc}")
        print("  请先启动：docker-compose up -d etcd minio milvus")
        print("  （若已有独立实例，设置 MILVUS_URI 后重试）")

    print("本次生效的入库配置：")
    print(f"  MILVUS_URI            = {uri}")
    print(f"  STORAGE_BACKEND       = {config.STORAGE_BACKEND}")
    print(f"  解析并发 (worker)      = {config.resolve_parse_concurrency()}")
    print(f"  切块并发 (常驻消费者)  = {config.resolve_chunk_concurrency()}"
          f"（进程池 worker = {config.resolve_chunk_workers()}）")
    print(f"  向量化并发             = {config.INGEST_EMBED_WORKERS}")
    print(f"  写入并发               = {config.resolve_write_concurrency()}")
    print(f"  队列容量               = {config.INGEST_QUEUE_MAXSIZE}")
    print(f"  切块内联阈值           = {config.INGEST_CHUNK_MIN_CHARS} 字符")
    print(f"  常驻流水线             = {config.INGEST_PIPELINE_ENABLED}")
    print()

    if not milvus_ok:
        raise SystemExit("预检失败：Milvus 未就绪，无法测量入库链路。")


def _stage_total(stage: str) -> float:
    """读取某阶段累计耗时（Prometheus 直方图 sum）。"""
    value = REGISTRY.get_sample_value(
        "finance_rag_ingest_stage_seconds_sum", {"stage": stage}
    )
    return float(value or 0.0)


# ---------------------------------------------------------------------------
# 单文档测量
# ---------------------------------------------------------------------------

async def _ingest_one(dm, path: Path, category: str) -> dict:
    """入库一份文档并返回耗时明细。"""
    content = path.read_bytes()
    upload = UploadFile(file=io.BytesIO(content), filename=path.name)
    tm = get_task_manager()

    started = time.perf_counter()
    submitted = await dm.upload_document_async(upload, category=category)
    submit_seconds = time.perf_counter() - started

    task_id = submitted["task_id"]
    ingest_started = time.perf_counter()
    while True:
        task = tm.get(task_id)
        if task is None:
            raise RuntimeError(f"任务丢失：{task_id}")
        if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
            break
        await asyncio.sleep(0.02)
    ingest_seconds = time.perf_counter() - ingest_started

    result = task.result or {}
    return {
        "filename": path.name,
        "task_id": task_id,
        "status": task.status,
        "error": task.error,
        "result": result,
        "skipped": bool(result.get("skipped")),
        "submit_seconds": submit_seconds,
        "ingest_seconds": ingest_seconds,
        "total_seconds": submit_seconds + ingest_seconds,
    }


async def _run(args: argparse.Namespace, paths: list[Path]) -> int:
    dm = get_document_manager(args.kb)
    before = {stage: _stage_total(stage) for stage in STAGES}

    print(f"\n开始入库测量：{len(paths)} 份文档，"
          f"{'一次性全部提交' if args.concurrent else '逐份提交'}…\n")

    batch_started = time.perf_counter()
    if args.concurrent:
        records = await asyncio.gather(
            *(_ingest_one(dm, path, args.category) for path in paths)
        )
    else:
        records = []
        for path in paths:
            record = await _ingest_one(dm, path, args.category)
            records.append(record)
            flag = "OK" if record["status"] is TaskStatus.COMPLETED else "FAIL"
            chunks = record["result"].get("chunk_count")
            chunk_text = f"{chunks} 块" if chunks is not None else "-"
            print(
                f"  [{flag}] {record['filename']:<22} "
                f"{chunk_text:>7} | "
                f"提交 {record['submit_seconds']:.3f}s | "
                f"入库 {record['ingest_seconds']:.3f}s | "
                f"合计 {record['total_seconds']:.3f}s"
            )
    batch_seconds = time.perf_counter() - batch_started

    after = {stage: _stage_total(stage) for stage in STAGES}

    # ---- 结果汇总 ----
    failed = [r for r in records if r["status"] is TaskStatus.FAILED]
    skipped = [r for r in records if r["skipped"]]
    ok = [r for r in records if r["status"] is TaskStatus.COMPLETED and not r["skipped"]]

    print("\n" + "=" * 68)
    print("结果")
    print("=" * 68)

    if failed:
        print(f"✗ 有 {len(failed)} 份文档入库失败（计入平均前请先修复）：")
        for record in failed[:5]:
            print(f"    {record['filename']}: {record['error']}")
    if skipped:
        print(f"✗ 有 {len(skipped)} 份文档命中增量跳过（内容未变更），测量无效：")
        for record in skipped[:5]:
            print(f"    {record['filename']}")
        print("  原因：同 source + 同内容哈希的文档会跳过入库。")
        print("  处理：确保脚本的清理步骤生效（勿用 --no-cleanup 重复跑），"
              "或改用 --dir 指向新的文档。")

    if not ok:
        print("\n无有效样本，无法给出耗时。")
        return 1

    totals = [r["total_seconds"] for r in ok]
    submits = [r["submit_seconds"] for r in ok]
    ingests = [r["ingest_seconds"] for r in ok]

    print(f"有效样本：{len(ok)} / {len(records)} 份\n")
    print(f"{'指标':<16}{'平均':>10}{'最小':>10}{'最大':>10}")
    print("-" * 46)
    for label, values in (
        ("提交耗时(s)", submits),
        ("入库耗时(s)", ingests),
        ("端到端(s)", totals),
    ):
        print(
            f"{label:<16}{statistics.mean(values):>10.3f}"
            f"{min(values):>10.3f}{max(values):>10.3f}"
        )

    # 分段平均（来自流水线内置的阶段耗时直方图）
    deltas = {stage: after[stage] - before[stage] for stage in STAGES}
    stage_sum = sum(deltas.values())
    if stage_sum > 0:
        print("\n单文档平均分段耗时（流水线各阶段累计值 / 文档数）：")
        print(f"{'阶段':<12}{'平均(s)':>10}{'占比':>10}")
        print("-" * 34)
        for stage in STAGES:
            avg = deltas[stage] / len(records)
            share = deltas[stage] / stage_sum * 100 if stage_sum else 0.0
            print(f"{STAGE_LABELS[stage]:<12}{avg:>10.3f}{share:>9.1f}%")
        print("-" * 34)
        print(f"{'合计':<12}{stage_sum / len(records):>10.3f}")
        slowest = max(STAGES, key=lambda s: deltas[s])
        print(f"\n瓶颈段：{STAGE_LABELS[slowest]}"
              f"（占流水线耗时 {deltas[slowest] / stage_sum * 100:.1f}%）")

    chunk_counts = [int(r["result"].get("chunk_count") or 0) for r in ok]
    avg_chunks = statistics.mean(chunk_counts) if chunk_counts else 0.0
    if avg_chunks:
        print(f"\n平均切块数：{avg_chunks:.0f} 块/篇")
        print(f"平均单块耗时：{statistics.mean(totals) / avg_chunks * 1000:.1f} 毫秒/块"
              f"（便于换算不同篇幅文档）")

    print(f"\n单文档平均端到端总耗时：{statistics.mean(totals):.2f} 秒"
          f"（n={len(ok)}，中位数 {statistics.median(totals):.2f}s）")
    if args.concurrent:
        print(f"批量总耗时：{batch_seconds:.2f} 秒 → "
              f"平均 {batch_seconds / len(records):.2f} 秒/份（含阶段重叠）")

    # ---- 清理（保证下次运行仍是真实入库） ----
    if args.cleanup:
        print("\n清理已入库文档…")
        for record in records:
            source = record["result"].get("source") or record["filename"]
            if record["status"] is not TaskStatus.COMPLETED:
                continue
            try:
                await asyncio.to_thread(dm.delete_document, source)
            except Exception as exc:  # noqa: BLE001 - 清理失败不掩盖测量结果
                print(f"  清理失败 {source}: {exc}")
        print("  完成（未清理时下次运行会命中增量跳过，导致测量无效）")

    return 0 if not failed and not skipped else 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="入库链路端到端耗时基准（分块 / 向量化 / 入库）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dir", default="", help="使用该目录下的真实文档（而非合成文档）")
    parser.add_argument("--ext", default="md", help="--dir 模式下筛选的文件扩展名")
    parser.add_argument("--count", type=int, default=5, help="文档份数")
    parser.add_argument("--chars", type=int, default=80000, help="合成文档目标字符数")
    parser.add_argument("--kb", default="", help="目标知识库集合名（空 = 默认）")
    parser.add_argument("--category", default="", help="文档分类（如 compliance_risk）")
    parser.add_argument("--concurrent", action="store_true", help="一次性提交全部文档（批量口径）")
    parser.add_argument("--no-cleanup", dest="cleanup", action="store_false",
                        help="测量后不删除已入库文档（下次运行会因增量跳过而失效）")
    parser.add_argument("--generate-only", action="store_true", help="只生成文档不测量")
    parser.set_defaults(cleanup=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    # --generate-only 用于离线自检（不依赖 Milvus），因此跳过预检
    if not args.generate_only:
        _preflight()
    paths = _prepare_documents(args)
    if args.generate_only:
        print("\n--generate-only：已生成文档，跳过测量。")
        return 0
    try:
        return asyncio.run(_run(args, paths))
    finally:
        # 常驻流水线持有进程池/线程池，必须显式释放，否则进程可能无法退出
        shutdown_ingestion_pipeline()


if __name__ == "__main__":
    raise SystemExit(main())
