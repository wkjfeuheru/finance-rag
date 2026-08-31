"""检索 Top-K 与重排序 Top-K 网格扫描评估脚本。

对「向量召回检索 top-k（``k``）」与「BGE 重排序 top-k（``rerank_top_n``）」
做二维网格扫描，仅用本地检索指标（hit_rate / MRR / NDCG，无 LLM 调用）
评估每个 ``(k, rerank_top_n)`` 组合的召回质量与单查询延迟，输出对比表、
CSV 与最优组合建议。

用法示例：

    # 默认网格 + 全量测试集
    python scripts/sweep_k.py

    # 自定义候选值 + 抽样
    python scripts/sweep_k.py --k-list 3,5,10,15,20 --rtn-list 3,5,10 --sample 10

    # 纯稠密检索模式 / 自定义测试集
    python scripts/sweep_k.py --use-dense-only --testset path/to/qa.md

结果落盘 ``scripts/results/k_sweep/<时间戳>/``。
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# 项目根目录（scripts/ 的上一级），保证以任意 CWD 运行时都能导入项目模块
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)

DEFAULT_TESTSET = (
    ROOT / "finance_rag" / "src" / "eval" / "data" / "evaluation_qa_generated.md"
)


def _parse_int_list(raw: str) -> list[int]:
    """解析逗号分隔的正整数列表，去重并升序返回。"""
    values: list[int] = []
    for part in (raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            value = int(part)
        except ValueError:
            raise ValueError(f"非法整数：{part!r}") from None
        if value <= 0:
            raise ValueError(f"K 取值必须为正整数：{value}")
        values.append(value)
    if not values:
        raise ValueError("候选值列表不能为空")
    return sorted(set(values))


def _load_entries(
    testset_path: Path,
    allow_missing: bool,
) -> tuple[list[Any], Any, list[tuple[Any, list[str]]]]:
    """加载测试集，剔除负样本与相关文档不在库内的条目。

    Returns:
        (保留条目, 知识库实例, 被剔除条目列表)
    """
    from finance_rag.src.eval.test_set import TestSetLoader
    from finance_rag.src.eval.testset_generator import filter_entries_by_kb
    from finance_rag.src.infrastructure.vector_store.milvus_kb import get_knowledge_base

    entries = TestSetLoader(testset_path).load_test_set()
    # 负样本无 related_docs，属于拒答评估而非召回质量，纳入会稀释指标
    entries = [e for e in entries if e.question_type != "negative"]

    kb = get_knowledge_base()
    entries, removed = filter_entries_by_kb(entries, kb, allow_missing=allow_missing)
    return entries, kb, removed


def _retrieve(
    kb: Any,
    query: str,
    k: int,
    rerank_top_n: int,
    use_dense_only: bool,
) -> tuple[list[dict[str, Any]], float]:
    """执行单次检索，返回 (结果列表, 耗时毫秒)。"""
    t0 = time.perf_counter()
    docs = kb.hybrid_search(
        query,
        k=k,
        use_rerank=True,
        rerank_top_n=rerank_top_n,
        expand_parents=True,
        use_dense_only=use_dense_only,
    )
    latency_ms = round((time.perf_counter() - t0) * 1000, 4)
    return docs, latency_ms


def _metric(entry: Any, docs: list[dict[str, Any]], rerank_top_n: int) -> dict[str, float]:
    """计算单条目的本地检索指标（hit_rate / mrr / ndcg）。"""
    from finance_rag.src.eval.ragas_eval import _compute_local_retrieval_metrics

    sources = [d.get("source", "") for d in docs if d.get("source")]
    return _compute_local_retrieval_metrics(sources, entry.related_docs, k=rerank_top_n)


def _best(rows: list[dict[str, Any]], metric: str) -> dict[str, Any] | None:
    """按指定指标选出最优组合（主指标 → mrr → ndcg → 更小 k/rtn → 更小延迟）。"""
    finite = [r for r in rows if r.get(metric) is not None]
    if not finite:
        return None
    return max(
        finite,
        key=lambda r: (
            r[metric],
            r.get("mrr") or 0.0,
            r.get("ndcg") or 0.0,
            -r["k"],
            -r["rerank_top_n"],
            -(r.get("avg_latency_ms") or 0.0),
        ),
    )


def run_sweep(args: argparse.Namespace) -> int:
    """执行网格扫描并输出报告，返回退出码。"""
    k_list = _parse_int_list(args.k_list)
    rtn_list = _parse_int_list(args.rtn_list)

    testset_path = Path(args.testset)
    if not testset_path.exists():
        print(f"测试集不存在：{testset_path}")
        return 1

    entries, kb, _removed = _load_entries(testset_path, args.allow_missing)
    if args.sample and 0 < args.sample < len(entries):
        entries = random.Random(args.seed).sample(entries, args.sample)
    if not entries:
        print("无可用测试条目（相关文档均不在知识库，或测试集为空）")
        return 1

    # 合法网格：k（候选数）>= rerank_top_n（最终输出数）
    grid = [(k, rtn) for k in k_list for rtn in rtn_list if k >= rtn]
    skipped = [(k, rtn) for k in k_list for rtn in rtn_list if k < rtn]
    for k, rtn in skipped:
        logger.warning("跳过非法组合 k=%d < rerank_top_n=%d", k, rtn)

    logger.info(
        "扫描 %d 个组合，测试条目 %d 条，dense_only=%s",
        len(grid), len(entries), args.use_dense_only,
    )

    rows: list[dict[str, Any]] = []
    for k, rtn in grid:
        hit_vals: list[float] = []
        mrr_vals: list[float] = []
        ndcg_vals: list[float] = []
        lat_vals: list[float] = []
        errors = 0
        for entry in entries:
            try:
                docs, latency = _retrieve(kb, entry.query, k, rtn, args.use_dense_only)
                m = _metric(entry, docs, rtn)
                hit_vals.append(m["hit_rate"])
                mrr_vals.append(m["mrr"])
                ndcg_vals.append(m["ndcg"])
                lat_vals.append(latency)
            except Exception as exc:
                errors += 1
                logger.warning("检索失败 k=%d rtn=%d query=%r: %s", k, rtn, entry.query, exc)

        rows.append({
            "k": k,
            "rerank_top_n": rtn,
            "entry_count": len(entries),
            "error_count": errors,
            "hit_rate": round(sum(hit_vals) / len(hit_vals), 4) if hit_vals else None,
            "mrr": round(sum(mrr_vals) / len(mrr_vals), 4) if mrr_vals else None,
            "ndcg": round(sum(ndcg_vals) / len(ndcg_vals), 4) if ndcg_vals else None,
            "avg_latency_ms": round(sum(lat_vals) / len(lat_vals), 4) if lat_vals else None,
        })

    outdir = Path(args.outdir) if args.outdir else ROOT / "scripts" / "results" / "k_sweep"
    run_dir = outdir / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    _write_csv(run_dir, rows)
    _write_report(run_dir, rows, testset_path, entries, args)
    print(_render_text(rows))

    logger.info("结果已写入 %s", run_dir)
    return 0


# ---------------------------------------------------------------------------
# 结果输出
# ---------------------------------------------------------------------------

_CSV_COLUMNS = [
    "k", "rerank_top_n", "entry_count", "error_count",
    "hit_rate", "mrr", "ndcg", "avg_latency_ms",
]


def _write_csv(run_dir: Path, rows: list[dict[str, Any]]) -> None:
    """写逐组合结果 CSV。"""
    with (run_dir / "per_combo.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _fmt(value: Any) -> str:
    return "N/A" if value is None else f"{float(value):.4f}"


def _render_text(rows: list[dict[str, Any]]) -> str:
    """控制台文本表 + 最优组合建议。"""
    lines = [
        "=" * 72,
        f"{'k':>6} {'rtn':>6} {'hit_rate':>10} {'mrr':>10} {'ndcg':>10} {'latency_ms':>12}",
        "-" * 72,
    ]
    for r in rows:
        lines.append(
            f"{r['k']:>6} {r['rerank_top_n']:>6} {_fmt(r['hit_rate']):>10} "
            f"{_fmt(r['mrr']):>10} {_fmt(r['ndcg']):>10} {_fmt(r['avg_latency_ms']):>12}"
        )
    lines.append("-" * 72)
    for label, metric in (("hit_rate", "hit_rate"), ("MRR", "mrr"), ("NDCG", "ndcg")):
        best = _best(rows, metric)
        if best:
            lines.append(
                f"{label} 最优：k={best['k']}, rerank_top_n={best['rerank_top_n']}"
            )
    return "\n".join(lines) + "\n"


def _write_report(
    run_dir: Path,
    rows: list[dict[str, Any]],
    testset_path: Path,
    entries: list[Any],
    args: argparse.Namespace,
) -> None:
    """写 Markdown 报告。"""
    lines = [
        "# K 扫描评估报告",
        "",
        f"- 测试集：`{testset_path.name}`；条目数：{len(entries)}；"
        f"dense_only：{args.use_dense_only}；sample：{args.sample or '全量'}；seed：{args.seed}",
        "",
        "| k | rerank_top_n | hit_rate | MRR | NDCG | avg_latency_ms |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['k']} | {r['rerank_top_n']} | {_fmt(r['hit_rate'])} | "
            f"{_fmt(r['mrr'])} | {_fmt(r['ndcg'])} | {_fmt(r['avg_latency_ms'])} |"
        )
    lines.append("")
    lines.append("## 最优组合建议")
    lines.append("")
    for label, metric in (("hit_rate", "hit_rate"), ("MRR", "mrr"), ("NDCG", "ndcg")):
        best = _best(rows, metric)
        if best:
            lines.append(
                f"- **{label} 最优**：k={best['k']}, rerank_top_n={best['rerank_top_n']} "
                f"（hit_rate={_fmt(best['hit_rate'])}, MRR={_fmt(best['mrr'])}, "
                f"NDCG={_fmt(best['ndcg'])}, latency={_fmt(best['avg_latency_ms'])}ms）"
            )
    lines.append("")
    (run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="检索 Top-K 与重排序 Top-K 网格扫描评估")
    parser.add_argument("--k-list", default="3,5,10,15,20", help="检索 top-k 候选值（逗号分隔，默认 3,5,10,15,20）")
    parser.add_argument("--rtn-list", default="3,5,10", help="重排序 top-k 候选值（逗号分隔，默认 3,5,10）")
    parser.add_argument("--testset", default=str(DEFAULT_TESTSET), help="测试集路径")
    parser.add_argument("--sample", type=int, default=0, help="抽样条目数（0=全量）")
    parser.add_argument("--seed", type=int, default=7, help="抽样随机种子（默认 7）")
    parser.add_argument("--allow-missing", action="store_true", help="保留相关文档不在库内的条目")
    parser.add_argument("--use-dense-only", action="store_true", help="走纯稠密检索（默认 RRF 混合）")
    parser.add_argument("--outdir", default=None, help="结果输出根目录")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    try:
        return run_sweep(args)
    except ValueError as exc:
        print(f"参数错误：{exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
