"""A/B 实验臂执行器与编排（子进程隔离 + Ragas 批量评估）。

臂执行模型：
* 每臂在独立子进程中运行（``python -m finance_rag.src.eval.ab_runner``），
  臂的环境变量覆盖（``ABArm.env_overrides``）随子进程 env 传入，
  优先级高于 ``.env``，保证配置开关在进程内真实生效；
* 入库臂先按臂开关把 ``files/`` 索引进独立集合 ``finance_kb_ab_*``；
* 逐题执行 检索 → 生成 →（可选拒答策略），随后 Ragas 批量评估；
* 父进程编排双臂（顺序按 seed 随机）、聚合对比并输出报告。

用法（子进程入口）::

    python -m finance_rag.src.eval.ab_runner \
        --experiment hyde_on_off --arm hyde_on --testset <md> --out <json>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)

# 负样本拒绝判定关键词
_REJECTION_KEYWORDS = (
    "无法回答", "未检索到", "暂未检索", "暂无相关", "无相关信息",
    "知识库中未", "无法给出可靠回答", "拒绝回答",
)


def _negative_rejection_score(response: str) -> float:
    """负样本是否正确拒绝（含拒绝表述或空回答）返回 1.0，否则 0.0。"""
    if not response or not response.strip():
        return 1.0
    return 1.0 if any(kw in response for kw in _REJECTION_KEYWORDS) else 0.0


# ---------------------------------------------------------------------------
# 入库臂（ingest）：把 files/ 文档按臂开关索引进独立集合
# ---------------------------------------------------------------------------

# 入库臂文档源目录（约定为项目根 files/）
_ARM_DOCS_DIR = ROOT / "files"

# 入库臂支持的可解析文件扩展名
_SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf"}


def collect_kb_files(docs_dir: Path) -> list[Path]:
    """递归收集文档源目录下的可解析文件（md/txt/pdf）。"""
    if not docs_dir.exists():
        return []
    return [
        p for p in sorted(docs_dir.rglob("*"))
        if p.is_file() and p.suffix.lower() in _SUPPORTED_EXTENSIONS
    ]


def _arm_collection_name(experiment: Any, arm: Any) -> str:
    """入库臂独立集合名：``finance_kb_ab_<实验名>_<臂名>``。"""
    return f"finance_kb_ab_{experiment.name}_{arm.name}"


def _index_arm_collection(collection: str, docs_dir: Path) -> Any:
    """把 docs_dir 文档按当前进程开关（如 ENABLE_SEMANTIC_CHUNKER）索引进独立集合。

    当前进程的 ``ENABLE_SEMANTIC_CHUNKER`` 由臂 env_overrides 注入，故
    ``KnowledgeBase`` 初始化时 ``get_chunker()`` 会选用对应切块器，
    实现「语义+层级 vs 仅层级」的索引差异。

    Returns:
        索引完成的 KnowledgeBase 实例。
    """
    from finance_rag.src.infrastructure.vector_store.milvus_kb import KnowledgeBase

    kb = KnowledgeBase(collection_name=collection)
    kb.ensure_collection()
    files = collect_kb_files(docs_dir)
    for path in files:
        try:
            parsed = kb.parse_document(
                path, source=path.name, title=path.stem
            )
            kb.add_parsed_document(parsed)
        except Exception as exc:
            logger.warning("入库臂索引失败（%s）：%s", path.name, exc)
    doc_count = len(kb.list_documents())
    logger.info("入库臂集合 %s 索引完成：%d 文档", collection, doc_count)
    return kb


# ---------------------------------------------------------------------------
# 检索 / 生成（臂开关在进程内已生效）
# ---------------------------------------------------------------------------

def _retrieve_for_arm(experiment: Any, arm: Any, kb: Any, query: str) -> list[dict[str, Any]]:
    """按臂定义执行检索。"""
    from finance_rag.src.core.config import CHAT_RERANK_TOP_K, CHAT_TOP_K
    from finance_rag.src.services.chat_service import reranking_enabled

    strategy = arm.strategy_overrides

    # 入库实验：针对臂的独立集合 kb 直接检索（RRF 混合 + 重排序）
    if experiment.kind == "ingest":
        return kb.hybrid_search(
            query,
            k=CHAT_TOP_K,
            expand_parents=True,
            use_rerank=reranking_enabled(True),
            rerank_top_n=CHAT_RERANK_TOP_K,
        )

    if strategy.get("use_dense_only"):
        return kb.hybrid_search(
            query,
            k=int(strategy.get("k", CHAT_TOP_K)),
            use_dense_only=True,
            expand_parents=True,
            use_rerank=reranking_enabled(True),
            rerank_top_n=int(strategy.get("rerank_top_n", CHAT_RERANK_TOP_K)),
        )

    if arm.retrieval == "langgraph":
        return _langgraph_retrieve(query)

    from finance_rag.src.services.chat_service import retrieve_pipeline

    pipeline = asyncio.run(
        retrieve_pipeline(query, infer_filters=False)
    )
    return pipeline["docs"]


def _langgraph_retrieve(query: str) -> list[dict[str, Any]]:
    """LangGraph 臂：改写 → 检索 → 反思重检（复用线上图公开检索入口，不含生成）。"""
    from finance_rag.src.orchestration.graph import run_langgraph_retrieval

    return run_langgraph_retrieval(query)


def _generate_answer(query: str, contexts: list[str]) -> str:
    """用生产回答 Prompt 生成答案（臂开关如温度已在进程内生效）。"""
    from finance_rag.src.agent.prompts.chat import ANSWER_PROMPT
    from finance_rag.src.eval.ragas_eval import build_context_text
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from finance_rag.src.core.config import get_model

    if get_model() is None:
        raise RuntimeError("LLM 未配置：请设置 DEEPSEEK_API_KEY")
    chain = (
        ChatPromptTemplate.from_messages([("human", ANSWER_PROMPT)])
        | get_model()
        | StrOutputParser()
    )
    return str(chain.invoke({
        "context": build_context_text(contexts),
        "query": query,
    }).strip())


# ---------------------------------------------------------------------------
# 臂执行（子进程内）
# ---------------------------------------------------------------------------

def execute_arm(
    experiment: Any,
    arm: Any,
    *,
    testset_path: Path,
    fast: bool,
    sample: int,
    seed: int,
    allow_missing: bool,
) -> dict[str, Any]:
    """执行单臂：逐题检索/生成 → Ragas 批量评估 → 聚合。"""
    from finance_rag.src.eval.ragas_eval import (
        RagasBatchEvaluator,
        _compute_local_retrieval_metrics,
        compute_composite_score,
    )
    from finance_rag.src.eval.test_set import TestSetLoader
    from finance_rag.src.eval.testset_generator import filter_entries_by_kb
    from finance_rag.src.infrastructure.vector_store.milvus_kb import get_knowledge_base

    if experiment.kind == "ingest":
        # 入库臂：按臂开关把 files/ 文档索引进独立集合，评估在该集合上进行
        collection = _arm_collection_name(experiment, arm)
        kb = _index_arm_collection(collection, _ARM_DOCS_DIR)
    else:
        kb = get_knowledge_base()
        collection = kb.collection_name

    entries = TestSetLoader(Path(testset_path)).load_test_set()
    entries, removed = filter_entries_by_kb(entries, kb, allow_missing=allow_missing)
    if sample and 0 < sample < len(entries):
        entries = random.Random(seed).sample(entries, sample)
    if not entries:
        raise RuntimeError("测试集为空（相关文档均不在知识库）")

    evaluator = RagasBatchEvaluator(kb)
    ragas_rows: list[dict[str, Any]] = []
    for entry in entries:
        errors: list[dict[str, str]] = []
        t0 = time.perf_counter()
        try:
            docs = _retrieve_for_arm(experiment, arm, kb, entry.query)
        except Exception as exc:
            docs = []
            errors.append({"stage": "retrieval", "message": str(exc)})
        retrieval_ms = round((time.perf_counter() - t0) * 1000, 4)

        contexts = [d.get("content", "") for d in docs if d.get("content")]
        sources = [
            {
                "title": d.get("title", ""),
                "source": d.get("source", ""),
                "score": round(float(d.get("score", 0.0) or 0.0), 4),
                "rerank_score": (
                    round(float(d["rerank_score"]), 4)
                    if isinstance(d.get("rerank_score"), (int, float)) else None
                ),
                "content": (d.get("content") or "")[:500],
            }
            for d in docs
        ]

        t0 = time.perf_counter()
        try:
            response = _generate_answer(entry.query, contexts)
        except Exception as exc:
            response = ""
            errors.append({"stage": "generation", "message": str(exc)})
        generation_ms = round((time.perf_counter() - t0) * 1000, 4)

        ragas_rows.append({
            "user_input": entry.query,
            "retrieved_contexts": contexts,
            "response": response,
            "reference": entry.ground_truth,
            "related_docs": entry.related_docs,
            "test_type": entry.test_type,
            "difficulty": entry.difficulty,
            "question_type": entry.question_type,
            "chunk_ids": list(entry.chunk_ids),
            "sources": sources,
            "retrieval_latency_ms": retrieval_ms,
            "generation_latency_ms": generation_ms,
            "errors": errors,
        })

    t0 = time.perf_counter()
    evaluation = evaluator.evaluate_batch(ragas_rows, fast=fast)
    ragas_ms = round((time.perf_counter() - t0) * 1000, 4)

    # 逐题：本地检索指标 + 综合分 + 分类型本地指标
    per_query = evaluation["per_query"]
    for i, entry in enumerate(entries):
        if i >= len(per_query):
            break
        retrieved_sources = [s["source"] for s in per_query[i]["sources"]]
        local = _compute_local_retrieval_metrics(
            retrieved_sources, entry.related_docs, k=5
        )
        per_query[i]["metrics"].update(local)

        # 多跳：相关文档全部被召回才算命中（跨文档证据覆盖）
        if entry.question_type == "multi_hop":
            related_stems = {
                Path(s).stem for s in re.split(r"[,，;；|]", entry.related_docs or "") if s.strip()
            }
            hit_stems = {Path(s).stem for s in retrieved_sources}
            per_query[i]["metrics"]["multi_hop_hit"] = (
                1.0 if related_stems and related_stems <= hit_stems else 0.0
            )
        # 负样本：是否正确拒绝（回答含拒绝表述或为空）
        if entry.question_type == "negative":
            per_query[i]["metrics"]["negative_rejection"] = _negative_rejection_score(
                per_query[i].get("response", "")
            )

        composite = compute_composite_score(per_query[i]["metrics"], tier="all")
        per_query[i]["metrics"]["composite_score"] = composite["composite_score"]

    # 聚合：各指标均值（仅有限值）
    metric_names: set[str] = set()
    for item in per_query:
        metric_names.update(item["metrics"].keys())
    aggregates: dict[str, float | None] = {}
    for name in sorted(metric_names):
        finite = [
            item["metrics"][name]
            for item in per_query
            if isinstance(item["metrics"].get(name), (int, float))
        ]
        aggregates[name] = round(sum(finite) / len(finite), 4) if finite else None

    # 按问题类型分层统计（single_hop / multi_hop / negative）
    metrics_by_type: dict[str, dict[str, float | None]] = {}
    for item in per_query:
        qtype = item.get("question_type", "single_hop")
        bucket = metrics_by_type.setdefault(qtype, {})
        for name, value in item.get("metrics", {}).items():
            if isinstance(value, (int, float)):
                bucket.setdefault(name, []).append(value)
    for qtype, values in metrics_by_type.items():
        metrics_by_type[qtype] = {
            name: round(sum(vals) / len(vals), 4)
            for name, vals in values.items()
            if vals
        }

    return {
        "experiment": experiment.name,
        "arm": arm.to_dict(),
        "collection": collection,
        "testset": str(testset_path),
        "fast": fast,
        "sample": sample,
        "seed": seed,
        "entry_count": len(entries),
        "removed_entries": [
            {"query": e.query, "missing": missing} for e, missing in removed
        ],
        "metrics": aggregates,
        "metrics_by_type": metrics_by_type,
        "per_query": per_query,
        "metric_errors": evaluation["metric_errors"],
        "fallback_metrics": evaluation["fallback_metrics"],
        "ragas_latency_ms": ragas_ms,
        "success": True,
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
    }


# ---------------------------------------------------------------------------
# 对比与报告（父进程）
# ---------------------------------------------------------------------------

def compare_arms(
    arm_a_result: dict[str, Any] | None,
    arm_b_result: dict[str, Any] | None,
) -> dict[str, Any]:
    """聚合双臂指标，计算 Δ / 相对变化 / 胜出。"""
    def _metrics(result: dict[str, Any] | None) -> dict[str, float | None]:
        return (result or {}).get("metrics") or {}

    a, b = _metrics(arm_a_result), _metrics(arm_b_result)
    rows: dict[str, dict[str, Any]] = {}
    for name in sorted(set(a) | set(b)):
        av, bv = a.get(name), b.get(name)
        delta = (
            round(bv - av, 4)
            if isinstance(av, (int, float)) and isinstance(bv, (int, float))
            else None
        )
        relative_pct = (
            round(delta / av * 100, 2)
            if delta is not None and isinstance(av, (int, float)) and av
            else None
        )
        winner = None
        if delta is not None:
            if delta > 0:
                winner = "B"
            elif delta < 0:
                winner = "A"
            else:
                winner = "tie"
        rows[name] = {
            "arm_a": av, "arm_b": bv,
            "delta": delta, "relative_pct": relative_pct, "winner": winner,
        }
    return {"metrics": rows,
            "arm_a_error": (arm_a_result or {}).get("error"),
            "arm_b_error": (arm_b_result or {}).get("error")}


def run_experiment(
    experiment: Any,
    *,
    testset_path: Path,
    fast: bool,
    sample: int,
    seed: int,
    allow_missing: bool,
    outdir: Path | None,
) -> dict[str, Any]:
    """编排双臂子进程执行并输出报告。"""
    outdir = Path(outdir) if outdir else ROOT / "scripts" / "results" / "ab"
    run_dir = outdir / experiment.name / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    arms = [experiment.arm_a, experiment.arm_b]
    rng = random.Random(seed)
    ordered = list(arms) if rng.random() < 0.5 else list(reversed(arms))
    logger.info("臂执行顺序：%s", " → ".join(a.name for a in ordered))

    results: dict[str, Any] = {}
    for arm in ordered:
        arm_out = run_dir / f"{arm.name}.json"
        cmd = [
            sys.executable, "-m", "finance_rag.src.eval.ab_runner",
            "--experiment", experiment.name,
            "--arm", arm.name,
            "--testset", str(testset_path),
            "--out", str(arm_out),
            "--sample", str(sample),
            "--seed", str(seed),
        ]
        if fast:
            cmd.append("--fast")
        if allow_missing:
            cmd.append("--allow-missing")
        env = {**os.environ, **arm.env_overrides}
        logger.info("执行臂 %s（env=%s）", arm.name, arm.env_overrides or "{}")
        proc = subprocess.run(
            cmd, env=env, cwd=str(ROOT),
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        if proc.returncode != 0 or not arm_out.exists():
            results[arm.name] = {
                "error": f"returncode={proc.returncode}",
                "stderr_tail": (proc.stderr or "")[-2000:],
            }
            continue
        try:
            results[arm.name] = json.loads(arm_out.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            results[arm.name] = {"error": f"结果解析失败：{exc}"}

    comparison = compare_arms(
        results.get(experiment.arm_a.name),
        results.get(experiment.arm_b.name),
    )
    report = {
        "experiment": experiment.to_dict(),
        "testset": str(testset_path),
        "fast": fast,
        "sample": sample,
        "seed": seed,
        "arm_order": [a.name for a in ordered],
        "arms": {
            arm.name: {
                "label": arm.label,
                "metrics": (results.get(arm.name) or {}).get("metrics"),
                "entry_count": (results.get(arm.name) or {}).get("entry_count"),
                "collection": (results.get(arm.name) or {}).get("collection"),
                "error": (results.get(arm.name) or {}).get("error"),
            }
            for arm in arms
        },
        "comparison": comparison,
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
    }
    (run_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    (run_dir / "report.md").write_text(render_report_md(report), encoding="utf-8")
    _write_per_query_csv(run_dir, experiment, results)
    print(render_report_text(report))
    return report


# ---------------------------------------------------------------------------
# 报告渲染
# ---------------------------------------------------------------------------

_METRIC_LABELS = {
    "faithfulness": "忠实度 faithfulness",
    "answer_relevancy": "回答相关性 answer_relevancy",
    "context_precision": "上下文精确率 context_precision",
    "context_recall": "上下文召回率 context_recall",
    "answer_correctness": "答案正确性 answer_correctness",
    "context_entity_recall": "实体召回 context_entity_recall",
    "hit_rate": "命中率 hit_rate@5",
    "mrr": "MRR@5",
    "ndcg": "NDCG@5",
    "composite_score": "综合分 composite",
}


def _fmt(value: Any) -> str:
    return "N/A" if value is None else f"{float(value):.4f}"


def _table_rows(report: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    return sorted(report["comparison"]["metrics"].items())


def render_report_text(report: dict[str, Any]) -> str:
    arms = report["experiment"]
    lines = [
        "=" * 78,
        f"  A/B 实验：{report['experiment']['label']}（{report['experiment']['name']}）",
        f"  臂 A [{arms['arm_a']['name']}]：{arms['arm_a']['label']}",
        f"  臂 B [{arms['arm_b']['name']}]：{arms['arm_b']['label']}",
        f"  测试集：{Path(report['testset']).name} | 样本 {report['sample'] or '全量'} | "
        f"模式：{'fast' if report['fast'] else 'full'} | seed={report['seed']}",
        "=" * 78,
        f"{'指标':<40} {'臂A':>10} {'臂B':>10} {'Δ(B-A)':>10} {'相对%':>10}",
        "-" * 78,
    ]
    for name, row in _table_rows(report):
        label = _METRIC_LABELS.get(name, name)
        rel = "N/A" if row["relative_pct"] is None else f"{row['relative_pct']:.2f}%"
        lines.append(
            f"{label:<40} {_fmt(row['arm_a']):>10} {_fmt(row['arm_b']):>10} "
            f"{_fmt(row['delta']):>10} {rel:>10}  ({row['winner'] or '-'})"
        )
    errors = []
    if report["comparison"].get("arm_a_error"):
        errors.append(f"臂A 错误：{report['comparison']['arm_a_error']}")
    if report["comparison"].get("arm_b_error"):
        errors.append(f"臂B 错误：{report['comparison']['arm_b_error']}")
    if errors:
        lines.append("-" * 78)
        lines.extend(errors)
    return "\n".join(lines) + "\n"


def render_report_md(report: dict[str, Any]) -> str:
    arms = report["experiment"]
    lines = [
        f"# A/B 实验报告：{report['experiment']['label']}",
        "",
        f"- 实验：`{report['experiment']['name']}`",
        f"- 臂 A：**{arms['arm_a']['name']}**（{arms['arm_a']['label']}）",
        f"- 臂 B：**{arms['arm_b']['name']}**（{arms['arm_b']['label']}）",
        f"- 测试集：`{Path(report['testset']).name}`；样本数：{report['sample'] or '全量'}；"
        f"模式：{'fast（3 项）' if report['fast'] else 'full（6 项）'}；seed：{report['seed']}",
        f"- 臂执行顺序：{' → '.join(report['arm_order'])}",
        "",
        "| 指标 | 臂 A | 臂 B | Δ (B-A) | 相对变化 | 胜出 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for name, row in _table_rows(report):
        label = _METRIC_LABELS.get(name, name)
        rel = "N/A" if row["relative_pct"] is None else f"{row['relative_pct']:.2f}%"
        lines.append(
            f"| {label} | {_fmt(row['arm_a'])} | {_fmt(row['arm_b'])} | "
            f"{_fmt(row['delta'])} | {rel} | {row['winner'] or '—'} |"
        )
    return "\n".join(lines) + "\n"


def _write_per_query_csv(
    run_dir: Path, experiment: Any, results: dict[str, Any]
) -> None:
    """合并双臂逐题明细为 CSV。"""
    import csv

    columns = [
        "arm", "query", "question_type", "chunk_ids", "related_docs",
        "difficulty", "test_type", "response", "errors",
        "faithfulness", "answer_relevancy", "context_precision",
        "context_recall", "answer_correctness", "context_entity_recall",
        "hit_rate", "mrr", "ndcg", "multi_hop_hit", "negative_rejection",
        "composite_score",
        "retrieval_latency_ms", "generation_latency_ms",
    ]
    with (run_dir / "per_query.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for arm in (experiment.arm_a, experiment.arm_b):
            result = results.get(arm.name) or {}
            for item in result.get("per_query", []):
                row = {
                    "arm": arm.name,
                    "query": item.get("user_input", ""),
                    "question_type": item.get("question_type", ""),
                    "chunk_ids": ";".join(item.get("chunk_ids", [])),
                    "related_docs": item.get("related_docs", ""),
                    "difficulty": item.get("difficulty", ""),
                    "test_type": item.get("test_type", ""),
                    "response": item.get("response", ""),
                    "errors": json.dumps(item.get("errors", []), ensure_ascii=False),
                    "retrieval_latency_ms": item.get("retrieval_latency_ms"),
                    "generation_latency_ms": item.get("generation_latency_ms"),
                }
                row.update(item.get("metrics", {}))
                writer.writerow(row)


# ---------------------------------------------------------------------------
# 子进程入口
# ---------------------------------------------------------------------------

def _parse_arm_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A/B 实验臂执行器（子进程入口）")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--testset", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-missing", action="store_true")
    return parser.parse_args(argv)


def arm_main(argv: list[str] | None = None) -> int:
    """子进程入口：执行单臂并写结果 JSON。"""
    from finance_rag.src.eval.ab_catalog import select_experiments

    args = _parse_arm_args(argv)
    experiment = select_experiments([args.experiment])[0]
    arm = next(
        a for a in (experiment.arm_a, experiment.arm_b) if a.name == args.arm
    )
    result = execute_arm(
        experiment,
        arm,
        testset_path=Path(args.testset),
        fast=args.fast,
        sample=args.sample,
        seed=args.seed,
        allow_missing=args.allow_missing,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    logger.info("臂 %s 结果已写入 %s", arm.name, out)
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    sys.exit(arm_main())
