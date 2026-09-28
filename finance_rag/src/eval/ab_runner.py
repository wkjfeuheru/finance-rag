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

# 拒答关键词与判定统一在 ``runner`` 里维护，避免两处副本各自漏词。
# 历史上 ``ab_runner`` 自带一份，与 ``runner`` 的同名实现都漏掉了「未提供」
# 这类常见拒答措辞，会把正确拒答判成幻觉作答（18 篇语料基线上真实发生过）。
from finance_rag.src.eval.runner import (  # noqa: E402
    negative_rejection_score as _negative_rejection_score,
)


def _stratified_sample(entries: list[Any], sample: int, seed: int) -> list[Any]:
    """按 ``question_type`` 分层抽样，保证小样本下每类问题都有代表。

    原先直接用 ``random.sample`` 全局抽样：``--sample 10`` 时 multi_hop 只有 2 题，
    很容易抽到 0 题或抽到同一个问题的重复条目，导致该类型指标（``multi_hop_hit``）
    失去意义——实际报告里 n=2 就是同一题在两臂各出现一次。

    策略：先按类型均分名额，再在类内随机；名额不足时把余量补给题量最多的类型。
    类型顺序与类内顺序都受 ``seed`` 控制，保证可复现。
    """
    rng = random.Random(seed)
    buckets: dict[str, list[Any]] = {}
    for entry in entries:
        buckets.setdefault(getattr(entry, "question_type", "") or "single_hop", []).append(entry)

    types = sorted(buckets)
    quota = {qtype: min(sample // len(types), len(buckets[qtype])) for qtype in types}
    remaining = sample - sum(quota.values())
    # 余量按「类内剩余题量」从多到少补齐，同类内保持稳定顺序
    while remaining > 0:
        candidates = [
            qtype for qtype in types if quota[qtype] < len(buckets[qtype])
        ]
        if not candidates:
            break
        candidates.sort(key=lambda qtype: (-len(buckets[qtype]), qtype))
        quota[candidates[0]] += 1
        remaining -= 1

    picked: list[Any] = []
    for qtype in types:
        pool = list(buckets[qtype])
        rng.shuffle(pool)
        picked.extend(pool[: quota[qtype]])
    # 输出顺序按类型稳定排列，便于逐题明细比对
    picked.sort(key=lambda entry: getattr(entry, "question_type", "") or "")
    return picked


def _flag_weak_multi_hop(diagnostics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """筛出「标注的相关文档一篇都没被召回」的多跳条目。

    这类条目的 ``multi_hop_hit=0`` 说明不了检索能力：自动生成的跨文档多跳题
    是把两篇随机抽到的文档硬配在一起的（见 ``testset_generator._generate_multi_hop``），
    第二篇常常与问题无关，检不到是正常的。把这类条目单独列出来，才能把
    「检索/重排序失败」与「评测集标注不可信」分开归因。

    仅报警、不自动剔除，避免悄悄改写评测基准。
    """
    weak: list[dict[str, Any]] = []
    for item in diagnostics:
        expected = {
            Path(name).stem.lower()
            for name in re.split(r"[,，;；|]", item.get("expected_docs") or "")
            if name.strip()
        }
        retrieved = {
            Path(stem).stem.lower()
            for stem in (item.get("retrieved_docs") or "").split(";")
            if stem.strip()
        }
        if item.get("doc_coverage") == 0.0:
            missing = sorted(name for name in (expected - retrieved) if name)
            weak.append({
                "query": item.get("query", ""),
                "expected_docs": item.get("expected_docs", ""),
                "missing_docs": ", ".join(missing),
                "retrieved_docs": item.get("retrieved_docs", ""),
                "note": "标注文档一篇都未召回：疑似该文档与问题无关（自动配对产物）",
            })
    return weak


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
    include_silver: bool = False,
) -> dict[str, Any]:
    """执行单臂：逐题检索/生成 → Ragas 批量评估 → 聚合。"""
    from finance_rag.src.eval.ragas_eval import (
        RagasBatchEvaluator,
        _compute_doc_coverage,
        _compute_evidence_metrics,
        _compute_local_retrieval_metrics,
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
    if testset_path.suffix.lower() == ".jsonl" and not include_silver:
        entries = [entry for entry in entries if entry.tier == "gold"]
    entries, removed = filter_entries_by_kb(entries, kb, allow_missing=allow_missing)
    if sample and 0 < sample < len(entries):
        entries = _stratified_sample(entries, sample, seed)
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
                # id / child_id 供评测侧做证据块对齐；父子扩展后 id 仍是子块主键
                "id": d.get("id", "") or d.get("child_id", ""),
                "chunk_key": d.get("chunk_key", ""),
                "parent_id": d.get("parent_id", ""),
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
            "case_id": entry.unique_id or str(entry.question_id),
            "tier": entry.tier,
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
    positive_rows = [
        row for row in ragas_rows if row.get("question_type") != "negative"
    ]
    evaluation = (
        evaluator.evaluate_batch(positive_rows, fast=fast)
        if positive_rows
        else {"per_query": [], "metric_errors": {}, "fallback_metrics": []}
    )
    ragas_ms = round((time.perf_counter() - t0) * 1000, 4)

    # 逐题：本地检索指标 + 证据块指标 + 综合分 + 分类型本地指标
    evaluated = iter(evaluation["per_query"])
    per_query = [
        ({**row, "metrics": {}} if row.get("question_type") == "negative" else next(evaluated))
        for row in ragas_rows
    ]
    multi_hop_diag: list[dict[str, Any]] = []
    for i, entry in enumerate(entries):
        if i >= len(per_query):
            break
        retrieved_sources = [s["source"] for s in per_query[i]["sources"]]
        expected_docs = entry.related_doc_names or entry.related_docs
        local = _compute_local_retrieval_metrics(
            retrieved_sources, expected_docs, k=5
        )
        local.update(_compute_doc_coverage(retrieved_sources, expected_docs, k=5))

        # 证据块级指标：检索结果里的 chunk id（父子扩展后仍保留子块身份）
        retrieved_chunk_ids = [
            str(s.get("id") or s.get("child_id") or "")
            for s in per_query[i]["sources"]
        ]
        evidence = _compute_evidence_metrics(
            retrieved_chunk_ids, entry.chunk_ids, k=5
        )
        local.update({k: v for k, v in evidence.items() if k != "evidence_rank"})
        per_query[i]["metrics"].update(local)
        # evidence_rank 未命中记 k+1，单独放在诊断里，避免拉高均值造成误读
        per_query[i]["diagnostics"] = {
            "evidence_rank": evidence["evidence_rank"],
            "retrieved_doc_stems": ";".join(
                Path(source).stem for source in retrieved_sources[:5] if source
            ),
            "retrieved_chunk_ids": ";".join(cid for cid in retrieved_chunk_ids[:5] if cid),
        }

        # 多跳：相关文档全部被召回才算命中（跨文档证据覆盖）
        if entry.question_type == "multi_hop":
            per_query[i]["metrics"]["multi_hop_hit"] = float(
                local["doc_recall"] if local["doc_recall"] is not None else 0.0
            )
            multi_hop_diag.append({
                "query": entry.query,
                "expected_docs": entry.related_docs,
                "retrieved_docs": per_query[i]["diagnostics"]["retrieved_doc_stems"],
                "doc_coverage": local["doc_coverage"],
                "evidence_coverage": local["evidence_coverage"],
                "evidence_rank": evidence["evidence_rank"],
            })
        # 负样本：是否正确拒绝（回答含拒绝表述或为空）
        if entry.question_type == "negative":
            per_query[i]["metrics"]["negative_rejection"] = _negative_rejection_score(
                per_query[i].get("response", "")
            )

    # 弱标注检测：多跳条目标注的相关文档从未被召回 → 疑似评测集标注不可信
    suspect_annotations = _flag_weak_multi_hop(multi_hop_diag)

    # 聚合：各指标均值（仅有限值）
    metric_names: set[str] = set()
    for item in per_query:
        metric_names.update(item["metrics"].keys())
    all_aggregates: dict[str, float | None] = {}
    for name in sorted(metric_names):
        finite = [
            item["metrics"][name]
            for item in per_query
            if isinstance(item["metrics"].get(name), (int, float))
        ]
        all_aggregates[name] = round(sum(finite) / len(finite), 4) if finite else None
    primary_names = {
        "answer_correctness", "faithfulness", "answer_relevancy", "negative_rejection",
    }
    aggregates = {name: all_aggregates.get(name) for name in sorted(primary_names) if name in all_aggregates}
    diagnostic_aggregates = {
        name: value for name, value in all_aggregates.items() if name not in primary_names
    }

    # 按问题类型分层统计（single_hop / multi_hop / negative）
    # None（n/a，例如测试集未标注证据块）不计入均值，但保留样本数，
    # 这样「某类指标全是 n/a」在报告里一眼可见。
    metrics_by_type: dict[str, dict[str, float | None]] = {}
    counts_by_type: dict[str, dict[str, int]] = {}
    for item in per_query:
        qtype = item.get("question_type", "single_hop")
        bucket = metrics_by_type.setdefault(qtype, {})
        count_bucket = counts_by_type.setdefault(qtype, {})
        for name, value in item.get("metrics", {}).items():
            if isinstance(value, (int, float)):
                bucket.setdefault(name, []).append(value)
                count_bucket[name] = count_bucket.get(name, 0) + 1
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
        "diagnostic_metrics": diagnostic_aggregates,
        "metrics_by_type": metrics_by_type,
        "metrics_by_type_counts": counts_by_type,
        "suspect_annotations": suspect_annotations,
        "multi_hop_diagnostics": multi_hop_diag,
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
    include_silver: bool = False,
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
        if include_silver:
            cmd.append("--include-silver")
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
                "metrics_by_type": (results.get(arm.name) or {}).get("metrics_by_type"),
                "metrics_by_type_counts": (
                    results.get(arm.name) or {}
                ).get("metrics_by_type_counts"),
                # 疑似标注不可信的条目：把「检索失败」与「评测集缺陷」分开归因
                "suspect_annotations": (
                    results.get(arm.name) or {}
                ).get("suspect_annotations"),
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
    "doc_coverage": "文档覆盖 doc_coverage",
    "doc_recall": "文档全召回 doc_recall",
    "evidence_hit": "证据块命中 evidence_hit",
    "evidence_coverage": "证据块覆盖 evidence_coverage",
    "evidence_all_in_topk": "证据块全在 top5",
    "multi_hop_hit": "多跳全召回 multi_hop_hit",
    "negative_rejection": "负样本拒绝 negative_rejection",
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
    lines.extend(_render_type_breakdown(report))
    lines.extend(_render_suspect_annotations(report))
    return "\n".join(lines) + "\n"


def _render_type_breakdown(report: dict[str, Any]) -> list[str]:
    """按问题类型分层表：带上样本数，n/a 指标一眼可见。

    ``multi_hop_hit`` 这类指标在单跳/负样本条目上不存在，且小样本下
    某类型可能只有 1～2 题，不写清 n 很容易把噪声当成结论。
    """
    arms = report.get("arms") or {}
    arm_a_name = report["experiment"]["arm_a"]["name"]
    arm_b_name = report["experiment"]["arm_b"]["name"]
    arm_a = arms.get(arm_a_name) or {}
    arm_b = arms.get(arm_b_name) or {}
    by_type_a = arm_a.get("metrics_by_type") or {}
    by_type_b = arm_b.get("metrics_by_type") or {}
    counts_a = arm_a.get("metrics_by_type_counts") or {}
    counts_b = arm_b.get("metrics_by_type_counts") or {}
    if not by_type_a and not by_type_b:
        return []

    lines = ["", "### 分问题类型指标", "", "| 类型 | 指标 | 臂 A | 臂 B |", "|---|---|---:|---:|"]
    for qtype in sorted(set(by_type_a) | set(by_type_b)):
        type_a = by_type_a.get(qtype) or {}
        type_b = by_type_b.get(qtype) or {}
        for metric in sorted(set(type_a) | set(type_b)):
            label = _METRIC_LABELS.get(metric, metric)
            # 两个臂的样本数应一致；取任一非空值展示
            n = (counts_a.get(qtype) or {}).get(metric) or (
                counts_b.get(qtype) or {}
            ).get(metric)
            suffix = f"（n={n}）" if n else ""
            lines.append(
                f"| {qtype} | {label}{suffix} | {_fmt(type_a.get(metric))} | "
                f"{_fmt(type_b.get(metric))} |"
            )
    return lines


def _render_suspect_annotations(report: dict[str, Any]) -> list[str]:
    """疑似评测集标注不可信的条目（多跳标注文档一篇都没召回）。"""
    suspects: dict[str, list[dict[str, Any]]] = {}
    for arm, result in (report.get("arms") or {}).items():
        flagged = (result or {}).get("suspect_annotations") or []
        if flagged:
            suspects[arm] = flagged
    if not suspects:
        return []
    lines = [
        "",
        "### [注意] 疑似标注不可信的条目",
        "",
        "以下多跳条目标注的相关文档**一篇都没有被召回**。自动生成的跨文档多跳题是",
        "把随机抽到的两篇文档硬配成对（见 `testset_generator._generate_multi_hop`），",
        "第二篇常常与问题无关，因此这类 `multi_hop_hit=0` **不能归因为检索失败**。",
        "",
    ]
    for arm, flagged in sorted(suspects.items()):
        lines.append(f"- 臂 `{arm}`：{len(flagged)} 条")
        for item in flagged[:10]:
            lines.append(f"  - 期望文档：{item.get('expected_docs', '')}")
            lines.append(f"    - 未召回：{item.get('missing_docs', '')}")
            lines.append(f"    - 实际召回：{item.get('retrieved_docs', '') or '（空）'}")
    return lines


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
        # 文档级：hit_rate=至少命中一篇；doc_coverage/doc_recall=覆盖程度（多跳关键）
        "hit_rate", "mrr", "ndcg", "doc_coverage", "doc_recall",
        # 块级证据指标（测试集未标注证据块时为空 = n/a）
        "evidence_hit", "evidence_coverage", "evidence_all_in_topk",
        # 证据块排名（未命中记 k+1=6），便于区分「未召回」与「被 rerank 截断」
        "evidence_rank",
        "multi_hop_hit", "negative_rejection",
        "composite_score",
        # 实测检索结果（对齐排查用）
        "retrieved_doc_stems", "retrieved_chunk_ids",
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
                # 诊断字段单独取，避免污染 metrics 聚合
                diagnostics = item.get("diagnostics", {}) or {}
                row["evidence_rank"] = diagnostics.get("evidence_rank")
                row["retrieved_doc_stems"] = diagnostics.get("retrieved_doc_stems", "")
                row["retrieved_chunk_ids"] = diagnostics.get("retrieved_chunk_ids", "")
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
    parser.add_argument("--include-silver", action="store_true")
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
        include_silver=args.include_silver,
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
