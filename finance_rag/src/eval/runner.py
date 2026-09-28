"""统一评测运行器：逐题执行、断点恢复并生成标准产物。"""

from __future__ import annotations

import csv
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Literal

from .dataset import EvalCase, EvaluationDataset
from .metrics import compute_chunk_retrieval_metrics

Mode = Literal["retrieval", "generation"]


@dataclass(frozen=True)
class RetrievalOutput:
    """一次检索的完整输出；候选集用于诊断，results 用于正式计分。"""

    candidates: list[dict[str, Any]]
    results: list[dict[str, Any]]
    cliff_triggered: bool = False
    pre_cliff_depth: int | None = None
    post_cliff_depth: int | None = None


@dataclass(frozen=True)
class EvaluationStrategy:
    """把具体检索/生成实现注入统一运行器。"""

    name: str
    retrieve: Callable[[str], RetrievalOutput]
    generate: Callable[[EvalCase, list[str]], str] | None = None
    evaluate_answer: Callable[[EvalCase, list[str], str], dict[str, float | None]] | None = None
    config: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationRun:
    records: list[dict[str, Any]]
    summary: dict[str, Any]
    output_dir: Path


def _chunk_id(item: dict[str, Any]) -> str:
    return str(item.get("id") or item.get("chunk_id") or item.get("pk") or "")


def _content(item: dict[str, Any]) -> str:
    return str(item.get("content") or item.get("text") or "")


#: 拒答识别关键词。真实的"拒答"远不止「无法回答」这一种说法，实测中模型
#: 更常说「检索到的内容**未提供**…」「**未提及**…」——漏词会把正确拒答判成
#: 幻觉作答，直接把负样本拒答正确率打对折（18 篇语料基线上真实发生过：
#: 「检索到的内容未提供宁德时代…具体数据」被误判为未拒答）。
REJECTION_KEYWORDS = (
    "无法回答", "无法给出可靠回答", "无法给出", "拒绝回答", "不能回答",
    "未检索到", "没有检索到", "暂未检索", "未能检索", "未找到",
    "暂无相关", "无相关信息", "无相关数据", "没有相关",
    "未提供", "未提及", "未包含", "未涉及", "不含",
    "知识库中未", "现有知识库中", "语料中没有", "不足以支撑", "无法支撑",
)


def negative_rejection_score(answer: str) -> float:
    """负样本是否正确拒绝（含拒绝表述或空回答）返回 1.0，否则 0.0。

    这是**关键词代理指标**，不是语义判定：答案里出现"…未提供…"即视为拒答，
    因此必须同时保留原始答案供人工复核（见 scripts/baseline_e2e.py 的 JSONL）。
    """
    if not answer or not answer.strip():
        return 1.0
    return 1.0 if any(kw in answer for kw in REJECTION_KEYWORDS) else 0.0


def _negative_rejection_score(answer: str) -> float:
    """向后兼容的私有别名。"""
    return negative_rejection_score(answer)


class EvaluationRunner:
    """对同一数据集运行检索或生成评测，并在每题后落盘。"""

    def run(
        self,
        dataset: EvaluationDataset,
        strategy: EvaluationStrategy,
        mode: Mode,
        checkpoint_dir: str | Path,
        *,
        include_silver: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> EvaluationRun:
        if mode not in {"retrieval", "generation"}:
            raise ValueError(f"不支持的评测模式：{mode}")
        if mode == "generation" and (strategy.generate is None or strategy.evaluate_answer is None):
            raise ValueError("生成评测必须提供 generate 和 evaluate_answer")

        dataset.validate()
        output_dir = Path(checkpoint_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        records_path = output_dir / "records.jsonl"
        selected = self._select_cases(dataset, mode, include_silver)
        self._validate_checkpoint(output_dir, dataset, strategy, mode, selected, metadata or {})

        existing = self._read_records(records_path)
        completed = {record["case_id"] for record in existing if record.get("status") == "completed"}
        for case in selected:
            if case.id in completed:
                continue
            record = self._run_case(case, strategy, mode)
            with records_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

        by_id = {record["case_id"]: record for record in self._read_records(records_path)}
        records = [by_id[case.id] for case in selected if case.id in by_id]
        summary = self._summarize(records, dataset, strategy, mode, include_silver, metadata or {})
        self._write_outputs(output_dir, records, summary)
        return EvaluationRun(records, summary, output_dir)

    @staticmethod
    def _select_cases(dataset: EvaluationDataset, mode: Mode, include_silver: bool) -> list[EvalCase]:
        if mode == "retrieval":
            return [case for case in dataset.cases if case.question_type != "negative"]
        tiers = {"gold", "silver"} if include_silver else {"gold"}
        return [case for case in dataset.cases if case.tier in tiers]

    @staticmethod
    def _dataset_fingerprint(cases: list[EvalCase]) -> str:
        payload = "\n".join(
            json.dumps(case.to_dict(), ensure_ascii=False, sort_keys=True) for case in cases
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _validate_checkpoint(
        self,
        output_dir: Path,
        dataset: EvaluationDataset,
        strategy: EvaluationStrategy,
        mode: Mode,
        selected: list[EvalCase],
        metadata: dict[str, Any],
    ) -> None:
        config_path = output_dir / "run_config.json"
        value = {
            "mode": mode,
            "strategy": strategy.name,
            "strategy_config": strategy.config,
            "dataset_fingerprint": self._dataset_fingerprint(selected),
            "metadata": metadata,
        }
        if config_path.exists():
            previous = json.loads(config_path.read_text(encoding="utf-8"))
            if previous != value:
                raise ValueError("检查点目录已属于不同的数据集、策略或运行配置")
            return
        config_path.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
        )

    @staticmethod
    def _read_records(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        rows: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
        return rows

    def _run_case(
        self, case: EvalCase, strategy: EvaluationStrategy, mode: Mode
    ) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            retrieval = strategy.retrieve(case.query)
            result_ids = [_chunk_id(item) for item in retrieval.results if _chunk_id(item)]
            diagnostics: dict[str, Any] = {
                "candidate_chunk_ids": [
                    _chunk_id(item) for item in retrieval.candidates if _chunk_id(item)
                ],
                "top5_chunk_ids": result_ids[:5],
                "evidence_ranks": self._evidence_ranks(case, result_ids),
                "cliff_triggered": retrieval.cliff_triggered,
                "pre_cliff_depth": retrieval.pre_cliff_depth,
                "post_cliff_depth": retrieval.post_cliff_depth,
            }
            answer: str | None = None
            if mode == "retrieval":
                raw = compute_chunk_retrieval_metrics(
                    result_ids, [item.chunk_id for item in case.evidence], k=5
                )
                metrics = {
                    "recall_at_5": raw["recall_at_k"],
                    "precision_at_5": raw["precision_at_k"],
                    "mrr": raw["mrr"],
                }
            else:
                contexts = [_content(item) for item in retrieval.results if _content(item)]
                assert strategy.generate is not None
                answer = strategy.generate(case, contexts)
                if case.question_type == "negative":
                    metrics = {"negative_rejection": _negative_rejection_score(answer)}
                else:
                    assert strategy.evaluate_answer is not None
                    evaluated = strategy.evaluate_answer(case, contexts, answer)
                    metrics = {
                        key: evaluated.get(key)
                        for key in ("answer_correctness", "faithfulness", "answer_relevancy")
                    }
                    diagnostics["generation_metrics"] = {
                        key: value for key, value in evaluated.items() if key not in metrics
                    }
            return {
                "case_id": case.id,
                "tier": case.tier,
                "question_type": case.question_type,
                "query": case.query,
                "status": "completed",
                "metrics": metrics,
                "diagnostics": diagnostics,
                "answer": answer,
                "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                "error": None,
            }
        except Exception as exc:  # 单题失败也必须写检查点，避免整批静默丢失
            return {
                "case_id": case.id,
                "tier": case.tier,
                "question_type": case.question_type,
                "query": case.query,
                "status": "failed",
                "metrics": {},
                "diagnostics": {},
                "answer": None,
                "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                "error": f"{type(exc).__name__}: {exc}",
            }

    @staticmethod
    def _evidence_ranks(case: EvalCase, result_ids: list[str]) -> dict[str, int | None]:
        ranks = {chunk_id: rank for rank, chunk_id in enumerate(result_ids, 1)}
        return {evidence.chunk_id: ranks.get(evidence.chunk_id) for evidence in case.evidence}

    @staticmethod
    def _summarize(
        records: list[dict[str, Any]],
        dataset: EvaluationDataset,
        strategy: EvaluationStrategy,
        mode: Mode,
        include_silver: bool,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        metric_names = sorted({key for row in records for key in row.get("metrics", {})})
        metrics: dict[str, float | None] = {}
        coverage: dict[str, float] = {}
        for name in metric_names:
            values = [
                row["metrics"][name]
                for row in records
                if row.get("metrics", {}).get(name) is not None
            ]
            metrics[name] = round(mean(values), 6) if values else None
            coverage[name] = round(len(values) / len(records), 6) if records else 0.0
        tiers: dict[str, Any] = {}
        for tier in ("gold", "silver"):
            subset = [row for row in records if row["tier"] == tier]
            tiers[tier] = {
                "n": len(subset),
                "metrics": {
                    name: round(mean(values), 6) if values else None
                    for name in metric_names
                    if (values := [
                        row["metrics"][name]
                        for row in subset
                        if row.get("metrics", {}).get(name) is not None
                    ])
                },
            }
        return {
            "mode": mode,
            "strategy": strategy.name,
            "strategy_config": strategy.config,
            "dataset_source": str(dataset.source_path) if dataset.source_path else None,
            "include_silver": include_silver,
            "n": len(records),
            "completed": sum(row["status"] == "completed" for row in records),
            "failed": sum(row["status"] == "failed" for row in records),
            "metrics": metrics,
            "metric_coverage": coverage,
            "tiers": tiers,
            "metadata": metadata,
        }

    def _write_outputs(
        self, output_dir: Path, records: list[dict[str, Any]], summary: dict[str, Any]
    ) -> None:
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
        )
        fields = [
            "case_id", "tier", "question_type", "status", "query",
            "recall_at_5", "precision_at_5", "mrr", "answer_correctness",
            "faithfulness", "answer_relevancy", "negative_rejection",
            "latency_ms", "error",
        ]
        with (output_dir / "per_query.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for record in records:
                writer.writerow({
                    **{key: record.get(key) for key in fields},
                    **{key: record.get("metrics", {}).get(key) for key in fields},
                })
        lines = [
            f"# 评测报告：{summary['strategy']}", "",
            f"- 模式：{summary['mode']}",
            f"- 样本数：{summary['n']}",
            f"- 完成/失败：{summary['completed']}/{summary['failed']}", "",
            "## 主指标", "",
            "| 指标 | 均值 | 覆盖率 |", "|---|---:|---:|",
        ]
        for name, value in summary["metrics"].items():
            shown = "N/A" if value is None else f"{value:.4f}"
            lines.append(f"| {name} | {shown} | {summary['metric_coverage'][name]:.1%} |")
        (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
