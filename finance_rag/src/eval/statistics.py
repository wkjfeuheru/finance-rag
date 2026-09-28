"""配对 Bootstrap、指标覆盖率与实验护栏判定。"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    primary_metric: str
    guardrails: tuple[str, ...]


EXPERIMENT_SPECS: dict[str, ExperimentSpec] = {
    "hybrid_vs_dense": ExperimentSpec(
        "hybrid_vs_dense", "recall_at_5", ("precision_at_5", "mrr")
    ),
    "rerank_on_off": ExperimentSpec(
        "rerank_on_off", "mrr", ("recall_at_5", "precision_at_5")
    ),
    "cliff_on_off": ExperimentSpec(
        "cliff_on_off", "precision_at_5", ("recall_at_5", "mrr")
    ),
}


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("values must not be empty")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _bootstrap_interval(
    differences: list[float],
    *,
    samples: int,
    rng: random.Random,
) -> tuple[float, float]:
    if not differences:
        raise ValueError("differences must not be empty")
    if samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    size = len(differences)
    means = [
        sum(differences[rng.randrange(size)] for _ in range(size)) / size
        for _ in range(samples)
    ]
    return _percentile(means, 0.025), _percentile(means, 0.975)


def _metric_verdict(ci_low: float, ci_high: float) -> str:
    if ci_low > 0:
        return "improvement"
    if ci_high < 0:
        return "regression"
    return "inconclusive"


def _index(records: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        case_id = str(record.get("case_id", ""))
        if not case_id:
            raise ValueError("run record missing case_id")
        if case_id in indexed:
            raise ValueError(f"duplicate case_id: {case_id}")
        indexed[case_id] = record
    return indexed


def compare_runs(
    baseline: Iterable[dict[str, Any]],
    candidate: Iterable[dict[str, Any]],
    experiment_spec: ExperimentSpec,
    *,
    bootstrap_samples: int = 10_000,
    seed: int = 7,
    minimum_gold_coverage: float = 0.95,
) -> dict[str, Any]:
    """按 case_id 配对比较两次运行，Gold 与 Silver 独立统计。"""
    baseline_by_id = _index(baseline)
    candidate_by_id = _index(candidate)
    all_ids = sorted(set(baseline_by_id) | set(candidate_by_id))
    tiers: dict[str, Any] = {}

    for tier in ("gold", "silver"):
        tier_ids = [
            case_id for case_id in all_ids
            if (baseline_by_id.get(case_id) or candidate_by_id.get(case_id) or {}).get("tier") == tier
            and baseline_by_id.get(case_id, {}).get("tier", tier) == tier
            and candidate_by_id.get(case_id, {}).get("tier", tier) == tier
        ]
        metric_names: set[str] = set()
        for case_id in tier_ids:
            metric_names.update((baseline_by_id.get(case_id, {}).get("metrics") or {}).keys())
            metric_names.update((candidate_by_id.get(case_id, {}).get("metrics") or {}).keys())

        metric_results: dict[str, Any] = {}
        for metric_name in sorted(metric_names):
            pairs: list[tuple[str, float, float]] = []
            for case_id in tier_ids:
                before = (baseline_by_id.get(case_id, {}).get("metrics") or {}).get(metric_name)
                after = (candidate_by_id.get(case_id, {}).get("metrics") or {}).get(metric_name)
                if _finite(before) and _finite(after):
                    pairs.append((case_id, float(before), float(after)))
            coverage = len(pairs) / len(tier_ids) if tier_ids else 0.0
            if not pairs:
                metric_results[metric_name] = {
                    "n": 0,
                    "coverage": round(coverage, 4),
                    "baseline": None,
                    "candidate": None,
                    "delta": None,
                    "ci95": None,
                    "verdict": "no_data",
                    "wins": 0,
                    "losses": 0,
                    "ties": 0,
                    "largest_changes": [],
                }
                continue
            differences = [after - before for _, before, after in pairs]
            metric_rng = random.Random(f"{seed}:{tier}:{metric_name}")
            ci_low, ci_high = _bootstrap_interval(
                differences,
                samples=bootstrap_samples,
                rng=metric_rng,
            )
            changes = sorted(
                (
                    {"case_id": case_id, "delta": round(after - before, 4)}
                    for case_id, before, after in pairs
                ),
                key=lambda item: (-abs(item["delta"]), item["case_id"]),
            )[:5]
            metric_results[metric_name] = {
                "n": len(pairs),
                "coverage": round(coverage, 4),
                "baseline": round(sum(before for _, before, _ in pairs) / len(pairs), 4),
                "candidate": round(sum(after for _, _, after in pairs) / len(pairs), 4),
                "delta": round(sum(differences) / len(differences), 4),
                "ci95": [round(ci_low, 4), round(ci_high, 4)],
                "verdict": _metric_verdict(ci_low, ci_high),
                "wins": sum(diff > 0 for diff in differences),
                "losses": sum(diff < 0 for diff in differences),
                "ties": sum(diff == 0 for diff in differences),
                "largest_changes": changes,
            }
        tiers[tier] = {"n": len(tier_ids), "metrics": metric_results}

    gold_metrics = tiers["gold"]["metrics"]
    primary = gold_metrics.get(experiment_spec.primary_metric)
    required_metrics = (experiment_spec.primary_metric, *experiment_spec.guardrails)
    insufficient_coverage = [
        metric_name for metric_name in required_metrics
        if (gold_metrics.get(metric_name) or {}).get("coverage", 0.0) < minimum_gold_coverage
    ]
    guardrail_failures = [
        metric_name
        for metric_name in experiment_spec.guardrails
        if (gold_metrics.get(metric_name) or {}).get("verdict") == "regression"
    ]
    if (
        not primary
        or insufficient_coverage
        or primary.get("verdict") == "no_data"
    ):
        decision = "no_decision"
    elif primary["verdict"] == "regression" or guardrail_failures:
        decision = "fail"
    elif primary["verdict"] == "improvement":
        decision = "pass"
    else:
        decision = "inconclusive"

    return {
        "experiment": experiment_spec.name,
        "primary_metric": experiment_spec.primary_metric,
        "guardrails": list(experiment_spec.guardrails),
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "minimum_gold_coverage": minimum_gold_coverage,
        "tiers": tiers,
        "guardrail_failures": guardrail_failures,
        "insufficient_coverage": insufficient_coverage,
        "decision": decision,
    }
