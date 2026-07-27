from __future__ import annotations

import csv
import json
import math
import os
import platform
import random
import re
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Protocol, Sequence

QUALITY_METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
STRATEGY_NAMES = ("baseline", "optimized")
DEFAULT_SEED = 20260724


@dataclass(frozen=True)
class BenchmarkStrategy:
    name: str
    dense_weight: float
    sparse_weight: float
    use_rerank: bool
    rerank_top_n: int = 3
    k: int = 5

    def to_strategy_config(self):
        from .evaluation import StrategyConfig
        return StrategyConfig(
            dense_weight=self.dense_weight,
            sparse_weight=self.sparse_weight,
            use_rerank=self.use_rerank,
            rerank_top_n=self.rerank_top_n,
            k=self.k,
        )


BASELINE = BenchmarkStrategy("baseline", 1.0, 0.0, False)
OPTIMIZED = BenchmarkStrategy("optimized", 0.7, 0.3, True)


class BenchmarkBackend(Protocol):
    def retrieve(self, strategy: BenchmarkStrategy, query: str) -> list[dict[str, Any]]: ...
    def generate(self, query: str, contexts: list[str]) -> str: ...
    def evaluate_quality(self, strategy: BenchmarkStrategy, entry: Any, contexts: list[str],
                         sources: list[dict[str, Any]], response: str) -> dict[str, Any]: ...


class EvaluationBackend:
    """Adapter over the production evaluator without changing the API path."""

    def __init__(self):
        from .evaluation import get_strategy_evaluator
        self.evaluator = get_strategy_evaluator()

    def retrieve(self, strategy: BenchmarkStrategy, query: str) -> list[dict[str, Any]]:
        config = strategy.to_strategy_config()
        return self.evaluator._kb.hybrid_search(
            query, k=config.k, dense_weight=config.dense_weight,
            sparse_weight=config.sparse_weight, expand_parents=True,
            use_rerank=config.use_rerank, rerank_top_n=config.rerank_top_n,
        )

    def generate(self, query: str, contexts: list[str]) -> str:
        from .evaluation import _ANSWER_PROMPT, model
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        if model is None:
            raise RuntimeError("LLM 未配置：请设置 DEEPSEEK_API_KEY")
        prompt = ChatPromptTemplate.from_messages([("human", _ANSWER_PROMPT)])
        chain = prompt | model | StrOutputParser()
        return chain.invoke({
            "context": self.evaluator._build_context(contexts),
            "query": query,
        }).strip()

    def evaluate_quality(self, strategy: BenchmarkStrategy, entry: Any,
                         contexts: list[str], sources: list[dict[str, Any]],
                         response: str) -> dict[str, Any]:
        data = [{
            "user_input": entry.query,
            "retrieved_contexts": contexts,
            "response": response,
            "reference": entry.ground_truth,
        }]
        return self.evaluator._run_ragas(
            data, strategy.to_strategy_config(), entry, sources, response
        )


def normalize_source(value: str) -> str:
    value = (value or "").strip().replace("\\", "/")
    return Path(value).name.casefold()


def parse_relevant_sources(value: str) -> set[str]:
    return {
        normalize_source(part)
        for part in re.split(r"[,，;；|\n]+", value or "")
        if normalize_source(part)
    }


def retrieval_metrics(sources: Sequence[str], relevant: set[str], k: int = 5) -> dict[str, float]:
    ranked = [normalize_source(source) for source in sources[:k]]
    hits = [index for index, source in enumerate(ranked, 1) if source in relevant]
    unique_hits = len(set(ranked) & relevant)
    return {
        "hit_rate_at_5": 1.0 if hits else 0.0,
        "precision_at_5": unique_hits / k,
        "recall_at_5": unique_hits / len(relevant) if relevant else 0.0,
        "mrr_at_5": 1.0 / hits[0] if hits else 0.0,
    }


def percentile(values: Sequence[float], p: float) -> float | None:
    clean = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not clean:
        return None
    if len(clean) == 1:
        return round(clean[0], 4)
    position = (len(clean) - 1) * p
    lower, upper = math.floor(position), math.ceil(position)
    value = clean[lower] + (clean[upper] - clean[lower]) * (position - lower)
    return round(value, 4)


def latency_stats(values: Sequence[float]) -> dict[str, float | None]:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    return {
        "mean": round(statistics.fmean(clean), 4) if clean else None,
        "p50": percentile(clean, 0.50),
        "p95": percentile(clean, 0.95),
    }


def safe_mean(values: Sequence[float | None]) -> float | None:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return round(statistics.fmean(clean), 4) if clean else None


def change(baseline: float | None, optimized: float | None) -> dict[str, float | None]:
    if baseline is None or optimized is None:
        return {"absolute": None, "relative_pct": None}
    absolute = optimized - baseline
    relative = None if baseline == 0 else absolute / abs(baseline) * 100
    return {
        "absolute": round(absolute, 4),
        "relative_pct": round(relative, 2) if relative is not None else None,
    }


def select_entries(entries: Sequence[Any], profile: str, query_limit: int | None = None) -> list[Any]:
    selected = list(entries)
    if profile == "quick":
        selected = selected[:3]
    if query_limit is not None:
        selected = selected[:query_limit]
    return selected


def build_sources(docs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{
        "title": doc.get("title", ""),
        "source": doc.get("source", ""),
        "score": doc.get("score", 0.0),
        "content": (doc.get("content", "") or "")[:500],
    } for doc in docs]


class BenchmarkRunner:
    def __init__(self, backend: BenchmarkBackend, retrieval_repeats: int = 3,
                 seed: int = DEFAULT_SEED):
        if retrieval_repeats < 1:
            raise ValueError("retrieval_repeats 必须大于 0")
        self.backend = backend
        self.retrieval_repeats = retrieval_repeats
        self.seed = seed

    def run(self, entries: Sequence[Any]) -> list[dict[str, Any]]:
        random.seed(self.seed)
        rows: list[dict[str, Any]] = []
        strategies = {"baseline": BASELINE, "optimized": OPTIMIZED}
        if entries:
            for strategy in strategies.values():
                try:
                    self.backend.retrieve(strategy, entries[0].query)
                except Exception:
                    pass
        for index, entry in enumerate(entries):
            order = STRATEGY_NAMES if index % 2 == 0 else tuple(reversed(STRATEGY_NAMES))
            for name in order:
                rows.append(self._run_one(index, entry, strategies[name]))
        return rows

    def _run_one(self, index: int, entry: Any, strategy: BenchmarkStrategy) -> dict[str, Any]:
        row: dict[str, Any] = {
            "query_index": index, "query": entry.query, "related_docs": entry.related_docs,
            "strategy": strategy.name, "strategy_config": asdict(strategy),
            "retrieval_latencies_ms": [], "generation_latency_ms": None,
            "ragas_latency_ms": None, "end_to_end_latency_ms": None,
            "retrieval": {}, "quality": {metric: None for metric in QUALITY_METRICS},
            "metric_errors": {}, "errors": [], "sources": [], "response": "",
        }
        overall = time.perf_counter()
        docs: list[dict[str, Any]] = []
        for _ in range(self.retrieval_repeats):
            started = time.perf_counter()
            try:
                current = self.backend.retrieve(strategy, entry.query)
                docs = current
            except Exception as exc:
                row["errors"].append({"stage": "retrieval", "message": str(exc)})
                current = []
            row["retrieval_latencies_ms"].append((time.perf_counter() - started) * 1000)
        sources = build_sources(docs)
        contexts = [doc.get("content", "") for doc in docs if doc.get("content")]
        row["sources"] = sources
        row["retrieval"] = retrieval_metrics(
            [source["source"] for source in sources],
            parse_relevant_sources(entry.related_docs),
            strategy.k,
        )
        started = time.perf_counter()
        try:
            row["response"] = self.backend.generate(entry.query, contexts)
        except Exception as exc:
            row["errors"].append({"stage": "generation", "message": str(exc)})
        row["generation_latency_ms"] = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        if row["response"]:
            try:
                result = self.backend.evaluate_quality(
                    strategy, entry, contexts, sources, row["response"]
                )
                row["quality"].update(result.get("metrics", {}))
                row["metric_errors"] = result.get("metric_errors", {})
            except Exception as exc:
                row["errors"].append({"stage": "ragas", "message": str(exc)})
        else:
            row["errors"].append({"stage": "ragas", "message": "因回答生成失败而跳过"})
        row["ragas_latency_ms"] = (time.perf_counter() - started) * 1000
        row["end_to_end_latency_ms"] = (time.perf_counter() - overall) * 1000
        row["empty_retrieval"] = not contexts
        row["success"] = not row["errors"]
        return row


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for strategy in STRATEGY_NAMES:
        group = [row for row in rows if row["strategy"] == strategy]
        count = len(group)
        retrieval = {
            metric: safe_mean([row["retrieval"].get(metric) for row in group])
            for metric in ("hit_rate_at_5", "precision_at_5", "recall_at_5", "mrr_at_5")
        }
        quality = {
            metric: safe_mean([row["quality"].get(metric) for row in group])
            for metric in QUALITY_METRICS
        }
        quality["quality_composite"] = safe_mean(list(quality.values()))
        missing = sum(
            1 for row in group for metric in QUALITY_METRICS
            if row["quality"].get(metric) is None
        )
        errors = sum(1 for row in group if row["errors"])
        total_seconds = sum(row["end_to_end_latency_ms"] for row in group) / 1000
        result[strategy] = {
            "query_count": count,
            "retrieval": retrieval,
            "quality": quality,
            "performance_ms": {
                "retrieval": latency_stats([
                    value for row in group for value in row["retrieval_latencies_ms"]
                ]),
                "generation": latency_stats([row["generation_latency_ms"] for row in group]),
                "ragas": latency_stats([row["ragas_latency_ms"] for row in group]),
                "end_to_end": latency_stats([row["end_to_end_latency_ms"] for row in group]),
            },
            "reliability": {
                "success_rate": round((count - errors) / count, 4) if count else 0.0,
                "error_rate": round(errors / count, 4) if count else 0.0,
                "empty_retrieval_rate": round(
                    sum(bool(row["empty_retrieval"]) for row in group) / count, 4
                ) if count else 0.0,
                "metric_missing_rate": round(missing / (count * len(QUALITY_METRICS)), 4)
                if count else 0.0,
            },
            "throughput_qps": round(count / total_seconds, 4) if total_seconds else None,
        }
    result["comparison"] = compare_summaries(result["baseline"], result["optimized"])
    result["gate"] = regression_gate(result["baseline"], result["optimized"])
    return result


def _flatten_metrics(summary: dict[str, Any]) -> dict[str, float | None]:
    flat: dict[str, float | None] = {}
    flat.update(summary["retrieval"])
    flat.update(summary["quality"])
    flat.update(summary["reliability"])
    flat["throughput_qps"] = summary["throughput_qps"]
    for stage, stats in summary["performance_ms"].items():
        for stat, value in stats.items():
            flat[f"{stage}_latency_{stat}_ms"] = value
    return flat


def compare_summaries(baseline: dict[str, Any], optimized: dict[str, Any]) -> dict[str, Any]:
    left, right = _flatten_metrics(baseline), _flatten_metrics(optimized)
    return {
        name: {"baseline": left.get(name), "optimized": right.get(name),
               **change(left.get(name), right.get(name))}
        for name in sorted(set(left) | set(right))
    }


def regression_gate(baseline: dict[str, Any], optimized: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "recall_at_5_not_lower": (
            optimized["retrieval"]["recall_at_5"], baseline["retrieval"]["recall_at_5"], ">="
        ),
        "quality_composite_not_lower": (
            optimized["quality"]["quality_composite"], baseline["quality"]["quality_composite"], ">="
        ),
        "error_rate_not_higher": (
            optimized["reliability"]["error_rate"], baseline["reliability"]["error_rate"], "<="
        ),
        "metric_missing_rate_not_higher": (
            optimized["reliability"]["metric_missing_rate"],
            baseline["reliability"]["metric_missing_rate"], "<="
        ),
    }
    details: dict[str, Any] = {}
    for name, (actual, expected, operator) in checks.items():
        passed = actual is not None and expected is not None and (
            actual >= expected if operator == ">=" else actual <= expected
        )
        details[name] = {
            "passed": passed, "optimized": actual, "baseline": expected, "operator": operator,
        }
    return {"passed": all(item["passed"] for item in details.values()), "checks": details}


def environment_metadata(profile: str, sample_count: int, repeats: int,
                         strategies: Sequence[BenchmarkStrategy]) -> dict[str, Any]:
    from .config import DEEPSEEK_MODEL, KB_COLLECTION_NAME, MILVUS_URI, RERANKER_MODEL
    packages = {}
    for name in ("ragas", "pymilvus", "langchain", "sentence-transformers"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "profile": profile, "sample_count": sample_count,
        "retrieval_repeats": repeats, "seed": DEFAULT_SEED,
        "python": sys.version, "platform": platform.platform(), "packages": packages,
        "models": {"llm": DEEPSEEK_MODEL, "reranker": RERANKER_MODEL},
        "milvus": {"uri": MILVUS_URI, "collection": KB_COLLECTION_NAME},
        "strategies": [asdict(strategy) for strategy in strategies],
    }


def write_reports(output_dir: Path, metadata_info: dict[str, Any],
                  rows: Sequence[dict[str, Any]], summary: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"metadata": metadata_info, "results": list(rows)}
    (output_dir / "raw_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps({"metadata": metadata_info, **summary}, ensure_ascii=False,
                   indent=2, allow_nan=False), encoding="utf-8"
    )
    fields = [
        "query_index", "query", "related_docs", "strategy",
        "hit_rate_at_5", "precision_at_5", "recall_at_5", "mrr_at_5",
        *QUALITY_METRICS, "retrieval_mean_ms", "generation_latency_ms",
        "ragas_latency_ms", "end_to_end_latency_ms", "success", "errors",
    ]
    with (output_dir / "per_query.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                **{key: row.get(key) for key in fields},
                **row["retrieval"], **row["quality"],
                "retrieval_mean_ms": safe_mean(row["retrieval_latencies_ms"]),
                "errors": json.dumps(row["errors"], ensure_ascii=False),
            })
    (output_dir / "report.md").write_text(
        render_markdown(metadata_info, rows, summary), encoding="utf-8"
    )


def render_markdown(meta: dict[str, Any], rows: Sequence[dict[str, Any]],
                    summary: dict[str, Any]) -> str:
    lines = [
        "# RAG A/B 基准测试报告", "",
        f"- 运行时间：{meta['created_at']}",
        f"- 档位/样本：{meta['profile']} / {meta['sample_count']}",
        f"- 检索重复次数：{meta['retrieval_repeats']}",
        f"- 回归门禁：{'PASS' if summary['gate']['passed'] else 'FAIL'}", "",
        "## 指标对比", "",
        "| 指标 | 基线 | 优化 | 绝对变化 | 相对提升 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, item in summary["comparison"].items():
        def fmt(value: Any) -> str:
            return "N/A" if value is None else f"{value:.4f}"
        relative = "N/A" if item["relative_pct"] is None else f"{item['relative_pct']:.2f}%"
        lines.append(
            f"| {name} | {fmt(item['baseline'])} | {fmt(item['optimized'])} | "
            f"{fmt(item['absolute'])} | {relative} |"
        )
    lines.extend(["", "## 门禁详情", ""])
    for name, check in summary["gate"]["checks"].items():
        lines.append(f"- {'PASS' if check['passed'] else 'FAIL'} {name}: "
                     f"{check['optimized']} {check['operator']} {check['baseline']}")
    failures = [row for row in rows if row["errors"]]
    lines.extend(["", "## 失败样本", ""])
    if failures:
        for row in failures:
            lines.append(f"- [{row['strategy']}] {row['query']}: "
                         f"{json.dumps(row['errors'], ensure_ascii=False)}")
    else:
        lines.append("- 无")
    return "\n".join(lines) + "\n"


def default_output_dir(root: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return root / "artifacts" / "benchmarks" / stamp

