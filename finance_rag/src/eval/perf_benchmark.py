"""性能测试对比模块。

对新的检索链路（ONNX INT8 量化 + RRF 融合 + 父子块）进行专项性能测试，
输出检索质量、向量检索吞吐量、单条查询延迟等指标。

核心组件：
* :class:`PerfBenchmarkRunner` — 性能测试运行器
* :class:`PerfMetric` — 单次性能指标数据类
* :func:`write_perf_report` — 将结果写入 JSON / Markdown 报告
"""

from __future__ import annotations

import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .benchmark import latency_stats, percentile, retrieval_metrics
from config.settings import EMBED_BATCH_SIZE, EMBEDDING_MODEL, KB_COLLECTION_NAME, MILVUS_URI, RRF_K
from .ragas_eval import get_test_set_loader
from finance_rag.src.infrastructure.vector_store.milvus_kb import get_knowledge_base


@dataclass(frozen=True)
class PerfMetric:
    """单次性能指标。"""

    name: str
    value: float | None = None
    unit: str = ""
    details: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "details": self.details or {},
        }


class PerfBenchmarkRunner:
    """性能测试运行器。

    测试项：
    1. 检索质量 — top-3/top-5 准确率、MRR、证据召回率
    2. 嵌入吞吐量 — 不同 batch size 下每秒嵌入文本数
    3. 检索延迟 — P50 / P95 混合检索延迟
    4. 重排序延迟 — P50 / P95 BGE 重排序延迟
    5. 端到端延迟 — 完整检索链路 P50 / P95
    """

    def __init__(self, repeat: int = 5, rrf_k: int = RRF_K):
        self._kb = get_knowledge_base()
        self._repeat = repeat
        self._rrf_k = rrf_k
        self._entries = get_test_set_loader().load_test_set()

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def run_all(self) -> dict[str, Any]:
        """执行全部性能测试并返回结构化结果。"""
        return {
            "metadata": self._metadata(),
            "retrieval_quality": [m.to_dict() for m in self._bench_retrieval_quality()],
            "embedding_throughput": [m.to_dict() for m in self._bench_embedding_throughput()],
            "retrieval_latency": [m.to_dict() for m in self._bench_retrieval_latency()],
            "rerank_latency": [m.to_dict() for m in self._bench_rerank_latency()],
            "end_to_end_latency": [m.to_dict() for m in self._bench_end_to_end()],
        }

    # ------------------------------------------------------------------
    # 1. 检索质量
    # ------------------------------------------------------------------

    def _bench_retrieval_quality(self) -> list[PerfMetric]:
        """在测试集上评估检索质量。"""
        if not self._entries:
            return [PerfMetric("retrieval_quality", None, "", {"error": "无测试集"})]

        top3_hits: list[float] = []
        top5_hits: list[float] = []
        mrr_values: list[float] = []
        recall_values: list[float] = []
        evidence_recalls: list[float] = []

        for entry in self._entries:
            docs = self._kb.hybrid_search(
                entry.query,
                k=5,
                expand_parents=True,
                use_rerank=False,
                rrf_k=self._rrf_k,
            )
            sources = [d.get("source", "") for d in docs]
            relevant = self._parse_relevant_sources(entry.related_docs)

            # top-3 / top-5 hit rate
            top3_hits.append(1.0 if any(normalize_source(s) in relevant for s in sources[:3]) else 0.0)
            top5_hits.append(1.0 if any(normalize_source(s) in relevant for s in sources[:5]) else 0.0)

            # MRR @ 5
            metrics = retrieval_metrics(sources, relevant, k=5)
            mrr_values.append(metrics["mrr_at_5"])
            recall_values.append(metrics["recall_at_5"])

            # 证据召回率：检索内容中是否包含 required_facts 关键词
            if entry.required_facts:
                contexts = " ".join(d.get("content", "") for d in docs)
                found = sum(1 for fact in entry.required_facts if fact in contexts)
                evidence_recalls.append(found / len(entry.required_facts))
            else:
                evidence_recalls.append(1.0 if not relevant else 0.0)

        def avg(values: Sequence[float]) -> float:
            return round(statistics.fmean(values), 4) if values else 0.0

        return [
            PerfMetric("top3_hit_rate", avg(top3_hits), "ratio", {"sample_count": len(self._entries)}),
            PerfMetric("top5_hit_rate", avg(top5_hits), "ratio", {"sample_count": len(self._entries)}),
            PerfMetric("mrr_at_5", avg(mrr_values), "ratio", {"sample_count": len(self._entries)}),
            PerfMetric("recall_at_5", avg(recall_values), "ratio", {"sample_count": len(self._entries)}),
            PerfMetric("evidence_recall", avg(evidence_recalls), "ratio", {"sample_count": len(self._entries)}),
        ]

    # ------------------------------------------------------------------
    # 2. 嵌入吞吐量
    # ------------------------------------------------------------------

    def _bench_embedding_throughput(self) -> list[PerfMetric]:
        """测试 ONNX INT8 嵌入器在不同 batch size 下的吞吐量。"""
        embedder = self._kb._get_embeddings()
        # 构造模拟文本（平均 100 字中文）
        dummy_texts = ["这是一段用于测试嵌入吞吐量的模拟金融文档内容，" * 5 for _ in range(200)]

        results: list[PerfMetric] = []
        for batch_size in [1, 8, 16, 32, 64]:
            latencies: list[float] = []
            for i in range(0, len(dummy_texts), batch_size):
                batch = dummy_texts[i : i + batch_size]
                t0 = time.perf_counter()
                embedder.embed_documents(batch)
                latencies.append(time.perf_counter() - t0)

            total_time = sum(latencies)
            total_texts = len(dummy_texts)
            throughput = total_texts / total_time if total_time > 0 else 0.0
            results.append(
                PerfMetric(
                    f"embedding_throughput_bs{batch_size}",
                    round(throughput, 2),
                    "texts/sec",
                    {
                        "batch_size": batch_size,
                        "total_texts": total_texts,
                        "total_time_sec": round(total_time, 4),
                        "p50_latency_ms": percentile([v * 1000 for v in latencies], 0.50),
                        "p95_latency_ms": percentile([v * 1000 for v in latencies], 0.95),
                    },
                )
            )
        return results

    # ------------------------------------------------------------------
    # 3. 检索延迟
    # ------------------------------------------------------------------

    def _bench_retrieval_latency(self) -> list[PerfMetric]:
        """测试纯混合检索（无重排序、无父子扩展）的延迟分布。"""
        if not self._entries:
            return [PerfMetric("retrieval_latency", None, "", {"error": "无测试集"})]

        latencies: list[float] = []
        for entry in self._entries:
            for _ in range(self._repeat):
                t0 = time.perf_counter()
                self._kb.hybrid_search(
                    entry.query,
                    k=5,
                    expand_parents=False,
                    use_rerank=False,
                    rrf_k=self._rrf_k,
                )
                latencies.append((time.perf_counter() - t0) * 1000)

        stats = latency_stats(latencies)
        return [
            PerfMetric("retrieval_p50_ms", stats.get("p50"), "ms", {"samples": len(latencies)}),
            PerfMetric("retrieval_p95_ms", stats.get("p95"), "ms", {"samples": len(latencies)}),
            PerfMetric("retrieval_mean_ms", stats.get("mean"), "ms", {"samples": len(latencies)}),
        ]

    # ------------------------------------------------------------------
    # 4. 重排序延迟
    # ------------------------------------------------------------------

    def _bench_rerank_latency(self) -> list[PerfMetric]:
        """测试 BGE 重排序的延迟分布。"""
        if not self._entries:
            return [PerfMetric("rerank_latency", None, "", {"error": "无测试集"})]

        # 先检索一批候选，用于重排序
        candidates_map: dict[str, list[dict[str, Any]]] = {}
        for entry in self._entries:
            candidates = self._kb.hybrid_search(
                entry.query,
                k=15,
                expand_parents=False,
                use_rerank=False,
                rrf_k=self._rrf_k,
            )
            candidates_map[entry.query] = candidates

        latencies: list[float] = []
        for entry in self._entries:
            candidates = candidates_map.get(entry.query, [])
            if not candidates:
                continue
            for _ in range(self._repeat):
                t0 = time.perf_counter()
                self._kb.hybrid_search(
                    entry.query,
                    k=5,
                    expand_parents=False,
                    use_rerank=True,
                    rerank_top_n=5,
                    rrf_k=self._rrf_k,
                )
                latencies.append((time.perf_counter() - t0) * 1000)

        if not latencies:
            return [PerfMetric("rerank_latency", None, "", {"error": "无候选可重排序"})]

        stats = latency_stats(latencies)
        return [
            PerfMetric("rerank_p50_ms", stats.get("p50"), "ms", {"samples": len(latencies)}),
            PerfMetric("rerank_p95_ms", stats.get("p95"), "ms", {"samples": len(latencies)}),
            PerfMetric("rerank_mean_ms", stats.get("mean"), "ms", {"samples": len(latencies)}),
        ]

    # ------------------------------------------------------------------
    # 5. 端到端延迟
    # ------------------------------------------------------------------

    def _bench_end_to_end(self) -> list[PerfMetric]:
        """测试完整链路（混合检索 + 父子扩展 + 重排序）的延迟分布。"""
        if not self._entries:
            return [PerfMetric("end_to_end_latency", None, "", {"error": "无测试集"})]

        latencies: list[float] = []
        for entry in self._entries:
            for _ in range(self._repeat):
                t0 = time.perf_counter()
                self._kb.hybrid_search(
                    entry.query,
                    k=5,
                    expand_parents=True,
                    use_rerank=True,
                    rerank_top_n=3,
                    rrf_k=self._rrf_k,
                )
                latencies.append((time.perf_counter() - t0) * 1000)

        stats = latency_stats(latencies)
        return [
            PerfMetric("end_to_end_p50_ms", stats.get("p50"), "ms", {"samples": len(latencies)}),
            PerfMetric("end_to_end_p95_ms", stats.get("p95"), "ms", {"samples": len(latencies)}),
            PerfMetric("end_to_end_mean_ms", stats.get("mean"), "ms", {"samples": len(latencies)}),
        ]

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------

    def _metadata(self) -> dict[str, Any]:
        return {
            "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
            "repeat": self._repeat,
            "rrf_k": self._rrf_k,
            "test_set_size": len(self._entries),
            "embedding_model": EMBEDDING_MODEL,
            "collection": KB_COLLECTION_NAME,
            "milvus_uri": MILVUS_URI,
        }

    @staticmethod
    def _parse_relevant_sources(value: str) -> set[str]:
        import re
        return {
            normalize_source(part)
            for part in re.split(r"[,，;；|\n]+", value or "")
            if normalize_source(part)
        }


def normalize_source(value: str) -> str:
    value = (value or "").strip().replace("\\", "/")
    return Path(value).name.casefold()


# ---------------------------------------------------------------------------
# 报告输出
# ---------------------------------------------------------------------------

def write_perf_report(output_dir: Path, results: dict[str, Any]) -> None:
    """将性能测试结果写入 JSON 与 Markdown 报告。"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    (output_dir / "perf_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    # Markdown
    (output_dir / "perf_report.md").write_text(
        _render_perf_markdown(results),
        encoding="utf-8",
    )


def _render_perf_markdown(results: dict[str, Any]) -> str:
    meta = results.get("metadata", {})
    lines = [
        "# RAG 性能测试报告", "",
        f"- 运行时间：{meta.get('created_at', 'N/A')}",
        f"- 测试集大小：{meta.get('test_set_size', 'N/A')}",
        f"- 重复次数：{meta.get('repeat', 'N/A')}",
        f"- RRF k：{meta.get('rrf_k', 'N/A')}",
        f"- 嵌入模型：{meta.get('embedding_model', 'N/A')}",
        f"- Milvus 集合：{meta.get('collection', 'N/A')}",
        "",
    ]

    def render_section(title: str, key: str) -> None:
        lines.extend([f"## {title}", ""])
        items = results.get(key, [])
        if not items:
            lines.append("- 无数据")
        else:
            lines.append("| 指标 | 值 | 单位 | 详情 |")
            lines.append("|---|---|---|---|")
            for item in items:
                name = item.get("name", "")
                value = item.get("value")
                value_str = f"{value:.4f}" if isinstance(value, float) else (str(value) if value is not None else "N/A")
                unit = item.get("unit", "")
                details = item.get("details", {})
                detail_str = ", ".join(f"{k}={v}" for k, v in details.items() if k != "samples")
                lines.append(f"| {name} | {value_str} | {unit} | {detail_str} |")
        lines.append("")

    render_section("检索质量", "retrieval_quality")
    render_section("嵌入吞吐量", "embedding_throughput")
    render_section("检索延迟", "retrieval_latency")
    render_section("重排序延迟", "rerank_latency")
    render_section("端到端延迟", "end_to_end_latency")

    return "\n".join(lines) + "\n"
