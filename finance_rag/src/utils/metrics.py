"""Prometheus 指标收集模块。

基于 prometheus-fastapi-instrumentator 自动暴露 HTTP 指标，
并注册自定义业务指标：检索延迟、重排序延迟、LLM 调用次数。
"""

from __future__ import annotations

from prometheus_client import Counter, Histogram

# ---------------------------------------------------------------------------
# 自定义业务指标
# ---------------------------------------------------------------------------

# 检索延迟（秒）
retrieval_latency = Histogram(
    "finance_rag_retrieval_latency_seconds",
    "Retrieval latency in seconds",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

# 重排序延迟（秒）
reranker_latency = Histogram(
    "finance_rag_reranker_latency_seconds",
    "Reranker latency in seconds",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

# LLM 调用次数（按阶段）
llm_call_total = Counter(
    "finance_rag_llm_call_total",
    "Total LLM calls by stage",
    ["stage"],  # rewrite, generate, evaluate
)

# 检索结果质量（rerank_score 分布）
rerank_score = Histogram(
    "finance_rag_rerank_score",
    "Reranker score distribution",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0],
)

# 文档操作计数
document_ops = Counter(
    "finance_rag_document_ops_total",
    "Document operations",
    ["operation"],  # upload, delete, skip
)

# 断崖检测触发次数
cliff_detection_total = Counter(
    "finance_rag_cliff_detection_total",
    "Cliff detection triggered count",
)
