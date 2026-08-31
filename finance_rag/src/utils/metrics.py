"""Prometheus 指标收集模块。

基于 prometheus-fastapi-instrumentator 自动暴露 HTTP 指标，
并注册自定义业务指标：重排序分数分布、文档操作计数。
"""

from __future__ import annotations

from prometheus_client import Counter, Histogram

# ---------------------------------------------------------------------------
# 自定义业务指标
# ---------------------------------------------------------------------------

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
