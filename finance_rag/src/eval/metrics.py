"""无外部依赖的统一评测指标实现。"""

from __future__ import annotations


def compute_chunk_retrieval_metrics(
    retrieved_chunk_ids: list[str],
    evidence_chunk_ids: list[str] | tuple[str, ...] | None,
    k: int = 5,
) -> dict[str, float | None]:
    """计算块级 Recall@k、固定分母 Precision@k 和 MRR。"""
    evidence = {str(chunk_id) for chunk_id in (evidence_chunk_ids or []) if chunk_id}
    if not evidence:
        return {"recall_at_k": None, "precision_at_k": None, "mrr": None}
    top_k = [str(chunk_id) for chunk_id in retrieved_chunk_ids[:k] if chunk_id]
    hit_ranks = [rank for rank, chunk_id in enumerate(top_k, 1) if chunk_id in evidence]
    hit_ids = {top_k[rank - 1] for rank in hit_ranks}
    return {
        "recall_at_k": round(len(hit_ids) / len(evidence), 4),
        "precision_at_k": round(len(hit_ids) / k, 4),
        "mrr": round(1.0 / min(hit_ranks), 4) if hit_ranks else 0.0,
    }
