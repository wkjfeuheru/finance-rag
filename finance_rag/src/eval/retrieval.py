"""生产检索策略到统一评测运行器的适配层。"""

from __future__ import annotations

from typing import Any, Callable

from .runner import EvaluationStrategy, RetrievalOutput

_ORIGINAL_CLIFF_DETECTION: Callable[..., list[dict[str, Any]]] | None = None

LAYER_SPECS: dict[str, dict[str, Any]] = {
    "L0": {
        "name": "纯稠密检索", "k": 20, "top_k": 5, "use_dense_only": True,
        "use_rerank": False, "cliff_detection": False,
    },
    "L1": {
        "name": "混合检索(RRF)", "k": 20, "top_k": 5, "use_dense_only": False,
        "use_rerank": False, "cliff_detection": False,
    },
    "L2": {
        "name": "混合检索，未重排序", "k": 5, "top_k": 5, "use_dense_only": False,
        "use_rerank": False, "cliff_detection": False,
    },
    "L3": {
        "name": "混合检索 + BGE", "k": 20, "top_k": 5, "use_dense_only": False,
        "use_rerank": True, "cliff_detection": False,
    },
    "L5": {
        "name": "混合检索 + BGE + 断崖检测", "k": 20, "top_k": 5,
        "use_dense_only": False, "use_rerank": True, "cliff_detection": True,
    },
}


def configure_cliff_detection(enabled: bool) -> None:
    """只在当前评测进程切换生产断崖函数，不修改线上配置。"""
    global _ORIGINAL_CLIFF_DETECTION
    from finance_rag.src.rag.retrieval import hybrid_retriever

    if _ORIGINAL_CLIFF_DETECTION is None:
        _ORIGINAL_CLIFF_DETECTION = hybrid_retriever._apply_cliff_detection
    if enabled:
        hybrid_retriever._apply_cliff_detection = _ORIGINAL_CLIFF_DETECTION
        return

    def passthrough(candidates: list[dict[str, Any]], *_args: Any, **_kwargs: Any):
        return candidates

    hybrid_retriever._apply_cliff_detection = passthrough


def build_retrieval_strategy(
    kb: Any,
    layer: str,
    *,
    candidate_k: int = 20,
    top_k: int = 5,
) -> EvaluationStrategy:
    """构造 L0/L1/L2/L3/L5 策略；L4 永久不在评测矩阵中。"""
    key = layer.upper()
    if key not in LAYER_SPECS:
        raise ValueError(f"未知检索层：{layer}；可选 {', '.join(LAYER_SPECS)}")
    spec = dict(LAYER_SPECS[key])
    spec["top_k"] = top_k
    spec["k"] = top_k if key == "L2" else candidate_k

    def retrieve(query: str) -> RetrievalOutput:
        if spec["use_rerank"]:
            configure_cliff_detection(bool(spec["cliff_detection"]))
        results = kb.hybrid_search(
            query,
            k=spec["k"],
            use_rerank=spec["use_rerank"],
            rerank_top_n=top_k,
            # 块级指标需要原始 chunk id；父块扩展既改变上下文粒度，又会额外依赖
            # PostgreSQL，不属于本次检索层优化变量。
            expand_parents=False,
            use_dense_only=spec["use_dense_only"],
        )
        final = list(results[:top_k])
        return RetrievalOutput(
            candidates=list(results),
            results=final,
            cliff_triggered=bool(spec["cliff_detection"] and len(final) < top_k),
            pre_cliff_depth=min(spec["k"], len(results)) if spec["use_rerank"] else len(results),
            post_cliff_depth=len(final),
        )

    return EvaluationStrategy(name=key, retrieve=retrieve, config=spec)
