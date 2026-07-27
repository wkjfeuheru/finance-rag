"""混合检索与重排序模块。

提供：
* :class:`BGEReranker` — 本地 BGE 重排序器（延迟加载并缓存模型）
* :class:`HybridRetriever` — 稠密 + 稀疏（BM25）混合检索器

检索流程：
1. 稠密向量检索（DashScope text-embedding-v2 + COSINE）
2. 稀疏向量检索（Milvus BM25 Function）
3. WeightedRanker 融合两路结果
4. 可选 BGE 重排序（异常时回退到原始排序）
"""

from __future__ import annotations

import logging
import time
from typing import Any

from pymilvus import (
    AnnSearchRequest,
    MilvusClient,
    WeightedRanker,
)

from .config import (
    HYBRID_DENSE_WEIGHT,
    HYBRID_SPARSE_WEIGHT,
    RERANKER_DEVICE,
    RERANKER_MODEL,
)
from .logging_config import agent_logger

logger = logging.getLogger(__name__)

# 检索时过采样倍数（为重排序提供更多候选）
_OVERSAMPLE_FACTOR = 3


# ---------------------------------------------------------------------------
# 本地 BGE 重排序器
# ---------------------------------------------------------------------------

class BGEReranker:
    """本地 BGE 重排序器（延迟加载并缓存模型）。

    使用 ``sentence-transformers`` 的 ``CrossEncoder`` 加载
    ``BAAI/bge-reranker-v2-m3``（或兼容模型），完全本地推理，无需 API。

    模型在首次调用时加载，并以类变量缓存，避免重复分配约 2GB 内存。
    """

    _model_instance = None  # 类级缓存

    def __init__(
        self,
        model_name: str = RERANKER_MODEL,
        max_length: int = 512,
        device: str = RERANKER_DEVICE,
        use_fp16: bool = False,
    ):
        self._model_name = model_name
        self._max_length = max_length
        self._device = device
        self._use_fp16 = use_fp16

    def _get_model(self):
        """延迟加载并缓存 CrossEncoder 模型。"""
        if BGEReranker._model_instance is None:
            from sentence_transformers import CrossEncoder

            BGEReranker._model_instance = CrossEncoder(
                self._model_name,
                max_length=self._max_length,
                device=self._device,
            )
            if self._use_fp16 and self._device != "cpu":
                BGEReranker._model_instance.model.half()
        return BGEReranker._model_instance

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_n: int = 3,
    ) -> list[dict[str, Any]]:
        """对候选结果重排序，返回 top_n 条。

        Args:
            query: 查询文本。
            candidates: 候选结果列表（每项含 ``content`` 字段）。
            top_n: 返回的结果数量。

        Returns:
            重排序后的 top_n 结果，每项含 ``rerank_score`` 字段。

        Note:
            模型加载或推理失败时回退到原始排序，保证检索链路可用。
        """
        if not candidates:
            return []

        agent_logger.rag_rerank_start(candidate_count=len(candidates))
        t0 = time.perf_counter()

        try:
            model = self._get_model()
            pairs = [[query, c.get("content", "")[:2000]] for c in candidates]
            scores = model.predict(pairs)
            for i, score in enumerate(scores):
                if i < len(candidates):
                    candidates[i]["rerank_score"] = float(score)
            candidates.sort(key=lambda c: c.get("rerank_score", 0.0), reverse=True)
        except Exception as exc:
            logger.warning("BGE 重排序失败，回退到原始排序：%s", exc)
            for c in candidates:
                c.setdefault("rerank_score", c.get("score", 0.0))
            candidates.sort(key=lambda c: c.get("rerank_score", 0.0), reverse=True)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        agent_logger.rag_rerank_end(
            elapsed_ms=elapsed_ms,
            result_count=min(top_n, len(candidates)),
        )
        return candidates[:top_n]


# ---------------------------------------------------------------------------
# 混合检索器
# ---------------------------------------------------------------------------

class HybridRetriever:
    """稠密 + 稀疏（BM25）混合检索器。

    封装 Milvus hybrid_search 调用，支持：
    * 稠密向量 + BM25 稀疏向量双路检索
    * WeightedRanker 权重融合（可调稠密/稀疏权重）
    * 可选 BGE 重排序（异常时回退到原始排序）

    Args:
        client: MilvusClient 实例。
        collection_name: Milvus 集合名。
        embed_query_fn: 查询文本 → 稠密向量的函数。
        reranker: 可选的 BGEReranker 实例（为 None 时按需创建）。
    """

    def __init__(
        self,
        client: MilvusClient,
        collection_name: str,
        embed_query_fn,
        reranker: BGEReranker | None = None,
    ):
        self._client = client
        self._collection_name = collection_name
        self._embed_query = embed_query_fn
        self._reranker = reranker

    def search(
        self,
        query: str,
        k: int = 5,
        *,
        dense_weight: float = HYBRID_DENSE_WEIGHT,
        sparse_weight: float = HYBRID_SPARSE_WEIGHT,
        expand_parents: bool = True,
        use_rerank: bool = False,
        rerank_top_n: int = 3,
    ) -> list[dict[str, Any]]:
        """混合检索 + 可选 BGE 重排序。

        Args:
            query: 查询文本。
            k: 返回结果数（重排序关闭时的返回数量）。
            dense_weight: 稠密向量权重。
            sparse_weight: 稀疏向量权重。
            expand_parents: 兼容参数；单层 Hybrid chunks 不执行父块扩展。
            use_rerank: 是否启用 BGE 本地重排序。
            rerank_top_n: 启用重排序时的最终返回数量。

        Returns:
            检索结果列表，每项含 content/source/title/chunk/score/parent_id。
            重排序启用时额外含 ``rerank_score`` 字段。

        Note:
            重排序失败时自动回退到 Milvus 原始排序，保证检索链路可用。
        """
        t0 = time.perf_counter()

        # 重排序开启时过采样候选，否则按 k 检索
        fetch_k = max(k, rerank_top_n * _OVERSAMPLE_FACTOR) if use_rerank else k
        limit = fetch_k * _OVERSAMPLE_FACTOR

        # 稠密向量检索请求
        dense_vector = self._embed_query(query)
        dense_req = AnnSearchRequest(
            data=[dense_vector],
            anns_field="dense_vector",
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=limit,
        )

        # 稀疏向量检索请求（Milvus 自动对 query 文本做 BM25 编码）
        sparse_req = AnnSearchRequest(
            data=[query],
            anns_field="sparse_vector",
            param={"metric_type": "BM25"},
            limit=limit,
        )

        # 混合检索：WeightedRanker 融合
        results = self._client.hybrid_search(
            collection_name=self._collection_name,
            reqs=[dense_req, sparse_req],
            ranker=WeightedRanker(dense_weight, sparse_weight),
            limit=limit,
            output_fields=["content", "source", "title", "chunk", "parent_id"],
        )

        matches: list[dict[str, Any]] = []
        for hit in results[0]:
            entity = hit.get("entity", {})
            matches.append({
                "content": entity.get("content"),
                "source": entity.get("source"),
                "title": entity.get("title") or "金融文档",
                "chunk": entity.get("chunk"),
                "parent_id": entity.get("parent_id", ""),
                "score": float(hit.get("distance", hit.get("score", 0.0))),
            })

        elapsed_ms = (time.perf_counter() - t0) * 1000
        agent_logger.knowledge_retrieved(
            faq_hits=len(matches),
            rule_hits=0,
            product_hits=0,
            elapsed_ms=elapsed_ms,
            query_len=len(query),
        )

        # BGE 重排序（异常时回退到原始排序）
        if use_rerank and matches:
            reranker = self._get_reranker()
            # 重排序后返回 max(k, rerank_top_n) 条，确保不少于 K
            # 避免启用重排序后上下文数量锐减导致 context_recall 下降
            final_n = max(k, rerank_top_n)
            matches = reranker.rerank(query, matches, top_n=final_n)
            return matches

        return matches[:k]

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _get_reranker(self) -> BGEReranker:
        """延迟创建 BGE 重排序器实例。"""
        if self._reranker is None:
            self._reranker = BGEReranker()
        return self._reranker
