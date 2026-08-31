"""混合检索与重排序模块。

提供：
* :class:`BGEReranker` — 本地 BGE 重排序器（延迟加载并缓存模型）
* :class:`HybridRetriever` — 稠密 + 稀疏（BM25）混合检索器

检索流程：
1. 稠密向量检索（ONNX INT8 本地嵌入 + COSINE）
2. 稀疏向量检索（Milvus 自动对 query 文本做 BM25 编码）
3. RRFRanker 融合两路结果（或纯稠密模式 use_dense_only=True）
4. 可选父子扩展：按 parent_id 去重并替换为父块全文
5. 可选 BGE 重排序（异常时回退到原始排序）
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from pymilvus import (
    AnnSearchRequest,
    MilvusClient,
    RRFRanker,
)

from finance_rag.src.core.config import (
    APP_ENV,
    ENABLE_VERSIONING,
    MILVUS_NPROBE,
    MODEL_OFFLINE,
    RERANKER_CLIFF_MIN_RESULTS,
    RERANKER_CLIFF_THRESHOLD,
    RERANKER_DEVICE,
    RERANKER_MODEL,
    RRF_K,
)
from finance_rag.src.core.exceptions import VectorStoreError
from finance_rag.src.core.logger import agent_logger
from finance_rag.src.rag.retrieval.parent_store import ParentStore

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
    # 类级推理锁：模型实例为类变量单例，多线程（asyncio.to_thread 后并发请求）
    # 同时调用 predict 时串行化推理，避免 sentence-transformers 非线程安全问题
    _infer_lock = threading.Lock()
    # 模型加载锁：保护 _model_instance 的 check-then-act，避免并发重复加载约 2GB 权重
    _model_lock = threading.Lock()

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
            with BGEReranker._model_lock:
                if BGEReranker._model_instance is None:
                    from sentence_transformers import CrossEncoder

                    device = self._device
                    if device != "cpu" and APP_ENV != "production":
                        try:
                            import torch

                            if not torch.cuda.is_available():
                                logger.warning(
                                    "RERANKER_DEVICE=%s 但 CUDA 不可用，回退到 CPU", device
                                )
                                device = "cpu"
                        except ImportError:
                            device = "cpu"

                    model_kwargs = {"local_files_only": True} if MODEL_OFFLINE else {}
                    BGEReranker._model_instance = CrossEncoder(
                        self._model_name,
                        max_length=self._max_length,
                        device=device,
                        model_kwargs=model_kwargs,
                    )
                    if self._use_fp16 and device != "cpu":
                        BGEReranker._model_instance.model.half()
        return BGEReranker._model_instance

    def warmup(self) -> None:
        """启动预热：加载模型并执行一次小规模推理。

        将模型加载（可能含下载/解析权重）从「首个请求」挪到服务启动阶段，
        避免生产环境首请求冷启动超时。预热失败必须由调用方处理，防止
        服务在模型不可用时被误标记为就绪。
        """
        model = self._get_model()
        with BGEReranker._infer_lock:
            model.predict([["预热请求", "预热上下文"]])
        logger.info(
            "BGE 重排序模型预热完成：%s (device=%s)", self._model_name, self._device
        )
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
            with BGEReranker._infer_lock:
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
        final_candidates = _apply_cliff_detection(candidates)
        result_count = min(top_n, len(final_candidates))
        agent_logger.rag_rerank_end(
            elapsed_ms=elapsed_ms,
            result_count=result_count,
        )
        return final_candidates[:top_n]


# ---------------------------------------------------------------------------
# 断崖检测
# ---------------------------------------------------------------------------

def _apply_cliff_detection(
    candidates: list[dict[str, Any]],
    threshold: float = RERANKER_CLIFF_THRESHOLD,
    min_results: int = RERANKER_CLIFF_MIN_RESULTS,
) -> list[dict[str, Any]]:
    """检测 rerank_score 断崖，截断低质结果。

    算法：遍历已按 score 降序排列的候选，计算相邻分数的相对下降。
    当 ``(score[i] - score[i+1]) / max(score[i], 0.001) > threshold`` 时，
    认为在 i+1 处出现断崖，丢弃 i+1 及之后的所有结果。

    至少保留 min_results 条，threshold=0 时禁用。
    """
    if not candidates or threshold <= 0 or len(candidates) <= 1:
        return candidates

    scores = [c.get("rerank_score", 0.0) for c in candidates]
    cut = len(candidates)  # 默认不截断

    for i in range(len(scores) - 1):
        if scores[i] <= 0:
            continue
        drop = (scores[i] - scores[i + 1]) / max(scores[i], 0.001)
        if drop > threshold:
            cut = max(min_results, i + 1)
            logger.info(
                "断崖检测：score[%d]=%.4f → score[%d]=%.4f (下降 %.1f%%)，截断为 %d/%d 条",
                i, scores[i], i + 1, scores[i + 1], drop * 100, cut, len(candidates),
            )
            break

    return candidates[:cut]


# ---------------------------------------------------------------------------
# 混合检索器
# ---------------------------------------------------------------------------

class HybridRetriever:
    """稠密 + 稀疏（BM25）混合检索器。

    封装 Milvus hybrid_search / search 调用，支持：
    * 稠密向量 + BM25 稀疏向量双路检索
    * RRFRanker 融合（默认 k=60）
    * 纯稠密模式（use_dense_only=True 时切换）
    * 父子扩展：按 parent_id 去重并替换为父块全文
    * 可选 BGE 重排序（异常时回退到原始排序）

    Args:
        client: MilvusClient 实例。
        collection_name: Milvus 集合名。
        embed_query_fn: 查询文本 -> 稠密向量的函数。
        reranker: 可选的 BGEReranker 实例（为 None 时按需创建）。
        parent_store: 可选的 ParentStore 实例，用于父子扩展。
    """

    def __init__(
        self,
        client: MilvusClient,
        collection_name: str,
        embed_query_fn,
        reranker: BGEReranker | None = None,
        parent_store: ParentStore | None = None,
    ):
        self._client = client
        self._collection_name = collection_name
        self._embed_query = embed_query_fn
        self._reranker = reranker
        self._parent_store = parent_store

    def search(
        self,
        query: str,
        k: int = 5,
        *,
        use_dense_only: bool = False,
        expand_parents: bool = True,
        use_rerank: bool = False,
        rerank_top_n: int = 3,
        filters: dict[str, Any] | None = None,
        rrf_k: int = RRF_K,
        keywords: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """混合检索 + 可选父子扩展 + 可选 BGE 重排序 + 关键词增强。

        流程：
        1. 稠密向量检索（ONNX INT8 本地嵌入 + COSINE）
        2. 稀疏向量检索（Milvus 自动对 query 文本做 BM25 编码）
        3. RRFRanker 融合两路结果（或纯稠密模式 use_dense_only=True）
        4. 可选父子扩展：按 parent_id 去重并替换为父块全文
        5. **关键词过滤重排序**：如果提供了关键词，优先保留包含关键词的文档
        6. 可选 BGE 重排序（异常时回退到原始排序）

        Args:
            query: 查询文本。
            k: 返回结果数（重排序关闭时的返回数量）。
            use_dense_only: True 时只走稠密向量检索，False 时走 RRF 混合检索。
            expand_parents: 是否按 parent_id 去重并替换为父块全文。
            use_rerank: 是否启用 BGE 本地重排序。
            rerank_top_n: 启用重排序时的最终返回数量。
            filters: 元数据过滤条件，如 {"source": "report.pdf"}。
                仅当 ENABLE_METADATA_FILTER=true 时生效。
            rrf_k: RRF 融合常数，默认 60。
            keywords: 可选的关键词列表，用于过滤和重排序。

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

        dense_vector = self._embed_query(query)
        filter_expr = self._build_filter(filters)
        # 版本保留模式：仅检索最新版本（is_current == true）
        if ENABLE_VERSIONING:
            filter_expr = self._combine_filter(filter_expr, "is_current == true")
        output_fields = ["content", "source", "title", "chunk", "parent_id", "category", "date"]

        if use_dense_only:
            # 纯稠密模式
            search_kwargs = {
                "collection_name": self._collection_name,
                "data": [dense_vector],
                "anns_field": "dense_vector",
                "search_params": {"metric_type": "COSINE", "params": {"nprobe": MILVUS_NPROBE}},
                "limit": limit,
                "output_fields": output_fields,
            }
            if filter_expr:
                search_kwargs["filter"] = filter_expr
            try:
                results = self._client.search(**search_kwargs)
            except Exception as exc:
                raise VectorStoreError(f"向量检索失败：{exc}") from exc
        else:
            # RRF 混合模式
            # 注意：pymilvus 的 hybrid_search 过滤条件必须放在每个 AnnSearchRequest
            # 的 expr 上，顶层 filter 参数会被忽略
            dense_req = AnnSearchRequest(
                data=[dense_vector],
                anns_field="dense_vector",
                param={"metric_type": "COSINE", "params": {"nprobe": MILVUS_NPROBE}},
                limit=limit,
                expr=filter_expr or None,
            )
            sparse_req = AnnSearchRequest(
                data=[query],
                anns_field="sparse_vector",
                param={"metric_type": "BM25"},
                limit=limit,
                expr=filter_expr or None,
            )
            search_kwargs = {
                "collection_name": self._collection_name,
                "reqs": [dense_req, sparse_req],
                "ranker": RRFRanker(k=rrf_k),
                "limit": limit,
                "output_fields": output_fields,
            }
            try:
                results = self._client.hybrid_search(**search_kwargs)
            except Exception as exc:
                raise VectorStoreError(f"混合检索失败：{exc}") from exc

        matches: list[dict[str, Any]] = []
        for hit in results[0]:
            entity = hit.get("entity", {})
            matches.append({
                "content": entity.get("content"),
                "source": entity.get("source"),
                "title": entity.get("title") or "金融文档",
                "chunk": entity.get("chunk"),
                "parent_id": entity.get("parent_id", ""),
                "category": entity.get("category", "") or "",
                "date": entity.get("date", "") or "",
                "score": float(hit.get("distance", hit.get("score", 0.0))),
            })

        # 父子扩展
        if expand_parents and self._parent_store and matches:
            matches = self._expand_to_parents(matches, k)

        # 关键词过滤与重排序（核心优化）
        if keywords and matches:
            matches = self._apply_keyword_boost(matches, keywords, k)

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
            matches = reranker.rerank(query, matches, top_n=rerank_top_n)
            # 重排序后以 rerank_score 作为对外展示的相关度分数
            for m in matches:
                if m.get("rerank_score") is not None:
                    m["score"] = m["rerank_score"]
        else:
            matches = matches[:k]

        # 分数归一化：将 RRF 融合分数（天然极小值，如 0.02）或 BGE 重排序分数
        # 映射到 0-1 区间，使首条结果分数为 1.0，其余按比例缩放，便于前端直观展示。
        if matches:
            scores = [m.get("score", 0) for m in matches if m.get("score") is not None]
            max_score = max(scores) if scores else 0
            if max_score > 0:
                for m in matches:
                    raw = m.get("score")
                    if raw is not None:
                        m["score"] = round(float(raw) / max_score, 4)

        return matches

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _get_reranker(self) -> BGEReranker:
        """延迟创建 BGE 重排序器实例。"""
        if self._reranker is None:
            self._reranker = BGEReranker()
        return self._reranker

    @staticmethod
    def _quote(value: Any) -> str:
        """Milvus 表达式值序列化：字符串加双引号，其余原样。"""
        if isinstance(value, str):
            return f'"{value}"'
        return str(value)

    @staticmethod
    def _build_scalar_condition(field: str, value: Any) -> str:
        """构造标量字段过滤条件（== 或 in）。"""
        if isinstance(value, (list, tuple)):
            items = ", ".join(HybridRetriever._quote(v) for v in value)
            return f"{field} in [{items}]"
        return f"{field} == {HybridRetriever._quote(value)}"

    @staticmethod
    def _build_date_condition(value: Any) -> str | None:
        """构造 date 字段过滤条件（ISO 日期字符串的字典序即时间序）。

        value 为标量时精确匹配；为 dict 时支持 gte/lte/gt/lt 范围比较。
        """
        op_map = {"gte": ">=", "lte": "<=", "gt": ">", "lt": "<"}
        if isinstance(value, dict):
            conditions = [
                f"date {op_map[op]} {HybridRetriever._quote(v)}"
                for op, v in value.items()
                if op in op_map
            ]
            return " and ".join(conditions) or None
        if value is None or value == "":
            return None
        return f"date == {HybridRetriever._quote(value)}"

    @staticmethod
    def _build_filter(filters: dict[str, Any] | None) -> str:
        """构造 Milvus 过滤表达式。

        支持的顶层键：
        - ``category``：分类，标量（==）或列表（in）
        - ``date``：日期，标量（==）或含 gte/lte/gt/lt 的 dict（范围比较）
        - 其他标量字段（如 source）：精确匹配 ``==`` 或 ``in``（向后兼容）

        示例::

            {"category": ["compliance_risk"], "date": {"gte": "2024-01-01"}}
            -> category in ["compliance_risk"] and date >= "2024-01-01"
        """
        if not filters:
            return ""
        conditions = []
        for field, value in filters.items():
            if field == "date":
                cond = HybridRetriever._build_date_condition(value)
                if cond:
                    conditions.append(cond)
            else:
                conditions.append(HybridRetriever._build_scalar_condition(field, value))
        return " and ".join(conditions)

    @staticmethod
    def _combine_filter(existing: str, extra: str) -> str:
        """将附加过滤条件与已有表达式按 and 组合。"""
        if not existing:
            return extra
        if not extra:
            return existing
        return f"({existing}) and ({extra})"

    def _expand_to_parents(
        self, matches: list[dict[str, Any]], k: int
    ) -> list[dict[str, Any]]:
        """按 parent_id 去重，查询父块全文替换子块 content。

        优化：
        - 父块过大（>3000字符）时，保留原始子块内容避免噪音扩散
        - 父块过小（<50字符）时，保留原始子块内容（可能只是标题）
        """
        PARENT_MAX_CHARS = 3000
        PARENT_MIN_CHARS = 50

        seen_parents: set[str] = set()
        unique: list[dict[str, Any]] = []
        for m in matches:
            pid = m.get("parent_id", "")
            if pid and pid not in seen_parents:
                seen_parents.add(pid)
                unique.append(m)
            elif not pid:
                unique.append(m)

        parent_ids = [m["parent_id"] for m in unique if m.get("parent_id")]
        if parent_ids:
            try:
                parents_map = {
                    p["id"]: p for p in self._parent_store.get_batch(parent_ids)
                }
            except Exception as exc:
                logger.warning("父子扩展查询父块失败，回退为子块内容：%s", exc)
                return unique[:k]
            for m in unique:
                pid = m.get("parent_id", "")
                if pid in parents_map:
                    parent_content = parents_map[pid]["content"]
                    # 只有当父块大小合适时才替换
                    if PARENT_MIN_CHARS <= len(parent_content) <= PARENT_MAX_CHARS:
                        m["content"] = parent_content
                        m["heading"] = parents_map[pid].get("heading", "")
                    elif len(parent_content) > PARENT_MAX_CHARS:
                        # 父块过大，保留子块内容但标注来源
                        m["heading"] = parents_map[pid].get("heading", "")
        return unique[:k]

    def _apply_keyword_boost(
        self, matches: list[dict[str, Any]], keywords: list[str], k: int
    ) -> list[dict[str, Any]]:
        """根据关键词对检索结果进行过滤和重排序。

        策略：
        1. 遍历所有匹配，计算每个文档的关键词命中分数
        2. 支持精确匹配和子串匹配
        3. 将文档分为两组：包含关键词的（relevant）和不包含的（other）
        4. 相关文档按关键词命中数排序，非相关文档保留在后面
        5. 返回 top_k 个结果
        """
        if not keywords or not matches:
            return matches

        # 计算每个文档的关键词命中情况
        for m in matches:
            content = m.get("content", "") or ""
            hit_count = 0
            for kw in keywords:
                if not kw:
                    continue
                # 精确匹配（直接包含）
                if kw in content:
                    hit_count += 2
                    continue
                # 多字符关键词的子串匹配
                # 检查关键词中是否有足够比例的字符出现在内容中
                kw_chars = [c for c in kw if c.strip()]
                if len(kw_chars) >= 2:
                    matched_chars = sum(1 for c in kw_chars if c in content)
                    # 至少60%的字符匹配
                    if matched_chars >= max(1, int(len(kw_chars) * 0.6)):
                        hit_count += 1
            m["_keyword_hits"] = hit_count

        # 分组：相关 vs 非相关
        relevant = [m for m in matches if m.get("_keyword_hits", 0) > 0]
        other = [m for m in matches if m.get("_keyword_hits", 0) == 0]

        # 相关文档按关键词命中数降序，然后按原始分数降序
        relevant.sort(key=lambda m: (m["_keyword_hits"], m.get("score", 0.0)), reverse=True)

        # 合并：相关文档在前，非相关文档在后
        boosted = relevant + other

        # 清理临时字段
        for m in boosted:
            m.pop("_keyword_hits", None)

        logger.info(
            "关键词增强：%d/%d 个文档包含关键词，top 关键词=%s",
            len(relevant), len(matches), keywords[:3],
        )

        return boosted[:k]


