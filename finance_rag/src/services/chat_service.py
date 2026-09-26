"""Agentic RAG 问答链（LangChain LCEL + 流式输出）。

基础 Agentic RAG 流程（单轮）：
1. 元数据过滤自动推断（Self-querying）：LLM 根据问题推断分类/日期过滤条件
2. 混合检索 + 可选 BGE 重排序：统一由 KnowledgeBase.hybrid_search 完成
3. 上下文组装：按来源拼接，标注引用编号
4. 流式生成：LangChain ``astream`` 逐 token 输出

注：查询改写（多路子查询检索）已并入 LangGraph Agentic 链路
（ENABLE_LANGGRAPH，见 finance_rag.src.orchestration.graph），标准链路直接使用原始查询。
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, AsyncGenerator, Callable

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from finance_rag.src.core.config import (
    CHAT_ENABLE_QUERY_REWRITE,
    CHAT_RERANK_TOP_K,
    CHAT_TOP_K,
    DYNAMIC_K,
    ENABLE_HYDE,
    ENABLE_LANGGRAPH,
    ENABLE_RERANKER,
    HYDE_WEIGHT,
    KB_COLLECTION_NAME,
    LLM_TIMEOUT_SECONDS,
    get_model,
    get_rewrite_model,
)
from finance_rag.src.rag.models.document_category import DOCUMENT_CATEGORIES
from finance_rag.src.rag.retrieval.hybrid_retriever import HybridRetriever
from finance_rag.src.rag.ingestion.metadata_extractor import (
    load_industry_taxonomy,
    metadata_field_validators,
)
from finance_rag.src.infrastructure.vector_store.milvus_kb import get_knowledge_base
from finance_rag.src.agent.prompts.chat import (
    ANSWER_PROMPT,
    HYDE_GENERATION_PROMPT,
    METADATA_FILTER_INFER_PROMPT,
)
from finance_rag.src.agent.query_classifier import resolve_dynamic_k
from finance_rag.src.services.citation_validator import (
    REFUSAL_ANSWER,
    STREAM_WARNING,
    apply_refusal_policy,
    run_citation_validation,
)

logger = logging.getLogger(__name__)

# 单块进入 LLM 上下文的字符上限：表格与普通文本区别对待
TEXT_CONTEXT_MAX_CHARS = 2000
TABLE_CONTEXT_MAX_CHARS = HybridRetriever.TABLE_MAX_CHARS


# ---------------------------------------------------------------------------
# 上下文组装
# ---------------------------------------------------------------------------

def _context_budget(block_type: str) -> int:
    """单块进入 LLM 上下文的字符上限。

    表格放宽到 8000（数据密度最高，截到 2000 会把表体和结论一起砍掉）；
    其余块维持 2000，避免个别长块挤占其它证据。
    """
    return TABLE_CONTEXT_MAX_CHARS if block_type == "table" else TEXT_CONTEXT_MAX_CHARS


def _page_label(doc: dict[str, Any]) -> str:
    """页码标签；页码缺失（0）时返回空串而不是编一个页码。"""
    start = int(doc.get("start_page") or 0)
    end = int(doc.get("end_page") or 0)
    if start <= 0:
        return ""
    if end <= 0 or end == start:
        return f"页码：{start}"
    return f"页码：{start}-{end}"


def _block_type_label(block_type: str) -> str:
    return {"table": "表格", "image": "图片"}.get(block_type, "")


def build_context(docs: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """将检索结果组装为 LLM 上下文文本，并返回引用来源列表。

    块头带**报告日期 / 页码 / 块类型**：分析师要判断「这个数字出自哪一天的报告、
    在哪一页」，缺了这些信息结论无法被验证。缺失时留空，不猜。

    Returns:
        (context_text, sources) — context_text 含编号的文档片段，
        sources 为精简的来源元数据列表。
    """
    if not docs:
        return "（未检索到相关文档）", []

    chunks: list[str] = []
    sources: list[dict[str, Any]] = []

    for i, doc in enumerate(docs, start=1):
        block_type = str(doc.get("block_type") or "text")
        content = (doc.get("content") or "")[:_context_budget(block_type)]
        if doc.get("truncated") and block_type == "table":
            content += HybridRetriever.TABLE_TRUNCATION_NOTE
        title = doc.get("title", "金融文档")

        labels = [
            label
            for label in (
                f"报告日期：{doc['date']}" if doc.get("date") else "",
                _page_label(doc),
                f"类型：{_block_type_label(block_type)}" if _block_type_label(block_type) else "",
            )
            if label
        ]
        header = f"[{i}] 来源：{title}"
        if labels:
            header += f"（{'，'.join(labels)}）"
        chunks.append(f"{header}\n{content}")

        sources.append({
            "index": i,
            "title": title,
            "source": doc.get("source", ""),
            "category": doc.get("category", ""),
            "collection": doc.get("collection", ""),
            "chunk": doc.get("chunk"),
            "score": round(doc.get("score", 0.0), 4),
            "rerank_score": (
                round(doc["rerank_score"], 4)
                if isinstance(doc.get("rerank_score"), (int, float))
                else None
            ),
            "preview": (doc.get("content") or "")[:200],
            # 证据定位：前端据此做页码跳转与元数据 chip
            "date": doc.get("date", ""),
            "block_type": block_type,
            "start_page": int(doc.get("start_page") or 0),
            "end_page": int(doc.get("end_page") or 0),
            "image_key": doc.get("image_key", ""),
            "truncated": bool(doc.get("truncated")),
            "security_code": doc.get("security_code", ""),
            "industry_l1": doc.get("industry_l1", ""),
            "report_type": doc.get("report_type", ""),
        })

    context_text = "\n\n---\n\n".join(chunks)
    return context_text, sources


def rerank_scores(sources: list[dict[str, Any]]) -> list[float]:
    """提取来源中的 rerank_score 列表（仅保留数值项）。"""
    return [
        float(s["rerank_score"])
        for s in sources
        if isinstance(s.get("rerank_score"), (int, float))
    ]


def _is_timeout_error(exc: Exception) -> bool:
    """判断异常是否为 LLM 请求超时/连接错误。"""
    from finance_rag.src.core.exceptions import classify_llm_error

    return classify_llm_error(exc) in ("timeout", "connection")


# ---------------------------------------------------------------------------
# 查询改写 + 多路子查询（标准链路可选增强，受 CHAT_ENABLE_QUERY_REWRITE 控制）
# ---------------------------------------------------------------------------

# 查询改写 Prompt：提取关键词 + 改写主查询 + 生成多路子查询
QUERY_REWRITE_PROMPT = """你是一个金融领域的查询分析和改写助手。

用户问题：{query}
历史对话：{history}

请按以下格式输出（不要加任何解释）：
1. 关键词列表：提取3-5个最核心的关键词，用逗号分隔
2. 改写查询：将问题改写为简洁的检索查询（保留所有核心关键词）
3. 子查询1：提取关键词"[时间]"相关的查询
4. 子查询2：提取关键词"[核心领域]"相关的查询
5. 子查询3：提取关键词"[动作/措施]"相关的查询

输出示例：
关键词：房地产,2023年,金融支持
改写查询：2023年金融支持房地产市场政策措施
子查询1：2023年金融政策
子查询2：房地产金融支持
子查询3：金融支持措施

请开始输出："""


def _format_history(history: list[dict[str, str]] | None) -> str:
    """格式化对话历史为文本。"""
    if not history:
        return "（无）"
    parts: list[str] = []
    for msg in history[-6:]:  # 最近 3 轮
        role = msg.get("role", "user")
        content = msg.get("content", "")[:200]
        label = "用户" if role == "user" else "助手"
        parts.append(f"{label}: {content}")
    return "\n".join(parts) if parts else "（无）"


def _extract_keywords_simple(query: str) -> list[str]:
    """简单关键词提取（无需 LLM，LLM 改写失败时的回退方案）。"""
    keywords: list[str] = []

    # 时间关键词
    for year in ['2021', '2022', '2023', '2024', '2025', '2026']:
        if year in query:
            keywords.append(year + '年')

    # 核心领域关键词
    domain_keywords = ['房地产', '金融', '银行', '货币政策', '财政政策',
                      '支持', '稳定', '风险', '监管', '市场']
    for kw in domain_keywords:
        if kw in query:
            keywords.append(kw)

    # 动作/措施关键词
    action_keywords = ['措施', '政策', '支持', '采取', '实施', '推动', '促进']
    for kw in action_keywords:
        if kw in query:
            keywords.append(kw)

    return keywords[:5]  # 最多5个


def _generate_sub_queries(main_query: str, keywords: list[str]) -> list[str]:
    """根据主查询和关键词生成子查询（关键词两两组合）。"""
    sub_queries: list[str] = []

    if len(keywords) >= 2:
        # 两两组合生成子查询
        for i in range(min(len(keywords), 3)):
            if i + 1 < len(keywords):
                sq = keywords[i] + keywords[i + 1]
                sub_queries.append(sq)
            elif len(keywords) >= 2:
                # 与最后一个关键词组合
                sq = keywords[i] + keywords[0]
                sub_queries.append(sq)

    if not sub_queries:
        sub_queries = [main_query]

    return sub_queries[:3]


def _parse_rewrite_output(raw: str, original_query: str) -> dict[str, Any]:
    """解析 LLM 返回的改写输出（关键词 / 改写查询 / 子查询）。"""
    rewritten = original_query
    keywords: list[str] = []
    sub_queries: list[str] = []

    for line in raw.split('\n'):
        line = line.strip()
        if not line:
            continue

        # 关键词
        if line.startswith('关键词') or line.startswith('关键字'):
            kw_part = line.split('：', 1)[-1].split(':', 1)[-1].strip()
            keywords = [k.strip() for k in kw_part.split(',') if k.strip()]

        # 改写查询
        elif '改写查询' in line or '改写' in line:
            rw_part = line.split('：', 1)[-1].split(':', 1)[-1].strip()
            if rw_part:
                rewritten = rw_part

        # 子查询
        elif '子查询' in line:
            sq_part = line.split('：', 1)[-1].split(':', 1)[-1].strip()
            if sq_part:
                sub_queries.append(sq_part)

    # 如果解析失败，使用简单提取
    if not keywords:
        keywords = _extract_keywords_simple(original_query)

    if not sub_queries:
        # 生成简单子查询
        sub_queries = _generate_sub_queries(rewritten, keywords)

    return {
        'rewritten': rewritten,
        'keywords': keywords,
        'sub_queries': sub_queries,
    }


def rewrite_query(query: str, history: list[dict[str, str]] | None = None) -> dict[str, Any]:
    """分析查询，返回改写查询、关键词和子查询。

    优先使用轻量改写模型（``rewrite_model``，默认 DashScope qwen-turbo），
    未配置时回退主模型 ``model``，两者均不可用或调用失败时使用规则改写。

    Returns:
        dict with keys:
        - 'rewritten': 改写后的主查询
        - 'keywords': 关键词列表
        - 'sub_queries': 子查询列表
    """
    chosen = get_rewrite_model() or get_model()
    if chosen is None:
        return {
            'rewritten': query,
            'keywords': _extract_keywords_simple(query),
            'sub_queries': _generate_sub_queries(query, _extract_keywords_simple(query)),
        }

    history_text = _format_history(history) if history else "（无）"
    try:
        prompt = ChatPromptTemplate.from_messages([
            ("human", QUERY_REWRITE_PROMPT),
        ])
        chain = prompt | chosen | StrOutputParser()
        raw_output = chain.invoke({"query": query, "history": history_text})

        # 解析结构化输出
        result = _parse_rewrite_output(raw_output, query)
        logger.info(
            "查询改写：%s → 主查询=%s, 关键词=%s, 子查询=%s",
            query, result['rewritten'], result['keywords'], result['sub_queries']
        )
        return result
    except Exception as exc:
        if _is_timeout_error(exc):
            logger.warning("查询改写超时（%ss），回退到原查询", int(LLM_TIMEOUT_SECONDS))
        else:
            logger.warning("查询改写失败，使用原查询：%s", exc)
        return {
            'rewritten': query,
            'keywords': _extract_keywords_simple(query),
            'sub_queries': [query],
        }


# ---------------------------------------------------------------------------
# 元数据过滤自动推断（Self-querying Retrieval）
# ---------------------------------------------------------------------------

# 中文标签 -> 英文枚举值映射（LLM 可能输出中文分类名）
_CATEGORY_LABEL_TO_VALUE = {
    label: value for value, label in DOCUMENT_CATEGORIES.items()
}


def _normalize_date_value(value: Any) -> str | None:
    """规范化日期过滤值：'2024' -> '2024-01-01'，非法格式返回 None。"""
    if not isinstance(value, str):
        return None
    value = value.strip()
    if re.fullmatch(r"\d{4}", value):
        return f"{value}-01-01"
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return value
    return None


def _parse_filters_output(raw: str) -> dict[str, Any]:
    """解析并校验 LLM 输出的过滤条件。

    容错策略：
    - 从原始文本中截取首尾花括号之间的 JSON（容忍代码块包裹/多余文本）
    - 兼容顶层键或 ``metadata`` 包装键两种输出形式
    - 真正的白名单与值校验统一走 :func:`normalize_metadata_filters`，
      与 API 直传路径共用一套规则（避免两条路各校验一半）
    """
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end <= start:
        return {}
    try:
        payload = json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}
    # 兼容 LLM 用 metadata 包装输出
    if isinstance(payload.get("metadata"), dict):
        payload = payload["metadata"]
    return normalize_metadata_filters(payload)


def _collect_valid_values(value: Any, validator: Callable[[str], bool]) -> list[str]:
    """把标量/列表统一收敛成「去重后的合法值列表」。"""
    if value is None:
        return []
    items = value if isinstance(value, (list, tuple)) else [value]
    collected: list[str] = []
    for item in items:
        if not isinstance(item, str):
            continue
        text = item.strip()
        if text and validator(text) and text not in collected:
            collected.append(text)
    return collected


def normalize_metadata_filters(filters: dict[str, Any] | None) -> dict[str, Any]:
    """把过滤条件收敛到白名单 + 合法值。

    **两条路径共用**：LLM 自动推断（:func:`infer_metadata_filters`）与 API 直传
    （``ChatRequest.filters``）。过滤条件最终会被拼成 Milvus 表达式，未校验的键
    等于把过滤面暴露给调用方，因此这里只放行已知字段与已知取值。

    值得注意的两条业务规则：

    - ``security_code`` 为**单值字段**。问题里出现多只票时，只过滤其中一只会让
      另外几只静默消失，因此多于一个不同代码时**整条过滤丢弃**（宁可召回宽）。
    - ``industry_l2`` 必须属于所选的 ``industry_l1``；否则丢二级、留一级。

    所有值统一以列表形式返回（Milvus 的 ``in [...]`` 对单值字段同样成立）。
    """
    if not filters:
        return {}
    taxonomy = load_industry_taxonomy()
    result: dict[str, Any] = {}

    # category：列表；中文标签 -> 英文枚举
    categories = _collect_valid_values(
        filters.get("category"),
        lambda value: _CATEGORY_LABEL_TO_VALUE.get(value, value) in DOCUMENT_CATEGORIES,
    )
    if categories:
        result["category"] = [
            _CATEGORY_LABEL_TO_VALUE.get(value, value) for value in categories
        ]

    # date：范围条件
    date_cond = filters.get("date")
    if isinstance(date_cond, dict):
        cond: dict[str, str] = {}
        for op in ("gte", "lte"):
            normalized = _normalize_date_value(date_cond.get(op))
            if normalized:
                cond[op] = normalized
        if cond:
            result["date"] = cond

    # 研报维度字段（校验器与入库/API 修正共用一套，避免三条路径规则漂移）
    validators: dict[str, Callable[[str], bool]] = {
        field: validator
        for field, validator in metadata_field_validators().items()
        if field != "security_name"      # 简称不用于过滤
    }
    for field, validator in validators.items():
        values = _collect_valid_values(filters.get(field), validator)
        if values:
            result[field] = values

    # 单值字段的收敛规则：多标的查询不能只过滤其中一只
    codes = result.get("security_code")
    if codes and len(codes) > 1:
        result.pop("security_code")

    # 二级行业必须属于所选一级
    level1 = result.get("industry_l1")
    if level1 and result.get("industry_l2"):
        allowed = {name for name in level1 for name in taxonomy.get(name, ())}
        narrowed = [name for name in result["industry_l2"] if name in allowed]
        if narrowed:
            result["industry_l2"] = narrowed
        else:
            result.pop("industry_l2")

    return result


def infer_metadata_filters(query: str) -> dict[str, Any]:
    """用 LLM 根据用户问题自动推断元数据过滤条件（Self-querying）。

    Returns:
        形如 ``{"category": [...], "date": {...}}`` 的过滤条件；
        推断失败、超时或问题中无明确限定时返回 ``{}``（不过滤）。
    """
    if get_model() is None:
        return {}
    try:
        prompt = ChatPromptTemplate.from_messages([
            ("human", METADATA_FILTER_INFER_PROMPT),
        ])
        chain = prompt | get_model() | StrOutputParser()
        raw = chain.invoke({"query": query})
        filters = _parse_filters_output(raw)
        if filters:
            logger.info("自动元数据过滤：%s -> %s", query, filters)
        return filters
    except Exception as exc:
        if _is_timeout_error(exc):
            logger.warning("元数据过滤推断超时，跳过过滤")
        else:
            logger.warning("元数据过滤推断失败，忽略：%s", exc)
        return {}


# ---------------------------------------------------------------------------
# 检索 + 重排
# ---------------------------------------------------------------------------

def reranking_enabled(requested: bool) -> bool:
    return ENABLE_RERANKER and requested


def iter_active_kbs() -> list[tuple[str, Any]]:
    """返回参与问答检索的 ``(集合名, KnowledgeBase)`` 列表。

    本项目是**单物理集合 + 逻辑分类视图**的设计：知识库注册表里的「类别」
    （投研类 / 合规风控类 / …）只是同一集合内的 ``category`` 标量取值，
    并不是独立的 Milvus 集合。因此这里恒为默认集合一项。

    历史实现会遍历注册表并检查 ``item.get("exists")``，但注册表从不产出该键，
    于是那段循环永远落空、每次都要白跑一次注册表查询——既慢又误导，
    已改为直接返回默认集合。
    """
    return [(KB_COLLECTION_NAME, get_knowledge_base())]


# ---------------------------------------------------------------------------
# 流式问答
# ---------------------------------------------------------------------------

def merge_docs(docs_a: list[dict[str, Any]], docs_b: list[dict[str, Any]], 
                top_k: int, boost_keywords: list[str] | None = None) -> list[dict[str, Any]]:
    """合并两组检索结果，基于 content 去重，支持关键词加权。

    加权策略：如果文档内容包含核心关键词，提升其排序分数。
    排序以分数为主键、**报告日期为次键**（同分时新报告优先）：研报的时效性
    直接影响结论是否成立，不能只靠召回顺序决定。
    """
    seen: set[str] = set()
    merged: list[dict[str, Any]] = []
    
    for doc in docs_a + docs_b:
        key = (doc.get("content", "") or "")[:200]
        if key not in seen:
            seen.add(key)
            merged.append(doc)
    
    # 关键词加权：如果文档包含核心关键词，提升其分数
    if boost_keywords:
        for doc in merged:
            content = doc.get("content", "")
            score = doc.get("score", 0.0)
            boost = 1.0
            for kw in boost_keywords:
                if kw and kw in content:
                    boost += 0.15  # 每个关键词+15%权重
            doc["score"] = score * boost
        # 加权后重新归一化，确保分数保持在 0-1 区间
        max_s = max((d.get("score", 0.0) for d in merged), default=0.0)
        if max_s > 0:
            for doc in merged:
                doc["score"] = round(doc["score"] / max_s, 4)
    
    merged.sort(
        key=lambda d: (d.get("score", 0.0), str(d.get("date") or "")),
        reverse=True,
    )
    return merged[:top_k]


def _resolve_k_params(
    query: str,
    k: int | None,
    rerank_top_n: int | None,
) -> tuple[int, int]:
    """解析召回参数：用户显式传参优先；否则应用默认值与动态 K。"""
    user_specified_k = k is not None
    effective_k = k if user_specified_k else CHAT_TOP_K
    effective_rtn = rerank_top_n if rerank_top_n is not None else CHAT_RERANK_TOP_K
    if DYNAMIC_K and not user_specified_k:
        return resolve_dynamic_k(query, effective_k, effective_rtn)
    return effective_k, effective_rtn


def generate_hyde_answer(query: str) -> str:
    """让 LLM 生成假设性回答（HyDE 检索增强用），失败返回空串。"""
    if get_model() is None:
        return ""
    try:
        prompt = ChatPromptTemplate.from_messages([("human", HYDE_GENERATION_PROMPT)])
        chain = prompt | get_model() | StrOutputParser()
        return str(chain.invoke({"query": query}) or "")
    except Exception as exc:
        logger.warning("HyDE 假设回答生成失败：%s", exc)
        return ""


def hyde_retrieve(
    query: str,
    k: int,
    kbs: list[tuple[str, Any]],
    filters: dict[str, Any] | None,
    use_rerank: bool,
    rerank_top_n: int,
) -> list[dict[str, Any]]:
    """HyDE 检索：用假设回答的稠密向量检索，分数按 HYDE_WEIGHT 加权。"""
    hyde_text = generate_hyde_answer(query)
    if not hyde_text or not hyde_text.strip():
        return []

    merged: list[dict[str, Any]] = []
    for coll_name, kb in kbs:
        try:
            docs = kb.hybrid_search(
                hyde_text,
                k=max(k, 5),
                use_dense_only=True,
                expand_parents=True,
                use_rerank=use_rerank,
                rerank_top_n=rerank_top_n,
                filters=filters,
            )
        except Exception as exc:
            logger.warning("HyDE 检索失败（%s）：%s", coll_name, exc)
            continue
        for d in docs:
            d["collection"] = coll_name
            d["score"] = float(d.get("score", 0.0) or 0.0) * HYDE_WEIGHT
        merged.extend(docs)
    return merged


async def retrieve_pipeline(
    query: str,
    history: list[dict[str, str]] | None = None,
    *,
    use_rerank: bool = True,
    k: int | None = None,
    rerank_top_n: int | None = None,
    filters: dict[str, Any] | None = None,
    infer_filters: bool = True,
    stage_cb: Callable[[str], Any] | None = None,
    keywords: list[str] | None = None,
) -> dict[str, Any]:
    """完整检索管线（无流式副作用）。

    流程：查询改写 + 多路子查询（可选）→ 元数据过滤推断（可选）→
    跨库混合检索 → 去重/关键词加权合并 → 可选 HyDE 增强 → 空结果稠密回退。

    查询改写由 ``CHAT_ENABLE_QUERY_REWRITE`` 开关控制（默认关闭，单查询检索）；
    改写逻辑与 LangGraph 的 rewrite 节点共用 :func:`rewrite_query`。

    供 :func:`chat_stream` 与 A/B 评估器复用。``stage_cb`` 为可选阶段
    回调（同步），阶段按执行顺序依次上报。

    Returns:
        ``{"docs", "rewritten", "filters", "k", "rerank_top_n"}``
    """
    def _stage(name: str) -> None:
        if stage_cb is not None:
            stage_cb(name)

    # 解析召回参数：用户显式传参优先，否则默认值 + 可选动态 K
    k, rerank_top_n = _resolve_k_params(query, k, rerank_top_n)

    # Step 1: 查询改写 + 多路子查询（可选）
    # 开启时改写主查询并拆分为多个子查询，多路检索后按关键词加权合并
    # 若调用方显式传入 keywords（如合规查询理解产出），则优先使用，不再走改写
    if keywords is None:
        keywords = []
    if CHAT_ENABLE_QUERY_REWRITE and not keywords:
        _stage("rewriting")
        query_info = rewrite_query(query, history)
        rewritten = query_info['rewritten']
        keywords = query_info['keywords']
        all_queries = [rewritten] + query_info['sub_queries']
        all_queries = list(dict.fromkeys(all_queries))  # 去重保序
    else:
        rewritten = query
        all_queries = [query]

    # Step 2: 元数据过滤自动推断（Self-querying）
    # 请求未显式携带 filters 时，由 LLM 根据问题自动推断；显式 filters 优先级更高
    if infer_filters and not filters:
        _stage("inferring_filters")
        filters = infer_metadata_filters(query)

    # Step 2b: 无论来自 API 还是 LLM 推断，都收敛到白名单 + 合法值。
    # 过滤条件最终会拼成 Milvus 表达式，未校验的键等于把过滤面暴露给调用方。
    filters = normalize_metadata_filters(filters)

    # Step 3: 混合检索 + 可选 BGE 重排序
    effective_use_rerank = reranking_enabled(use_rerank)
    _stage("retrieving")
    if effective_use_rerank:
        _stage("reranking")

    # 单物理集合检索（逻辑分类通过 filters.category 收窄）
    kbs = iter_active_kbs()
    infra_failed_kbs: set[str] = set()
    all_kb_names = {name for name, _ in kbs}

    def _search_one_kb(
        search_query: str,
        coll_name: str,
        kb: Any,
        *,
        dense_only: bool = False,
    ) -> list[dict[str, Any]]:
        """对单个知识库执行检索，结果打上 collection 标记。"""
        from finance_rag.src.core.exceptions import (
            EmbeddingError,
            VectorStoreError,
        )

        try:
            docs = kb.hybrid_search(
                search_query,
                k=k,
                use_dense_only=dense_only,
                expand_parents=True,
                use_rerank=effective_use_rerank,
                rerank_top_n=rerank_top_n,
                filters=filters,
                keywords=keywords,
            )
        except (VectorStoreError, EmbeddingError) as exc:
            infra_failed_kbs.add(coll_name)
            logger.error("检索基础设施异常（%s）: %s", coll_name, exc)
            return []
        except Exception as exc:
            logger.warning("检索失败（%s）: %s", coll_name, exc)
            return []
        for d in docs:
            d["collection"] = coll_name
        return docs

    def _do_search(search_query: str) -> list[dict[str, Any]]:
        """跨所有知识库执行单次检索并合并。"""
        merged: list[dict[str, Any]] = []
        for coll_name, kb in kbs:
            merged.extend(_search_one_kb(search_query, coll_name, kb))
        return merged

    # 多查询跨库检索（改写开启时为「改写查询 + 子查询」，关闭时为单查询）
    # to_thread：Milvus 检索 + BGE 重排序均为同步阻塞调用，放线程池避免阻塞事件循环
    all_results: list[dict[str, Any]] = []
    for search_query in all_queries:
        all_results.extend(await asyncio.to_thread(_do_search, search_query))

    # 合并去重结果（改写开启时按关键词加权）
    docs = merge_docs(all_results, [], k, boost_keywords=keywords)

    # HyDE 检索增强（可选）：假设回答向量检索结果加权合并
    if ENABLE_HYDE and get_model() is not None:
        _stage("hyde")
        hyde_docs = await asyncio.to_thread(
            hyde_retrieve,
            query,
            k,
            kbs,
            filters,
            effective_use_rerank,
            rerank_top_n,
        )
        if hyde_docs:
            docs = merge_docs(docs, hyde_docs, k, boost_keywords=keywords)

    # 如果结果为空，回退到简单稠密检索
    if not docs:
        logger.info("混合检索无结果，尝试简单稠密检索")
        simple_docs: list[dict[str, Any]] = []
        for coll_name, kb in kbs:
            simple_docs.extend(
                await asyncio.to_thread(
                    _search_one_kb, query, coll_name, kb, dense_only=True
                )
            )
        if simple_docs:
            docs = merge_docs(docs, simple_docs, k, boost_keywords=keywords)

    # 所有知识库均因基础设施异常而失败（而非真正的空结果）时给出错误说明
    retrieval_error: str | None = None
    if not docs and infra_failed_kbs and infra_failed_kbs == all_kb_names:
        retrieval_error = (
            "检索服务暂不可用（向量数据库/嵌入服务异常），请稍后重试"
        )

    return {
        "docs": docs,
        "rewritten": rewritten,
        "filters": filters,
        "k": k,
        "rerank_top_n": rerank_top_n,
        "retrieval_error": retrieval_error,
    }


async def chat_stream(
    query: str,
    history: list[dict[str, str]] | None = None,
    *,
    use_rerank: bool = True,
    k: int | None = None,
    rerank_top_n: int | None = None,
    filters: dict[str, Any] | None = None,
) -> AsyncGenerator[dict[str, Any], None]:
    """流式问答异步生成器。

    检索流程（标准链路）：
    1. 元数据过滤自动推断（可选）
    2. 混合检索 + 去重合并 + 可选 BGE 重排序
    3. 组装上下文并流式生成

    按顺序 yield 以下事件：
    * ``{"type": "status", "stage": "retrieving"|"reranking"|"generating"}``
    * ``{"type": "sources", "sources": [...]}``
    * ``{"type": "token", "content": "..."}``（多次）
    * ``{"type": "done", "rewritten_query": "...", "source_count": N}``
    """
    if get_model() is None:
        yield {"type": "error", "message": "LLM 未配置（缺少 DEEPSEEK_API_KEY），请在 .env 中配置后重启服务"}
        return

    # LangGraph Agentic RAG 路由
    if ENABLE_LANGGRAPH:
        from finance_rag.src.orchestration.graph import run_agent_stream as langgraph_stream
        async for event in langgraph_stream(query, history):
            yield event
        return

    # 检索管线（阶段经回调收集，完成后按顺序回放状态事件）
    stages: list[str] = []
    try:
        pipeline = await retrieve_pipeline(
            query, history,
            use_rerank=use_rerank,
            k=k,
            rerank_top_n=rerank_top_n,
            filters=filters,
            stage_cb=stages.append,
        )
    except Exception as exc:
        logger.exception("检索失败：%s", exc)
        yield {"type": "error", "message": f"检索失败：{exc}"}
        return

    for stage in stages:
        yield {"type": "status", "stage": stage}

    docs = pipeline["docs"]
    rewritten = pipeline["rewritten"]
    filters = pipeline["filters"]
    retrieval_error = pipeline.get("retrieval_error")

    if not docs:
        # 基础设施故障（向量库/嵌入服务不可用）与「库内无内容」区分对待
        if retrieval_error:
            yield {"type": "error", "message": retrieval_error}
            return
        # 拒答机制：检索为空时按策略返回拒答文案
        refused, _ = apply_refusal_policy(None, 0)
        fallback = REFUSAL_ANSWER if refused else "未在知识库中检索到相关内容，请先上传文档或调整问题。"
        yield {"type": "sources", "sources": []}
        yield {"type": "token", "content": fallback}
        yield {
            "type": "done",
            "rewritten_query": rewritten,
            "source_count": 0,
            "filters": filters,
            "answer_rejected": refused,
            "low_confidence": False,
        }
        return

    # Step 3: 组装上下文
    context_text, sources = build_context(docs)
    yield {"type": "sources", "sources": sources}

    # Step 4: 流式生成答案
    yield {"type": "status", "stage": "generating"}
    prompt = ChatPromptTemplate.from_messages([
        ("human", ANSWER_PROMPT),
    ])
    chain = prompt | get_model() | StrOutputParser()

    answer_parts: list[str] = []
    try:
        async for chunk in chain.astream({
            "context": context_text,
            "query": query,
        }):
            if chunk:
                answer_parts.append(chunk)
                yield {"type": "token", "content": chunk}
    except Exception as exc:
        if _is_timeout_error(exc):
            logger.warning("LLM 流式生成超时（%ss）：%s", int(LLM_TIMEOUT_SECONDS), exc)
            yield {"type": "error", "message": f"连接超时（{int(LLM_TIMEOUT_SECONDS)}s），请稍后重试"}
        else:
            logger.exception("流式生成失败：%s", exc)
            yield {"type": "error", "message": f"生成出错：{exc}"}

    # 引用验证 + 拒答策略（流式已输出不可撤回：低分时尾部追加警示）
    full_answer = "".join(answer_parts)
    validation = run_citation_validation(full_answer, sources)
    rejected, low_conf = apply_refusal_policy(
        validation, len(sources), rerank_scores=rerank_scores(sources)
    )
    if rejected and full_answer and not full_answer.startswith(REFUSAL_ANSWER):
        score = validation.get("score", 0.0) if validation else 0.0
        yield {"type": "token", "content": STREAM_WARNING.format(score=score)}
        low_conf = True

    yield {
        "type": "done",
        "rewritten_query": rewritten,
        "source_count": len(sources),
        "filters": filters,
        "citation_validation": validation,
        "answer_rejected": False,  # 流式内容已输出，仅警示不替换
        "low_confidence": low_conf,
    }


async def chat(
    query: str,
    history: list[dict[str, str]] | None = None,
    *,
    use_rerank: bool = True,
    k: int | None = None,
    rerank_top_n: int | None = None,
    filters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """非流式问答，返回完整答案 + 来源。

    Returns:
        ``{"answer", "sources", "rewritten_query"}``
    """
    answer_parts: list[str] = []
    sources: list[dict[str, Any]] = []
    rewritten_query = query
    validation: dict[str, Any] | None = None

    async for event in chat_stream(
        query, history,
        use_rerank=use_rerank,
        k=k,
        rerank_top_n=rerank_top_n,
        filters=filters,
    ):
        etype = event.get("type")
        if etype == "token":
            answer_parts.append(event.get("content", ""))
        elif etype == "sources":
            sources = event.get("sources", [])
        elif etype == "done":
            rewritten_query = event.get("rewritten_query", query)
            # 复用 chat_stream 已产出的引用校验结果，避免重复做嵌入计算
            validation = event.get("citation_validation")

    result = {
        "answer": "".join(answer_parts),
        "sources": sources,
        "rewritten_query": rewritten_query,
    }

    if validation is not None:
        result["citation_validation"] = validation
        logger.info(
            "引用验证完成：score=%.2f, issues=%d",
            validation["score"],
            len(validation["issues"]),
        )

    # 拒答策略：低分引用 / 空检索 / 来源相关性不足时替换答案为拒答文案
    rejected, low_conf = apply_refusal_policy(
        validation, len(sources), rerank_scores=rerank_scores(sources)
    )
    result["answer_rejected"] = rejected
    result["low_confidence"] = low_conf
    if rejected:
        result["answer"] = REFUSAL_ANSWER
        logger.info("拒答策略触发：source_count=%d", len(sources))

    return result
