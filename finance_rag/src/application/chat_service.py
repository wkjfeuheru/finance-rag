"""Agentic RAG 问答链（LangChain LCEL + 流式输出）。

基础 Agentic RAG 流程（单轮）：
1. 查询改写（可选）：LLM 将口语化问题改写为检索友好 query
2. 混合检索 + 可选 BGE 重排序：统一由 KnowledgeBase.hybrid_search 完成
3. 上下文组装：按来源拼接，标注引用编号
4. 流式生成：LangChain ``astream`` 逐 token 输出

"""

from __future__ import annotations

import logging
from typing import Any, AsyncGenerator

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from config.settings import (
    CHAT_ENABLE_QUERY_REWRITE,
    CHAT_RERANK_TOP_K,
    CHAT_TOP_K,
    ENABLE_CITATION_VALIDATION,
    ENABLE_FINANCIAL_EXPERT_PROMPT,
    ENABLE_LANGGRAPH,
    ENABLE_METADATA_FILTER,
    ENABLE_MULTI_STAGE_RETRIEVAL,
    ENABLE_RERANKER,
    model,
)
from finance_rag.src.infrastructure.vector_store.milvus_kb import get_knowledge_base
from finance_rag.src.application.chat_prompts import (
    FINANCIAL_EXPERT_ANSWER_PROMPT,
    FINANCIAL_EXPERT_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_QUERY_REWRITE_PROMPT = """你是一个金融领域的查询分析和改写助手。

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


_ANSWER_PROMPT = """你是一个专业的金融知识助手。请严格基于以下检索到的金融文档内容直接回答用户问题。

回答要求：
1. 直接回答问题，不要出现"根据现有知识库内容"、"检索到了相关内容"等前置表述
2. 严格基于【检索内容】回答，不要编造未提供的信息
3. 在引用某段内容时，标注引用编号，如 [1]、[2]
4. 即使检索到的内容来自"展望"、"综述"等章节，只要包含相关信息即可引用
5. 如果检索内容不足以完全回答问题，直接回答已知部分即可，不要额外声明
6. 只有当检索内容完全不相关时，才说明"现有知识库中暂未检索到相关内容"
7. 回答使用清晰的中文，适当使用列表、分段提升可读性
8. 涉及具体数字、政策、操作步骤时，务必引用对应来源
9. 检索内容中不包含页码、章节号，严禁编造"第X页"、"第X章"等说法

【检索内容】
{context}

【用户问题】
{query}

请直接开始回答："""


def _get_answer_prompt() -> str:
    """根据开关返回对应的回答 Prompt 模板。"""
    if ENABLE_FINANCIAL_EXPERT_PROMPT:
        return FINANCIAL_EXPERT_ANSWER_PROMPT.format(
            system_prompt=FINANCIAL_EXPERT_SYSTEM_PROMPT,
            context="{context}",
            query="{query}",
        )
    return _ANSWER_PROMPT


# ---------------------------------------------------------------------------
# 上下文组装
# ---------------------------------------------------------------------------

def build_context(docs: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """将检索结果组装为 LLM 上下文文本，并返回引用来源列表。

    Returns:
        (context_text, sources) — context_text 含编号的文档片段，
        sources 为精简的来源元数据列表。
    """
    if not docs:
        return "（未检索到相关文档）", []

    chunks: list[str] = []
    sources: list[dict[str, Any]] = []

    for i, doc in enumerate(docs, start=1):
        content = (doc.get("content") or "")[:2000]
        title = doc.get("title", "金融文档")
        chunks.append(f"[{i}] 来源：{title}\n{content}")

        sources.append({
            "index": i,
            "title": title,
            "source": doc.get("source", ""),
            "chunk": doc.get("chunk"),
            "score": round(doc.get("score", 0.0), 4),
            "preview": (doc.get("content") or "")[:200],
        })

    context_text = "\n\n---\n\n".join(chunks)
    return context_text, sources


# ---------------------------------------------------------------------------
# 查询改写
# ---------------------------------------------------------------------------

def rewrite_query(query: str, history: list[dict[str, str]] | None = None) -> dict[str, Any]:
    """用 LLM 分析查询，返回改写查询、关键词和子查询。

    Returns:
        dict with keys:
        - 'rewritten': 改写后的主查询
        - 'keywords': 关键词列表
        - 'sub_queries': 子查询列表
    """
    if model is None:
        return {
            'rewritten': query,
            'keywords': [],
            'sub_queries': [query],
        }

    history_text = _format_history(history) if history else "（无）"
    try:
        prompt = ChatPromptTemplate.from_messages([
            ("human", _QUERY_REWRITE_PROMPT),
        ])
        chain = prompt | model | StrOutputParser()
        raw_output = chain.invoke({"query": query, "history": history_text})
        
        # 解析结构化输出
        result = _parse_rewrite_output(raw_output, query)
        logger.info(
            "查询改写：%s → 主查询=%s, 关键词=%s, 子查询=%s",
            query, result['rewritten'], result['keywords'], result['sub_queries']
        )
        return result
    except Exception as exc:
        logger.warning("查询改写失败，使用原查询：%s", exc)
        return {
            'rewritten': query,
            'keywords': _extract_keywords_simple(query),
            'sub_queries': [query],
        }


def _parse_rewrite_output(raw: str, original_query: str) -> dict[str, Any]:
    """解析 LLM 返回的改写输出。"""
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


def _extract_keywords_simple(query: str) -> list[str]:
    """简单关键词提取（无需 LLM）。"""
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
    """根据主查询和关键词生成子查询。"""
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


# ---------------------------------------------------------------------------
# 检索 + 重排
# ---------------------------------------------------------------------------

def reranking_enabled(requested: bool) -> bool:
    return ENABLE_RERANKER and requested


def retrieve_and_rerank(
    query: str,
    k: int = CHAT_TOP_K,
    rerank_top_n: int = CHAT_RERANK_TOP_K,
    use_rerank: bool = True,
) -> list[dict[str, Any]]:
    """混合检索 + 可选 BGE 重排序，返回 top_n 结果。

    所有检索逻辑统一由 :class:`KnowledgeBase.hybrid_search` 完成。
    """
    kb = get_knowledge_base()
    return kb.hybrid_search(
        query,
        k=k,
        expand_parents=True,
        use_rerank=reranking_enabled(use_rerank),
        rerank_top_n=rerank_top_n,
    )


# ---------------------------------------------------------------------------
# 流式问答
# ---------------------------------------------------------------------------

def _merge_docs(docs_a: list[dict[str, Any]], docs_b: list[dict[str, Any]], 
                top_k: int, boost_keywords: list[str] | None = None) -> list[dict[str, Any]]:
    """合并两组检索结果，基于 content 去重，支持关键词加权。

    加权策略：如果文档内容包含核心关键词，提升其排序分数。
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
    
    merged.sort(key=lambda d: d.get("score", 0.0), reverse=True)
    return merged[:top_k]


async def chat_stream(
    query: str,
    history: list[dict[str, str]] | None = None,
    *,
    use_rewrite: bool | None = None,
    use_rerank: bool = True,
    k: int = CHAT_TOP_K,
    rerank_top_n: int = CHAT_RERANK_TOP_K,
    strategy: str = "default",
    filters: dict[str, Any] | None = None,
) -> AsyncGenerator[dict[str, Any], None]:
    """流式问答异步生成器。

    检索策略（查询拆分增强）：
    1. 分析查询，提取关键词和生成子查询
    2. 使用主查询和所有子查询分别检索
    3. 合并所有结果，去重 + 关键词加权排序
    4. 取 top_k 结果
    
    按顺序 yield 以下事件：
    * ``{"type": "status", "stage": "rewriting"|"retrieving"|"reranking"|"generating"}``
    * ``{"type": "sources", "sources": [...]}``
    * ``{"type": "token", "content": "..."}``（多次）
    * ``{"type": "done", "rewritten_query": "...", "source_count": N}``
    """
    if model is None:
        yield {"type": "error", "message": "LLM 未配置（缺少 DEEPSEEK_API_KEY），请在 .env 中配置后重启服务"}
        return

    # LangGraph Agentic RAG 路由
    if ENABLE_LANGGRAPH:
        from finance_rag.src.application.langgraph_service import run_agent_stream as langgraph_stream
        async for event in langgraph_stream(query, history):
            yield event
        return

    should_rewrite = CHAT_ENABLE_QUERY_REWRITE if use_rewrite is None else use_rewrite

    # Step 1: 查询分析与改写
    query_info = {
        'rewritten': query,
        'keywords': [],
        'sub_queries': [query],
    }
    if should_rewrite:
        yield {"type": "status", "stage": "rewriting"}
        query_info = rewrite_query(query, history)

    rewritten = query_info['rewritten']
    keywords = query_info['keywords']
    sub_queries = query_info['sub_queries']
    
    # 构建所有需要检索的查询列表
    all_queries = [rewritten] + sub_queries
    all_queries = list(dict.fromkeys(all_queries))  # 去重，保持顺序
    logger.info("检索查询列表: %s, 关键词: %s", all_queries, keywords)

    # Step 2: 多查询并行检索 + 可选 BGE 重排序
    effective_use_rerank = reranking_enabled(use_rerank)
    yield {"type": "status", "stage": "retrieving"}
    if effective_use_rerank:
        yield {"type": "status", "stage": "reranking"}

    kb = get_knowledge_base()

    use_multi_stage = strategy == "optimized" or ENABLE_MULTI_STAGE_RETRIEVAL
    use_filters = filters if (strategy == "optimized" or ENABLE_METADATA_FILTER) else None

    def _do_search(search_query: str) -> list[dict[str, Any]]:
        """执行单次检索。"""
        try:
            if use_multi_stage:
                from finance_rag.src.infrastructure.vector_store.hybrid_retriever import AdvancedRetrievalPipeline
                retriever = kb._get_retriever()
                pipeline = AdvancedRetrievalPipeline(
                    retriever=retriever,
                    llm=model,
                )
                return pipeline.retrieve(
                    search_query,
                    k=k,
                    use_rerank=effective_use_rerank,
                    filters=use_filters,
                )
            else:
                return kb.hybrid_search(
                    search_query,
                    k=k,
                    expand_parents=True,
                    use_rerank=effective_use_rerank,
                    rerank_top_n=rerank_top_n,
                    filters=use_filters,
                    keywords=keywords,  # 传递关键词用于增强检索
                )
        except Exception as exc:
            logger.warning("检索失败: %s", exc)
            return []

    try:
        # 多查询检索，收集所有结果
        all_results: list[dict[str, Any]] = []
        for search_query in all_queries:
            logger.info("检索子查询: %s", search_query)
            docs = _do_search(search_query)
            all_results.extend(docs)
        
        # 合并结果，关键词加权
        docs = _merge_docs(all_results, [], k, boost_keywords=keywords)
        
        # 如果结果不理想，回退到简单策略
        if not docs or _check_low_quality_results(docs, keywords):
            logger.info("结果质量检查未通过，尝试简单稠密检索")
            simple_docs = kb.hybrid_search(
                rewritten,
                k=k,
                use_dense_only=True,
                expand_parents=True,
                use_rerank=effective_use_rerank,
                rerank_top_n=rerank_top_n,
                filters=use_filters,
            )
            if simple_docs:
                docs = _merge_docs(docs, simple_docs, k, boost_keywords=keywords)
    except Exception as exc:
        logger.exception("检索失败：%s", exc)
        yield {"type": "error", "message": f"检索失败：{exc}"}
        return

    if not docs:
        yield {"type": "sources", "sources": []}
        yield {"type": "token", "content": "未在知识库中检索到相关内容，请先上传文档或调整问题。"}
        yield {"type": "done", "rewritten_query": rewritten, "source_count": 0}
        return

    # Step 3: 组装上下文
    context_text, sources = build_context(docs)
    yield {"type": "sources", "sources": sources}

    # Step 4: 流式生成答案
    yield {"type": "status", "stage": "generating"}
    prompt = ChatPromptTemplate.from_messages([
        ("human", _get_answer_prompt()),
    ])
    chain = prompt | model | StrOutputParser()

    try:
        async for chunk in chain.astream({
            "context": context_text,
            "query": query,
        }):
            if chunk:
                yield {"type": "token", "content": chunk}
    except Exception as exc:
        logger.exception("流式生成失败：%s", exc)
        yield {"type": "token", "content": f"\n\n[生成出错：{exc}]"}

    yield {"type": "done", "rewritten_query": rewritten, "source_count": len(sources)}


def _check_low_quality_results(docs: list[dict[str, Any]], keywords: list[str]) -> bool:
    """检查检索结果质量，如果前3个结果都不包含任何关键词，则视为低质量。"""
    if not docs:
        return True
    
    check_count = min(3, len(docs))
    for doc in docs[:check_count]:
        content = doc.get("content", "")
        for kw in keywords:
            if kw and kw in content:
                return False  # 至少有一个包含关键词
    
    return True  # 前3个都不包含任何关键词


async def chat(
    query: str,
    history: list[dict[str, str]] | None = None,
    *,
    use_rewrite: bool | None = None,
    use_rerank: bool = True,
    k: int = CHAT_TOP_K,
    rerank_top_n: int = CHAT_RERANK_TOP_K,
    strategy: str = "default",
    filters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """非流式问答，返回完整答案 + 来源。

    Returns:
        ``{"answer", "sources", "rewritten_query"}``
    """
    answer_parts: list[str] = []
    sources: list[dict[str, Any]] = []
    rewritten_query = query

    async for event in chat_stream(
        query, history,
        use_rewrite=use_rewrite,
        use_rerank=use_rerank,
        k=k,
        rerank_top_n=rerank_top_n,
        strategy=strategy,
        filters=filters,
    ):
        etype = event.get("type")
        if etype == "token":
            answer_parts.append(event.get("content", ""))
        elif etype == "sources":
            sources = event.get("sources", [])
        elif etype == "done":
            rewritten_query = event.get("rewritten_query", query)

    result = {
        "answer": "".join(answer_parts),
        "sources": sources,
        "rewritten_query": rewritten_query,
    }

    # 引用验证（仅非流式模式）
    if ENABLE_CITATION_VALIDATION and sources:
        try:
            from finance_rag.src.application.citation_validator import CitationValidator

            kb = get_knowledge_base()
            embed_fn = kb._get_embeddings().embed_query
            validator = CitationValidator(embed_fn)
            validation = validator.validate(result["answer"], sources)
            result["citation_validation"] = validation
            logger.info(
                "引用验证完成：score=%.2f, issues=%d",
                validation["score"],
                len(validation["issues"]),
            )
        except Exception as exc:
            logger.warning("引用验证失败，跳过：%s", exc)

    return result
