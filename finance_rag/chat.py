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

from .config import (
    CHAT_ENABLE_QUERY_REWRITE,
    CHAT_RERANK_TOP_K,
    CHAT_TOP_K,
    model,
)
from .knowledge_base import get_knowledge_base

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_QUERY_REWRITE_PROMPT = """你是一个金融领域的查询改写助手。请将用户的口语化问题改写为更适合检索的简洁查询。

改写要求：
1. 保留核心意图，去除冗余
2. 补充隐含的金融术语
3. 输出一句简洁的检索查询，不要解释

用户问题：{query}
历史对话：{history}

请直接输出改写后的查询（一句话，不要加引号或前缀）："""


_ANSWER_PROMPT = """你是一个专业的金融知识助手。请基于以下检索到的金融文档内容片段回答用户问题。

回答要求：
1. 严格基于【检索内容】回答，不要编造未提供的信息，说明是依据哪个文档内容回答的
2. 在引用某段内容时，标注引用编号，如 [1]、[2]
3. 如果检索内容不足以回答问题，请明确说明"根据现有知识库内容，暂无法完整回答该问题"
4. 回答使用清晰的中文，适当使用列表、分段提升可读性
5. 涉及具体数字、政策、操作步骤时，务必引用对应来源

【检索内容】
{context}

【用户问题】
{query}

请开始回答："""


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

def rewrite_query(query: str, history: list[dict[str, str]] | None = None) -> str:
    """用 LLM 改写查询为检索友好形式。

    若 LLM 不可用或改写失败，返回原查询。
    """
    if model is None:
        return query

    history_text = _format_history(history) if history else "（无）"
    try:
        prompt = ChatPromptTemplate.from_messages([
            ("human", _QUERY_REWRITE_PROMPT),
        ])
        chain = prompt | model | StrOutputParser()
        rewritten = chain.invoke({"query": query, "history": history_text})
        rewritten = rewritten.strip().strip('"').strip("'")
        if rewritten and rewritten != query:
            logger.info("查询改写：%s → %s", query, rewritten)
            return rewritten
    except Exception as exc:
        logger.warning("查询改写失败，使用原查询：%s", exc)
    return query


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
        use_rerank=use_rerank,
        rerank_top_n=rerank_top_n,
    )


# ---------------------------------------------------------------------------
# 流式问答
# ---------------------------------------------------------------------------

async def chat_stream(
    query: str,
    history: list[dict[str, str]] | None = None,
    *,
    use_rewrite: bool | None = None,
    use_rerank: bool = True,
    k: int = CHAT_TOP_K,
    rerank_top_n: int = CHAT_RERANK_TOP_K,
) -> AsyncGenerator[dict[str, Any], None]:
    """流式问答异步生成器。

    按顺序 yield 以下事件：
    * ``{"type": "status", "stage": "rewriting"|"retrieving"|"reranking"|"generating"}``
    * ``{"type": "sources", "sources": [...]}``
    * ``{"type": "token", "content": "..."}``（多次）
    * ``{"type": "done", "rewritten_query": "...", "source_count": N}``
    """
    if model is None:
        yield {"type": "error", "message": "LLM 未配置（缺少 DEEPSEEK_API_KEY），请在 .env 中配置后重启服务"}
        return

    should_rewrite = CHAT_ENABLE_QUERY_REWRITE if use_rewrite is None else use_rewrite

    # Step 1: 查询改写
    rewritten = query
    if should_rewrite:
        yield {"type": "status", "stage": "rewriting"}
        rewritten = rewrite_query(query, history)

    # Step 2: 混合检索 + 可选 BGE 重排序（统一由 KnowledgeBase.hybrid_search 完成）
    yield {"type": "status", "stage": "retrieving"}
    if use_rerank:
        # 保留 SSE ``reranking`` 状态事件
        yield {"type": "status", "stage": "reranking"}

    kb = get_knowledge_base()
    try:
        docs = kb.hybrid_search(
            rewritten,
            k=k,
            expand_parents=True,
            use_rerank=use_rerank,
            rerank_top_n=rerank_top_n,
        )
    except Exception as exc:
        logger.exception("混合检索失败：%s", exc)
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
        ("human", _ANSWER_PROMPT),
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


async def chat(
    query: str,
    history: list[dict[str, str]] | None = None,
    *,
    use_rewrite: bool | None = None,
    use_rerank: bool = True,
    k: int = CHAT_TOP_K,
    rerank_top_n: int = CHAT_RERANK_TOP_K,
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
    ):
        etype = event.get("type")
        if etype == "token":
            answer_parts.append(event.get("content", ""))
        elif etype == "sources":
            sources = event.get("sources", [])
        elif etype == "done":
            rewritten_query = event.get("rewritten_query", query)

    return {
        "answer": "".join(answer_parts),
        "sources": sources,
        "rewritten_query": rewritten_query,
    }
