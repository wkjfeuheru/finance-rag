"""LangGraph Agentic RAG 问答链。

基于 LangGraph StateGraph 实现"查询扩写→检索→反思→重检索"循环：
1. rewrite: 查询改写，生成关键词和子查询
2. retrieve: 多查询并行检索 + 去重 + 重排序
3. reflect: LLM 反思检索质量，决定是否重新检索
4. generate: 基于最终上下文生成答案

优势：非线性流程，检索质量不佳时会自动调整查询重试（最多 2 轮），
避免一次性检索失败导致答案质量差。
"""

from __future__ import annotations

import logging
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from config.settings import (
    CHAT_RERANK_TOP_K,
    CHAT_TOP_K,
    ENABLE_RERANKER,
    model,
)
from finance_rag.src.infrastructure.vector_store.milvus_kb import get_knowledge_base

logger = logging.getLogger(__name__)

# 最大反思轮次
MAX_REFLECTION_ROUNDS = 2

# 检索质量阈值（rerank_score 均值低于此值触发反思）
RETRIEVAL_QUALITY_THRESHOLD = 0.3

# 反思 Prompt
_REFLECTION_PROMPT = """你是一个金融检索质量评估助手。

用户原始问题：{query}
已检索到的内容摘要：{context_summary}
检索质量评分（rerank_score 均值）：{avg_score:.3f}

请判断检索结果是否足够回答用户问题：
- 如果检索内容相关且包含足够信息，回复 "SUFFICIENT"
- 如果检索内容不相关或信息不足，回复 "INSUFFICIENT" 并给出更好的检索关键词

请直接回复 "SUFFICIENT" 或 "INSUFFICIENT: <建议的新查询>"："""


class AgentState(TypedDict):
    """LangGraph Agent 状态"""
    query: str
    history: list[dict[str, str]]
    rewritten_query: str
    keywords: list[str]
    sub_queries: list[str]
    docs: list[dict[str, Any]]
    reflection_round: int
    answer: str
    sources: list[dict[str, Any]]
    error: str


def _rewrite_node(state: AgentState) -> AgentState:
    """节点 1: 查询改写"""
    from finance_rag.src.application.chat_service import rewrite_query

    query_info = rewrite_query(state["query"], state.get("history"))
    return {
        **state,
        "rewritten_query": query_info["rewritten"],
        "keywords": query_info["keywords"],
        "sub_queries": query_info["sub_queries"],
        "reflection_round": state.get("reflection_round", 0),
    }


def _retrieve_node(state: AgentState) -> AgentState:
    """节点 2: 多查询检索 + 去重 + 重排序"""
    kb = get_knowledge_base()
    all_queries = [state["rewritten_query"]] + state["sub_queries"]
    all_queries = list(dict.fromkeys(all_queries))
    all_docs: list[dict[str, Any]] = []

    for q in all_queries:
        try:
            docs = kb.hybrid_search(
                q,
                k=CHAT_TOP_K,
                expand_parents=True,
                use_rerank=ENABLE_RERANKER,
                rerank_top_n=CHAT_RERANK_TOP_K,
                keywords=state["keywords"],
            )
            all_docs.extend(docs)
        except Exception as exc:
            logger.warning("LangGraph 检索失败: %s", exc)

    # 去重
    from finance_rag.src.application.chat_service import _merge_docs
    merged = _merge_docs(all_docs, [], CHAT_TOP_K, boost_keywords=state["keywords"])

    return {**state, "docs": merged}


def _reflect_node(state: AgentState) -> AgentState:
    """节点 3: 反思检索质量"""
    docs = state["docs"]
    if not docs:
        return {**state, "reflection_round": state.get("reflection_round", 0) + 1}

    # 计算平均 rerank_score
    scores = [d.get("rerank_score", d.get("score", 0.0)) for d in docs]
    avg_score = sum(scores) / len(scores) if scores else 0

    if avg_score >= RETRIEVAL_QUALITY_THRESHOLD:
        logger.info("LangGraph 反思：检索质量达标 (avg=%.3f)，跳过反思", avg_score)
        return {**state, "reflection_round": 99}

    current_round = state.get("reflection_round", 0) + 1
    if current_round > MAX_REFLECTION_ROUNDS:
        logger.info("LangGraph 反思：已达最大轮次 %d，停止反思", MAX_REFLECTION_ROUNDS)
        return {**state, "reflection_round": 99}

    if model is None:
        return {**state, "reflection_round": 99}

    # 构建上下文摘要
    context_summary = "\n".join(
        (d.get("content", "") or "")[:200] for d in docs[:3]
    )

    try:
        prompt = ChatPromptTemplate.from_messages([("human", _REFLECTION_PROMPT)])
        chain = prompt | model | StrOutputParser()
        response = chain.invoke({
            "query": state["query"],
            "context_summary": context_summary,
            "avg_score": avg_score,
        })

        if response.strip().upper().startswith("SUFFICIENT"):
            logger.info("LangGraph 反思：LLM 判断检索充分")
            return {**state, "reflection_round": 99}
        else:
            # 提取新的检索查询
            new_query = response.replace("INSUFFICIENT:", "").replace("INSUFFICIENT", "").strip()
            if new_query:
                logger.info("LangGraph 反思：LLM 建议新查询 '%s'，重新检索", new_query)
                return {
                    **state,
                    "rewritten_query": new_query,
                    "reflection_round": current_round,
                }
    except Exception as exc:
        logger.warning("LangGraph 反思失败：%s", exc)

    return {**state, "reflection_round": 99}


def _should_retrieve(state: AgentState) -> Literal["retrieve", "generate"]:
    """条件边：是否重新检索"""
    if state.get("reflection_round", 0) < MAX_REFLECTION_ROUNDS:
        return "retrieve"
    return "generate"


def _generate_node(state: AgentState) -> AgentState:
    """节点 4: 生成答案"""
    if model is None:
        return {**state, "answer": "LLM 未配置", "sources": []}

    if not state["docs"]:
        return {**state, "answer": "未检索到相关内容", "sources": []}

    from finance_rag.src.application.chat_service import build_context, _get_answer_prompt

    context_text, sources = build_context(state["docs"])

    prompt = ChatPromptTemplate.from_messages([("human", _get_answer_prompt())])
    chain = prompt | model | StrOutputParser()

    try:
        answer = chain.invoke({"context": context_text, "query": state["query"]})
    except Exception as exc:
        logger.exception("LangGraph 生成失败：%s", exc)
        answer = f"生成答案时出错：{exc}"

    return {**state, "answer": answer, "sources": sources}


def build_langgraph_chain() -> StateGraph:
    """构建 LangGraph Agentic RAG 工作流。

    流程:
        rewrite → retrieve → reflect → generate
                    ↑              ↓
                    └─ INSUFFICIENT ─┘

    Returns:
        编译后的 StateGraph
    """
    graph = StateGraph(AgentState)

    graph.add_node("rewrite", _rewrite_node)
    graph.add_node("retrieve", _retrieve_node)
    graph.add_node("reflect", _reflect_node)
    graph.add_node("generate", _generate_node)

    graph.set_entry_point("rewrite")
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("retrieve", "reflect")
    graph.add_conditional_edges(
        "reflect",
        _should_retrieve,
        {"retrieve": "retrieve", "generate": "generate"},
    )
    graph.add_edge("generate", END)

    return graph.compile()


# 全局单例
_langgraph_app: Any = None


def get_langgraph_app():
    """获取 LangGraph 编译后的 app 单例。"""
    global _langgraph_app
    if _langgraph_app is None:
        _langgraph_app = build_langgraph_chain()
    return _langgraph_app
