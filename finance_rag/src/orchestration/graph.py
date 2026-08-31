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

import asyncio
import logging
from typing import Any, AsyncGenerator, Literal

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph

from finance_rag.src.core.config import (
    CHAT_RERANK_TOP_K,
    CHAT_TOP_K,
    ENABLE_HYDE,
    ENABLE_RERANKER,
    LANGGRAPH_MAX_REFLECT_ROUNDS,
    get_model,
    get_rewrite_model,
)
from finance_rag.src.agent.prompts.chat import ANSWER_PROMPT, REFLECTION_PROMPT
from finance_rag.src.agent.query_classifier import resolve_dynamic_k
from finance_rag.src.agent.state.models import AgentState
from finance_rag.src.agent.tools.retrieval import RetrievalTool

logger = logging.getLogger(__name__)

# 最大反思轮次（由 LANGGRAPH_MAX_REFLECT_ROUNDS 环境变量控制，默认 2）
MAX_REFLECTION_ROUNDS = LANGGRAPH_MAX_REFLECT_ROUNDS

# 检索质量阈值（rerank_score 均值低于此值触发反思）
RETRIEVAL_QUALITY_THRESHOLD = 0.3


# ---------------------------------------------------------------------------
# 查询改写：逻辑已移至 chat_service.rewrite_query（标准链路与 LangGraph 共用）
# ---------------------------------------------------------------------------


class _KnowledgeBaseRetriever:
    """将现有 KnowledgeBase 的 hybrid_search 适配为 RetrievalTool 检索器。"""

    def __init__(self, knowledge_base: Any) -> None:
        self._knowledge_base = knowledge_base

    def search(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
        return self._knowledge_base.hybrid_search(query, **kwargs)


def _rewrite_node(state: AgentState) -> AgentState:
    """节点 1: 查询改写（复用 chat_service.rewrite_query）。"""
    from finance_rag.src.services.chat_service import rewrite_query

    query_info = rewrite_query(state["query"], state.get("history"))
    return {
        **state,
        "rewritten_query": query_info["rewritten"],
        "keywords": query_info["keywords"],
        "sub_queries": query_info["sub_queries"],
        "reflection_round": state.get("reflection_round", 0),
    }


def _retrieve_node(state: AgentState) -> AgentState:
    """节点 2: 多查询跨库检索 + 去重 + 重排序（可选动态 K / HyDE 增强）

    同步函数，由 :func:`_retrieve_node_async` 在线程池中包装调用，
    避免 Milvus 检索与 BGE 重排序阻塞事件循环。
    """
    from finance_rag.src.services.chat_service import (
        hyde_retrieve,
        iter_active_kbs,
        merge_docs,
    )

    all_queries = [state["rewritten_query"]] + state["sub_queries"]
    all_queries = list(dict.fromkeys(all_queries))

    # 动态 K：按问题复杂度调整召回深度
    k, rerank_top_n = resolve_dynamic_k(
        state["query"], CHAT_TOP_K, CHAT_RERANK_TOP_K
    )

    kbs = list(iter_active_kbs())
    all_docs: list[dict[str, Any]] = []

    # 跨所有知识库联合检索，结果打上 collection 标记
    for q in all_queries:
        for coll_name, kb in kbs:
            try:
                result = RetrievalTool(_KnowledgeBaseRetriever(kb)).invoke(
                    q,
                    k=k,
                    expand_parents=True,
                    use_rerank=ENABLE_RERANKER,
                    rerank_top_n=rerank_top_n,
                    keywords=state["keywords"],
                )
                docs = result["evidence"]
                for d in docs:
                    d["collection"] = coll_name
                all_docs.extend(docs)
            except Exception as exc:
                logger.warning("LangGraph 检索失败（%s）: %s", coll_name, exc)

    # 去重
    merged = merge_docs(all_docs, [], k, boost_keywords=state["keywords"])

    # HyDE 检索增强（可选）
    if ENABLE_HYDE and get_model() is not None:
        try:
            hyde_docs = hyde_retrieve(
                state["query"], k, kbs, None, ENABLE_RERANKER, rerank_top_n
            )
            if hyde_docs:
                merged = merge_docs(
                    merged, hyde_docs, k, boost_keywords=state["keywords"]
                )
        except Exception as exc:
            logger.warning("LangGraph HyDE 检索失败：%s", exc)

    return {**state, "docs": merged}


async def _retrieve_node_async(state: AgentState) -> AgentState:
    """节点 2 的异步包装：整段同步检索（含重排序）在线程池执行，不阻塞事件循环。"""
    return await asyncio.to_thread(_retrieve_node, state)


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

    # 反思判断为轻量任务，优先使用轻量模型（qwen-turbo），未配置时回退主模型
    reflect_model = get_rewrite_model() or get_model()
    if reflect_model is None:
        return {**state, "reflection_round": 99}

    # 构建上下文摘要
    context_summary = "\n".join(
        (d.get("content", "") or "")[:200] for d in docs[:3]
    )

    try:
        prompt = ChatPromptTemplate.from_messages([("human", REFLECTION_PROMPT)])
        chain = prompt | reflect_model | StrOutputParser()
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
    if get_model() is None:
        return {**state, "answer": "LLM 未配置", "sources": []}

    if not state["docs"]:
        return {**state, "answer": "未检索到相关内容", "sources": []}

    from finance_rag.src.services.chat_service import build_context

    context_text, sources = build_context(state["docs"])

    prompt = ChatPromptTemplate.from_messages([("human", ANSWER_PROMPT)])
    chain = prompt | get_model() | StrOutputParser()

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
    graph.add_node("retrieve", _retrieve_node_async)
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


async def run_agent_stream(
    query: str,
    history: list[dict[str, str]] | None = None,
) -> AsyncGenerator[dict[str, Any], None]:
    """运行 LangGraph Agentic RAG 工作流并流式产出事件。

    产出的事件类型与 :func:`chat_stream` 兼容：
    ``status`` / ``sources`` / ``token`` / ``done`` / ``error``
    """
    app = get_langgraph_app()
    initial_state: AgentState = {
        "query": query,
        "history": history or [],
        "rewritten_query": query,
        "keywords": [],
        "sub_queries": [],
        "docs": [],
        "reflection_round": 0,
        "answer": "",
        "sources": [],
        "error": "",
    }

    # 记录已产出的阶段，避免重复
    yielded_stages: set[str] = set()

    try:
        async for event in app.astream(initial_state):
            for node_name, node_state in event.items():
                # 按节点产出阶段状态
                if node_name == "rewrite":
                    if "rewriting" not in yielded_stages:
                        yield {"type": "status", "stage": "rewriting"}
                        yielded_stages.add("rewriting")
                elif node_name == "retrieve":
                    if "retrieving" not in yielded_stages:
                        yield {"type": "status", "stage": "retrieving"}
                        yielded_stages.add("retrieving")
                elif node_name == "reflect":
                    if "reranking" not in yielded_stages:
                        yield {"type": "status", "stage": "reranking"}
                        yielded_stages.add("reranking")
                elif node_name == "generate":
                    if "generating" not in yielded_stages:
                        yield {"type": "status", "stage": "generating"}
                        yielded_stages.add("generating")

                final_state = node_state

        # 产出来源
        if final_state:
            sources = final_state.get("sources", [])
            yield {"type": "sources", "sources": sources}

            answer = final_state.get("answer", "")

            # 引用验证 + 拒答策略（此时尚未输出任何 token，可直接替换答案）
            from finance_rag.src.services.chat_service import rerank_scores
            from finance_rag.src.services.citation_validator import (
                REFUSAL_ANSWER,
                apply_refusal_policy,
                run_citation_validation,
            )
            validation = run_citation_validation(answer, sources)
            rejected, low_conf = apply_refusal_policy(
                validation, len(sources), rerank_scores=rerank_scores(sources)
            )
            if rejected:
                answer = REFUSAL_ANSWER

            # 流式产出答案（按字符块输出，模拟 token 流）
            if answer:
                chunk_size = 10
                for i in range(0, len(answer), chunk_size):
                    chunk = answer[i : i + chunk_size]
                    if chunk:
                        yield {"type": "token", "content": chunk}

            rewritten = final_state.get("rewritten_query", query)
            yield {
                "type": "done",
                "rewritten_query": rewritten,
                "source_count": len(sources),
                "citation_validation": validation,
                "answer_rejected": rejected,
                "low_confidence": low_conf,
            }
        else:
            yield {"type": "error", "message": "LangGraph 执行异常：未获取到最终状态"}

    except Exception as exc:
        logger.exception("LangGraph Agentic RAG 执行失败：%s", exc)
        yield {"type": "error", "message": f"LangGraph 执行失败：{exc}"}


def build_graph() -> Any:
    """规范入口别名，返回已编译的 LangGraph 应用。"""
    return build_langgraph_chain()


def run_agent(
    query: str,
    history: list[dict[str, str]] | None = None,
) -> Any:
    """同步兼容入口，返回 LangGraph 应用及初始状态的执行结果。"""
    app = get_langgraph_app()
    initial_state: AgentState = {
        "query": query,
        "history": history or [],
        "rewritten_query": query,
        "keywords": [],
        "sub_queries": [],
        "docs": [],
        "reflection_round": 0,
        "answer": "",
        "sources": [],
        "error": "",
    }
    return app.invoke(initial_state)


def run_langgraph_retrieval(
    query: str,
    history: list[dict[str, str]] | None = None,
) -> list[dict[str, Any]]:
    """运行 LangGraph 检索段（改写→检索→反思→重检索），返回最终 docs，不含生成。

    与线上图结构一致，供评估器复用；避免外部代码直接调用私有节点函数。
    """
    state: AgentState = {
        "query": query,
        "history": history or [],
        "rewritten_query": query,
        "keywords": [],
        "sub_queries": [],
        "docs": [],
        "reflection_round": 0,
        "answer": "",
        "sources": [],
        "error": "",
    }
    state = _rewrite_node(state)
    for _ in range(MAX_REFLECTION_ROUNDS + 1):
        state = _retrieve_node(state)
        state = _reflect_node(state)
        if state.get("reflection_round", 0) >= MAX_REFLECTION_ROUNDS:
            break
    return state.get("docs", [])
