"""Agent 检索工具边界。"""

from __future__ import annotations

from typing import Any


class RetrievalTool:
    """提供统一的检索输入和规范化证据输出。"""

    def __init__(self, retriever: Any | None = None) -> None:
        self._retriever = retriever

    def invoke(
        self,
        query: str,
        *,
        k: int = 5,
        filters: dict[str, Any] | None = None,
        keywords: list[str] | None = None,
        use_dense_only: bool = False,
        expand_parents: bool = True,
        use_rerank: bool = False,
        rerank_top_n: int = 3,
    ) -> dict[str, Any]:
        """执行检索并返回稳定的 evidence/metadata 结构。"""
        if not query or not query.strip():
            return {"query": query, "evidence": [], "metadata": {"count": 0}}

        retriever = self._retriever or _default_retriever()
        search = retriever.search if hasattr(retriever, "search") else retriever
        matches = search(
            query.strip(),
            k=k,
            filters=filters,
            keywords=keywords,
            use_dense_only=use_dense_only,
            expand_parents=expand_parents,
            use_rerank=use_rerank,
            rerank_top_n=rerank_top_n,
        )
        evidence = [_normalize_evidence(item, index) for index, item in enumerate(matches)]
        return {
            "query": query,
            "evidence": evidence,
            "metadata": {"count": len(evidence), "filters": filters or {}},
        }

    __call__ = invoke


class AgentRetrievalTool(RetrievalTool):
    """将 RAG 检索能力暴露给 Agent，隔离底层检索实现。"""

    name = "retrieve_finance_evidence"
    description = "检索金融知识库并返回规范化证据。"


def _normalize_evidence(item: dict[str, Any], index: int) -> dict[str, Any]:
    """统一检索结果为平铺字段，与 ``KnowledgeBase.hybrid_search`` 的返回契约一致。

    平铺字段保证下游 ``build_context`` / 来源展示能直接读取 source/title/
    category/chunk 等键，避免嵌套 ``metadata`` 造成字段丢失。
    """
    return {
        "id": item.get("id") or item.get("chunk_id") or f"evidence-{index}",
        "content": item.get("content") or "",
        "source": item.get("source", ""),
        "title": item.get("title", ""),
        "chunk": item.get("chunk"),
        "parent_id": item.get("parent_id", ""),
        "category": item.get("category", ""),
        "date": item.get("date", ""),
        "heading": item.get("heading", ""),
        "score": float(item.get("score") or 0.0),
        "rerank_score": item.get("rerank_score"),
    }


def _default_retriever() -> Any:
    """延迟创建默认知识库检索器，导入工具时不连接外部服务。"""
    from finance_rag.src.infrastructure.vector_store.milvus_kb import get_knowledge_base

    return get_knowledge_base()


def get_retrieval_tool(retriever: Any | None = None) -> RetrievalTool:
    """创建标准 Retrieval Tool，可注入测试替身或自定义检索器。"""
    return RetrievalTool(retriever)


def get_agent_tools(retriever: Any | None = None) -> list[AgentRetrievalTool]:
    """创建当前 Agent 可用工具集合。"""
    return [AgentRetrievalTool(retriever)]


__all__ = ["AgentRetrievalTool", "RetrievalTool", "get_agent_tools", "get_retrieval_tool"]
