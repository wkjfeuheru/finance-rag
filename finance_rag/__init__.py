"""金融 Agentic RAG 知识库检索包。"""

from importlib import import_module


_EXPORTS = {
    "KnowledgeBase": ("finance_rag.src.infrastructure.vector_store.milvus_kb", "KnowledgeBase"),
    "get_knowledge_base": ("finance_rag.src.infrastructure.vector_store.milvus_kb", "get_knowledge_base"),
    "DoclingHybridChunker": ("finance_rag.src.rag.ingestion.chunker", "DoclingHybridChunker"),
    "DoclingChunks": ("finance_rag.src.rag.ingestion.chunker", "DoclingChunks"),
    "BGEReranker": ("finance_rag.src.rag.retrieval.hybrid_retriever", "BGEReranker"),
    "HybridRetriever": ("finance_rag.src.rag.retrieval.hybrid_retriever", "HybridRetriever"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    """仅在访问兼容导出时加载重量级基础设施依赖。"""
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attribute_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value
