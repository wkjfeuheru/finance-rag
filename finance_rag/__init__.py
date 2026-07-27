"""金融 Agentic RAG 知识库检索包。

公开接口：
* :class:`KnowledgeBase` — Docling Hybrid 切块 + 混合检索 + BGE 重排序的统一知识库
* :func:`get_knowledge_base` — 全局 KnowledgeBase 单例
* :class:`DoclingHybridChunker` — Docling 结构感知文档切块器
* :class:`HybridRetriever` — 稠密 + 稀疏混合检索器
* :class:`BGEReranker` — 本地 BGE 重排序器
* :class:`StrategyConfig` / :class:`StrategyEvaluator` — ragas 策略评估
"""

from .knowledge_base import KnowledgeBase, get_knowledge_base
from .chunking import DoclingChunks, DoclingHybridChunker
from .retrieve import BGEReranker, HybridRetriever

__all__ = [
    "KnowledgeBase",
    "get_knowledge_base",
    "DoclingHybridChunker",
    "DoclingChunks",
    "BGEReranker",
    "HybridRetriever",
]
