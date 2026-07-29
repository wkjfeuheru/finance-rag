from __future__ import annotations

import unittest
from unittest.mock import patch

import finance_rag.chat as chat_module


class EffectiveRerankerTests(unittest.TestCase):
    def test_global_disable_cannot_be_overridden_by_request(self):
        with patch.object(chat_module, "ENABLE_RERANKER", False):
            self.assertFalse(chat_module.reranking_enabled(True))

    def test_global_enable_still_allows_request_to_disable(self):
        with patch.object(chat_module, "ENABLE_RERANKER", True):
            self.assertFalse(chat_module.reranking_enabled(False))

    def test_both_global_and_request_must_enable_reranking(self):
        with patch.object(chat_module, "ENABLE_RERANKER", True):
            self.assertTrue(chat_module.reranking_enabled(True))


class FakeKnowledgeBase:
    def __init__(self):
        self.search_kwargs = None

    def hybrid_search(self, query, **kwargs):
        self.search_kwargs = kwargs
        return []


class ChatStreamRerankerTests(unittest.IsolatedAsyncioTestCase):
    async def collect_events(self, *, globally_enabled, requested):
        kb = FakeKnowledgeBase()
        with (
            patch.object(chat_module, "ENABLE_RERANKER", globally_enabled),
            patch.object(chat_module, "model", object()),
            patch.object(chat_module, "get_knowledge_base", return_value=kb),
        ):
            events = [
                event
                async for event in chat_module.chat_stream(
                    "query",
                    use_rewrite=False,
                    use_rerank=requested,
                )
            ]
        return kb, events

    async def test_global_disable_suppresses_status_and_normal_retrieval_rerank(self):
        kb, events = await self.collect_events(
            globally_enabled=False, requested=True
        )
        self.assertNotIn("reranking", [event.get("stage") for event in events])
        self.assertFalse(kb.search_kwargs["use_rerank"])

    async def test_both_enabled_emit_status_and_enable_normal_retrieval_rerank(self):
        kb, events = await self.collect_events(
            globally_enabled=True, requested=True
        )
        self.assertIn("reranking", [event.get("stage") for event in events])
        self.assertTrue(kb.search_kwargs["use_rerank"])

    async def test_request_disable_suppresses_status_and_normal_retrieval_rerank(self):
        kb, events = await self.collect_events(
            globally_enabled=True, requested=False
        )
        self.assertNotIn("reranking", [event.get("stage") for event in events])
        self.assertFalse(kb.search_kwargs["use_rerank"])


class FakePipeline:
    last_kwargs = None

    def __init__(self, **kwargs):
        pass

    def retrieve(self, query, **kwargs):
        type(self).last_kwargs = kwargs
        return []


class MultiStageRerankerTests(unittest.IsolatedAsyncioTestCase):
    async def test_global_disable_reaches_multi_stage_pipeline(self):
        kb = FakeKnowledgeBase()
        kb._get_retriever = lambda: object()
        with (
            patch.object(chat_module, "ENABLE_RERANKER", False),
            patch.object(chat_module, "model", object()),
            patch.object(chat_module, "get_knowledge_base", return_value=kb),
            patch("finance_rag.retrieve.AdvancedRetrievalPipeline", FakePipeline),
        ):
            events = [
                event
                async for event in chat_module.chat_stream(
                    "query",
                    use_rewrite=False,
                    use_rerank=True,
                    strategy="optimized",
                )
            ]
        self.assertFalse(FakePipeline.last_kwargs["use_rerank"])
        self.assertNotIn("reranking", [event.get("stage") for event in events])
