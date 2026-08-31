"""测试集加载器与生成器解析逻辑单元测试。"""

from __future__ import annotations

from finance_rag.src.eval.test_set import TestSetEntry, TestSetLoader
from finance_rag.src.eval.testset_generator import _parse_llm_qa, filter_entries_by_kb


class TestParseLlmQa:
    def test_valid(self):
        raw = "## 问题\n什么是适当性管理？\n\n### 标准答案\n适当性管理是指对客户进行风险等级评估的完整流程。"
        query, answer = _parse_llm_qa(raw)
        assert query == "什么是适当性管理？"
        assert answer == "适当性管理是指对客户进行风险等级评估的完整流程。"

    def test_missing_answer_returns_none(self):
        assert _parse_llm_qa("## 问题\n什么是X？") is None

    def test_short_query_returns_none(self):
        assert _parse_llm_qa("## 问题\nX\n\n### 标准答案\n很长的标准答案内容") is None

    def test_garbage_returns_none(self):
        assert _parse_llm_qa("random text without markers") is None


class TestTestSetLoader:
    def test_parse_single_entry(self, tmp_path):
        md = tmp_path / "qa.md"
        md.write_text(
            "## 1. 问题一？\n\n"
            "- 问题类型：single_hop\n"
            "- 相关文档：doc.pdf\n"
            "- 证据chunk：chunk-001\n\n"
            "### 标准答案\n\n答案一\n\n---\n",
            encoding="utf-8",
        )
        entries = TestSetLoader(md).load_test_set()
        assert len(entries) == 1
        assert entries[0].query == "问题一？"
        assert entries[0].ground_truth == "答案一"
        assert entries[0].related_docs == "doc.pdf"
        assert entries[0].question_type == "single_hop"
        assert entries[0].chunk_ids == ("chunk-001",)
        assert isinstance(entries[0], TestSetEntry)

    def test_parse_multi_hop_with_two_chunks(self, tmp_path):
        md = tmp_path / "qa.md"
        md.write_text(
            "## 1. 综合题？\n\n"
            "- 问题类型：multi_hop\n"
            "- 相关文档：a.pdf, b.pdf\n"
            "- 证据chunk：c1, c2\n\n"
            "### 标准答案\n\n需要综合两处的答案\n\n---\n",
            encoding="utf-8",
        )
        entries = TestSetLoader(md).load_test_set()
        assert entries[0].question_type == "multi_hop"
        assert entries[0].chunk_ids == ("c1", "c2")

    def test_negative_allows_empty_ground_truth(self, tmp_path):
        md = tmp_path / "qa.md"
        md.write_text(
            "## 1. 陷阱问题？\n\n"
            "- 问题类型：negative\n"
            "- 期望行为：应拒绝回答或指出知识库无相关信息\n\n"
            "---\n",
            encoding="utf-8",
        )
        entries = TestSetLoader(md).load_test_set()
        assert len(entries) == 1
        assert entries[0].question_type == "negative"
        assert entries[0].ground_truth == ""

    def test_multi_hop_with_one_chunk_raises(self, tmp_path):
        md = tmp_path / "qa.md"
        md.write_text(
            "## 1. 问题一？\n\n"
            "- 问题类型：multi_hop\n"
            "- 相关文档：a.pdf, b.pdf\n"
            "- 证据chunk：c1\n\n"
            "### 标准答案\n\n答案\n\n---\n",
            encoding="utf-8",
        )
        import pytest

        with pytest.raises(ValueError, match="证据 chunk"):
            TestSetLoader(md).load_test_set()

    def test_missing_file_returns_empty(self, tmp_path):
        entries = TestSetLoader(tmp_path / "nope.md").load_test_set()
        assert entries == []

    def test_missing_ground_truth_raises(self, tmp_path):
        md = tmp_path / "qa.md"
        md.write_text(
            "## 1. 问题一？\n\n"
            "- 问题类型：single_hop\n"
            "- 相关文档：doc.pdf\n"
            "- 证据chunk：c1\n\n"
            "---\n",
            encoding="utf-8",
        )
        import pytest

        with pytest.raises(ValueError):
            TestSetLoader(md).load_test_set()


class TestFilterEntriesByKb:
    def _fake_kb(self, sources):
        return type("KB", (), {"list_documents": lambda self: [
            {"source": s} for s in sources
        ]})()

    def test_keeps_entries_with_existing_docs(self):
        entries = [type("E", (), {"related_docs": "doc.pdf"})()]
        kb = self._fake_kb(["doc.pdf"])
        kept, removed = filter_entries_by_kb(entries, kb)
        assert len(kept) == 1
        assert removed == []

    def test_removes_entries_with_missing_docs(self):
        entries = [type("E", (), {"related_docs": "missing.pdf"})()]
        kb = self._fake_kb(["doc.pdf"])
        kept, removed = filter_entries_by_kb(entries, kb)
        assert kept == []
        assert len(removed) == 1

    def test_allow_missing_keeps_all(self):
        entries = [type("E", (), {"related_docs": "missing.pdf"})()]
        kb = self._fake_kb(["doc.pdf"])
        kept, removed = filter_entries_by_kb(entries, kb, allow_missing=True)
        assert len(kept) == 1
        assert removed == []
