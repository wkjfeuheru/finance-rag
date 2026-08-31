"""测试集数据结构与加载器（无 ragas 依赖）。

本模块可被服务端 API 直接导入，无需安装 ragas。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# 测试集文件路径（默认使用自动生成的分层测试集）
_TEST_SET_FILE = Path(__file__).resolve().parent / "data" / "evaluation_qa_generated.md"


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class TestSetEntry:
    """测试集条目。

    Attributes:
        query: 测试查询文本。
        ground_truth: 标准答案（负样本为拒绝参考文案，可为空）。
        related_docs: 相关文档列表（逗号/分号分隔）。
        question_type: 问题类型 single_hop（单跳事实）/ multi_hop（跨文档多跳推理）/
            negative（负样本陷阱）。
        chunk_ids: 支撑该问题的证据 chunk id（Milvus 主键，可多个）。
        question_id: 唯一问题 ID（如 q-001）。
        difficulty: 难度等级（easy/medium/hard）。
        expected_answer_type: 预期答案类型（factual/list/comparison/procedural）。
        min_chunks: 回答该问题最少需要的检索块数。
        evidence_snippets: 关键证据片段（原文引用）。
        expected_behavior: 负样本/对抗层期望的系统行为描述。
    """

    query: str
    question_id: int = 0
    ground_truth: str = ""
    related_docs: str = ""
    related_sections: str = ""
    required_facts: tuple[str, ...] = ()
    test_type: str = "standard"
    unique_id: str = ""
    difficulty: str = "medium"
    expected_answer_type: str = "factual"
    min_chunks: int = 1
    evidence_snippets: tuple[str, ...] = ()
    expected_behavior: str = ""
    question_type: str = "single_hop"
    chunk_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "unique_id": self.unique_id,
            "query": self.query,
            "ground_truth": self.ground_truth,
            "related_docs": self.related_docs,
            "related_sections": self.related_sections,
            "required_facts": list(self.required_facts),
            "test_type": self.test_type,
            "difficulty": self.difficulty,
            "expected_answer_type": self.expected_answer_type,
            "min_chunks": self.min_chunks,
            "evidence_snippets": list(self.evidence_snippets),
            "expected_behavior": self.expected_behavior,
            "question_type": self.question_type,
            "chunk_ids": list(self.chunk_ids),
        }


# ---------------------------------------------------------------------------
# 测试集加载器
# ---------------------------------------------------------------------------

class TestSetLoader:
    """从 ``data/evaluation_qa.md`` 解析加载静态测试集。

    MD 文件结构：每个问题以 ``## N. 问题`` 开头，后接元数据，
    再以 ``### 标准答案`` 引出答案内容，以 ``---`` 分隔条目。
    """

    _QUESTION_PATTERN = re.compile(r"^##\s+(\d+)\.\s+(.+?)\s*$", re.MULTILINE)
    _RELATED_DOCS_PATTERN = re.compile(r"^-\s*相关文档[：:]\s*(.+?)\s*$", re.MULTILINE)
    _RELATED_SECTIONS_PATTERN = re.compile(r"^-\s*相关章节[：:]\s*(.+?)\s*$", re.MULTILINE)
    _REQUIRED_FACTS_PATTERN = re.compile(r"^-\s*必需事实[：:]\s*(.+?)\s*$", re.MULTILINE)
    _TEST_TYPE_PATTERN = re.compile(r"^-\s*测试类型[：:]\s*(.+?)\s*$", re.MULTILINE)
    _UNIQUE_ID_PATTERN = re.compile(r"^-\s*问题ID[：:]\s*(.+?)\s*$", re.MULTILINE)
    _DIFFICULTY_PATTERN = re.compile(r"^-\s*难度[：:]\s*(.+?)\s*$", re.MULTILINE)
    _ANSWER_TYPE_PATTERN = re.compile(r"^-\s*预期答案类型[：:]\s*(.+?)\s*$", re.MULTILINE)
    _MIN_CHUNKS_PATTERN = re.compile(r"^-\s*最小检索块数[：:]\s*(\d+)\s*$", re.MULTILINE)
    _EVIDENCE_PATTERN = re.compile(r"^-\s*\"(.+?)\"\s*$", re.MULTILINE)
    _EXPECTED_BEHAVIOR_PATTERN = re.compile(r"^-\s*期望行为[：:]\s*(.+?)\s*$", re.MULTILINE)
    _QUESTION_TYPE_PATTERN = re.compile(r"^-\s*问题类型[：:]\s*(single_hop|multi_hop|negative)\s*$", re.MULTILINE)
    _CHUNK_IDS_PATTERN = re.compile(r"^-\s*证据chunk[：:]\s*(.+?)\s*$", re.MULTILINE)

    def __init__(self, file_path: Path | None = None):
        self._file_path = file_path or _TEST_SET_FILE

    def load_test_set(self) -> list[TestSetEntry]:
        """从 ``data/evaluation_qa.md`` 解析加载测试集。"""
        if not self._file_path.exists():
            logger.warning("测试集文件不存在：%s", self._file_path)
            return []

        text = self._file_path.read_text(encoding="utf-8")
        entries = self._parse(text)
        self._validate(entries)
        return entries

    def has_ground_truth(self) -> bool:
        entries = self.load_test_set()
        return bool(entries) and all(e.ground_truth for e in entries)

    def _parse(self, text: str) -> list[TestSetEntry]:
        entries: list[TestSetEntry] = []
        matches = list(self._QUESTION_PATTERN.finditer(text))

        for i, match in enumerate(matches):
            question_id = int(match.group(1))
            query = match.group(2).strip()
            block_start = match.end()
            block_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            block = text[block_start:block_end]

            ground_truth = self._extract_ground_truth(block)
            related_docs_match = self._RELATED_DOCS_PATTERN.search(block)
            related_docs = related_docs_match.group(1).strip() if related_docs_match else ""
            sections_match = self._RELATED_SECTIONS_PATTERN.search(block)
            related_sections = sections_match.group(1).strip() if sections_match else ""
            facts_match = self._REQUIRED_FACTS_PATTERN.search(block)
            required_facts = tuple(
                fact.strip()
                for fact in re.split(r"[；;|]", facts_match.group(1))
                if fact.strip()
            ) if facts_match else ()
            type_match = self._TEST_TYPE_PATTERN.search(block)
            test_type = type_match.group(1).strip() if type_match else "standard"

            unique_id_match = self._UNIQUE_ID_PATTERN.search(block)
            unique_id = unique_id_match.group(1).strip() if unique_id_match else ""
            difficulty_match = self._DIFFICULTY_PATTERN.search(block)
            difficulty = difficulty_match.group(1).strip() if difficulty_match else "medium"
            answer_type_match = self._ANSWER_TYPE_PATTERN.search(block)
            expected_answer_type = answer_type_match.group(1).strip() if answer_type_match else "factual"
            min_chunks_match = self._MIN_CHUNKS_PATTERN.search(block)
            min_chunks = int(min_chunks_match.group(1)) if min_chunks_match else 1
            evidence_snippets = tuple(
                m.group(1).strip()
                for m in self._EVIDENCE_PATTERN.finditer(block)
            )
            behavior_match = self._EXPECTED_BEHAVIOR_PATTERN.search(block)
            expected_behavior = behavior_match.group(1).strip() if behavior_match else ""
            type_match = self._QUESTION_TYPE_PATTERN.search(block)
            question_type = type_match.group(1).strip() if type_match else "single_hop"
            chunk_match = self._CHUNK_IDS_PATTERN.search(block)
            chunk_ids = tuple(
                part.strip()
                for part in re.split(r"[,，;；|\s]+", chunk_match.group(1))
                if part.strip()
            ) if chunk_match else ()

            entries.append(TestSetEntry(
                question_id=question_id,
                query=query,
                ground_truth=ground_truth,
                related_docs=related_docs,
                related_sections=related_sections,
                required_facts=required_facts,
                test_type=test_type,
                unique_id=unique_id,
                difficulty=difficulty,
                expected_answer_type=expected_answer_type,
                min_chunks=min_chunks,
                evidence_snippets=evidence_snippets,
                expected_behavior=expected_behavior,
                question_type=question_type,
                chunk_ids=chunk_ids,
            ))

        logger.info("从 %s 解析到 %d 条测试查询", self._file_path, len(entries))
        return entries

    @staticmethod
    def _validate(entries: list[TestSetEntry]) -> None:
        errors: list[str] = []
        seen_queries: set[str] = set()
        for index, entry in enumerate(entries, 1):
            if not entry.query:
                errors.append(f"第 {index} 条缺少问题")
            elif entry.query in seen_queries:
                errors.append(f"第 {index} 条问题重复：{entry.query}")
            seen_queries.add(entry.query)

            if entry.question_type == "negative":
                if not entry.expected_behavior:
                    errors.append(f"第 {index} 条负样本缺少期望行为")
                continue

            if not entry.ground_truth:
                errors.append(f"第 {index} 条缺少标准答案")
            if entry.question_type not in ("single_hop", "multi_hop"):
                errors.append(f"第 {index} 条问题类型非法：{entry.question_type}")
            if not entry.related_docs:
                errors.append(f"第 {index} 条缺少相关文档")
            elif any(Path(item.strip()).name != item.strip()
                     for item in re.split(r"[,，;；|]", entry.related_docs) if item.strip()):
                errors.append(f"第 {index} 条相关文档路径无效：{entry.related_docs}")
            if entry.question_type == "single_hop" and len(entry.chunk_ids) < 1:
                errors.append(f"第 {index} 条单跳问题缺少证据 chunk id")
            if entry.question_type == "multi_hop" and len(entry.chunk_ids) < 2:
                errors.append(f"第 {index} 条多跳问题需至少 2 个证据 chunk id")
        if errors:
            raise ValueError("评估测试集校验失败：" + "；".join(errors))

    @staticmethod
    def _extract_ground_truth(block: str) -> str:
        marker = "### 标准答案"
        idx = block.find(marker)
        if idx == -1:
            return ""
        rest = block[idx + len(marker):]
        sep_idx = rest.find("\n---")
        if sep_idx != -1:
            rest = rest[:sep_idx]
        return rest.strip()
