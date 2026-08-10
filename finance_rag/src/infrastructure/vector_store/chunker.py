"""基于层级父子结构的文档解析与切块."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import (
    CHILD_OVERLAP_TOKENS,
    DOCLING_CHUNK_MAX_TOKENS,
    ENABLE_SMART_CHUNKER,
    PARENT_MAX_TOKENS,
)


@dataclass
class ParentChunk:
    """父块（按 Markdown 标题切分）."""

    id: str
    heading: str
    content: str
    source: str
    title: str


@dataclass
class DoclingChunks:
    """层级父子切块结果."""

    chunks: list[Document] = field(default_factory=list)
    markdown: str | None = None
    parents: list[ParentChunk] = field(default_factory=list)


class HierarchicalChunker:
    """将文档解析为层级父子结构：按 Markdown 标题切分为父块，再递归切分子块."""

    def __init__(
        self,
        max_tokens: int = DOCLING_CHUNK_MAX_TOKENS,
        parent_max_tokens: int = PARENT_MAX_TOKENS,
        overlap: int = CHILD_OVERLAP_TOKENS,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens 必须大于 0")
        if parent_max_tokens <= 0:
            raise ValueError("parent_max_tokens 必须大于 0")
        self.max_tokens = max_tokens
        self.parent_max_tokens = parent_max_tokens
        self.overlap = overlap

    def parse_and_chunk(
        self,
        file_path: str | Path,
        *,
        source: str = "",
        title: str = "",
    ) -> DoclingChunks:
        """解析文件并返回层级父子切块结果."""
        path = Path(file_path)
        source = source or path.name
        title = title or Path(source).stem

        from finance_rag.src.infrastructure.parsing import get_parser

        result = get_parser().parse(path, source=source, title=title)
        markdown = result.markdown.strip()
        if not markdown:
            raise ValueError(f"解析结果为空：{source}")

        # PDF 等无标题结构的文档：注入 Markdown 标题以改善切块质量
        if not re.search(r"^#{1,6}\s+", markdown, re.MULTILINE):
            markdown = _inject_heading_structure(markdown)

        parents = self._split_into_parents(markdown, source, title)

        documents: list[Document] = []
        for parent in parents:
            child_texts = self._split_parent_into_children(parent.content)
            for child_text in child_texts:
                index = len(documents)
                chunk_id = hashlib.sha256(
                    f"{source}:child:{parent.id}:{index}:{child_text}".encode("utf-8")
                ).hexdigest()
                documents.append(
                    Document(
                        page_content=child_text,
                        metadata={
                            "id": chunk_id,
                            "source": source,
                            "title": title,
                            "chunk": index,
                            "parent_id": parent.id,
                        },
                    )
                )

        if not documents:
            raise ValueError(f"切块结果为空：{source}")

        return DoclingChunks(
            chunks=documents,
            markdown=markdown if path.suffix.lower() == ".pdf" else None,
            parents=parents,
        )

    def _split_into_parents(
        self, markdown: str, source: str, title: str
    ) -> list[ParentChunk]:
        """按 Markdown 标题切分为父块；无标题时按 parent_max_tokens 切分."""
        heading_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
        matches = list(heading_pattern.finditer(markdown))

        parents: list[ParentChunk] = []

        if not matches:
            # 无标题 fallback
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.parent_max_tokens * 4,
                chunk_overlap=0,
                separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
            )
            for idx, chunk in enumerate(splitter.split_text(markdown)):
                if not chunk.strip():
                    continue
                chunk_id = hashlib.sha256(
                    f"{source}:parent:{idx}:".encode("utf-8")
                ).hexdigest()
                parents.append(
                    ParentChunk(
                        id=chunk_id,
                        heading="",
                        content=chunk.strip(),
                        source=source,
                        title=title,
                    )
                )
            return parents

        # 处理第一个标题之前的正文
        if matches[0].start() > 0:
            preamble = markdown[: matches[0].start()].strip()
            if preamble:
                chunk_id = hashlib.sha256(
                    f"{source}:parent:0:".encode("utf-8")
                ).hexdigest()
                parents.append(
                    ParentChunk(
                        id=chunk_id,
                        heading="",
                        content=preamble,
                        source=source,
                        title=title,
                    )
                )

        for i, match in enumerate(matches):
            heading_level = match.group(1)
            heading_text = match.group(2).strip()
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
            content = markdown[start:end].strip()

            idx = len(parents)
            chunk_id = hashlib.sha256(
                f"{source}:parent:{idx}:{heading_text}".encode("utf-8")
            ).hexdigest()
            parents.append(
                ParentChunk(
                    id=chunk_id,
                    heading=heading_text,
                    content=content,
                    source=source,
                    title=title,
                )
            )

        return parents

    def _split_parent_into_children(self, content: str) -> list[str]:
        """将父块内容切分为子块；表格作为原子单元保护."""
        if not content.strip():
            return []

        units = _extract_atomic_units(content)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.max_tokens * 4,
            chunk_overlap=self.overlap * 4,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        )

        pieces: list[str] = []
        for kind, text in units:
            if kind == "table":
                pieces.append(text.strip())
            else:
                for piece in text_splitter.split_text(text):
                    if piece.strip():
                        pieces.append(piece.strip())

        max_chars = self.max_tokens * 4
        chunks: list[str] = []
        current = ""
        for piece in pieces:
            if not current:
                current = piece
            elif len(current) + len(piece) + 2 <= max_chars:
                current = current + "\n\n" + piece
            else:
                chunks.append(current)
                current = piece
        if current:
            chunks.append(current)

        return chunks


def _inject_heading_structure(markdown: str) -> str:
    """为无 Markdown 标题的中文文档注入标题结构，改善切块质量。

    识别以下模式并转换为 Markdown 标题：
    - ``第X章`` / ``第X节`` → ``##`` / ``###``
    - ``综　述`` / ``综述`` → ``##``
    - ``一、`` ``二、`` 等中文编号 → ``###``
    - ``（一）`` ``（二）`` 等 → ``####``
    """
    lines = markdown.split("\n")
    result: list[str] = []

    # 中文章节编号
    ch_num = "一二三四五六七八九十百千万"
    chapter_re = re.compile(rf"^第[{ch_num}]+章\s*(.*)")
    section_re = re.compile(rf"^第[{ch_num}]+节\s*(.*)")
    cn_item_re = re.compile(rf"^([{ch_num}]+)、\s*(.*)")
    paren_re = re.compile(rf"^（[{ch_num}]+）\s*(.*)")

    for line in lines:
        stripped = line.strip()

        # 第X章 → ##
        m = chapter_re.match(stripped)
        if m:
            heading = m.group(1).strip() or stripped
            result.append(f"## {stripped}")
            continue

        # 第X节 → ###
        m = section_re.match(stripped)
        if m:
            heading = m.group(1).strip() or stripped
            result.append(f"### {stripped}")
            continue

        # 综述 / 综　述 → ##
        if re.match(r"^综[　\s]*述\s*$", stripped):
            result.append(f"## {stripped}")
            continue

        # 一、二、 → ###
        m = cn_item_re.match(stripped)
        if m and len(stripped) < 80:
            result.append(f"### {stripped}")
            continue

        # （一）（二）→ ####
        m = paren_re.match(stripped)
        if m and len(stripped) < 80:
            result.append(f"#### {stripped}")
            continue

        result.append(line)

    return "\n".join(result)


def _extract_atomic_units(content: str) -> list[tuple[str, str]]:
    """提取 Markdown 表格作为原子单元，其余作为文本."""
    pattern = re.compile(
        r"((?:^[ \t]*\|.*\|[ \t]*(?:\r?\n|$))+)",
        re.MULTILINE,
    )
    units: list[tuple[str, str]] = []
    last_end = 0
    for m in pattern.finditer(content):
        if m.start() > last_end:
            text = content[last_end : m.start()]
            if text.strip():
                units.append(("text", text))
        units.append(("table", m.group()))
        last_end = m.end()
    if last_end < len(content):
        text = content[last_end:]
        if text.strip():
            units.append(("text", text))
    return units


class SmartChunker(HierarchicalChunker):
    """根据文档类型自动调整切块策略的智能切块器."""

    CHUNK_PROFILES = {
        "regulation": {"max_tokens": 800},
        "technical": {"max_tokens": 600},
        "financial_report": {"max_tokens": 1000},
        "tutorial": {"max_tokens": 400},
    }

    def parse_and_chunk(
        self,
        file_path: str | Path,
        *,
        source: str = "",
        title: str = "",
    ) -> DoclingChunks:
        """检测文档类型并使用对应的切块参数."""
        doc_type = self._detect_document_type(file_path)
        profile = self.CHUNK_PROFILES.get(
            doc_type, self.CHUNK_PROFILES["technical"]
        )

        original_max = self.max_tokens
        self.max_tokens = profile["max_tokens"]
        try:
            return super().parse_and_chunk(file_path, source=source, title=title)
        finally:
            self.max_tokens = original_max

    @staticmethod
    def _detect_document_type(file_path: str | Path) -> str:
        """基于文件名关键词判断文档类型."""
        name = Path(file_path).name.lower()
        if any(kw in name for kw in ["报告", "财报", "financial", "report"]):
            return "financial_report"
        if any(kw in name for kw in ["指南", "教程", "guide", "tutorial"]):
            return "tutorial"
        if any(kw in name for kw in ["法", "条例", "regulation", "law"]):
            return "regulation"
        return "technical"


# 向后兼容别名
DoclingHybridChunker = HierarchicalChunker


def get_chunker() -> HierarchicalChunker:
    """根据配置开关返回对应切块器实例."""
    if ENABLE_SMART_CHUNKER:
        return SmartChunker()
    return HierarchicalChunker()
