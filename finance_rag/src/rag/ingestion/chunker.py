"""基于层级父子结构的文档解析与切块."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from finance_rag.src.core.config import (
    COMPLIANCE_CATEGORIES,
    CHILD_OVERLAP_TOKENS,
    DOCLING_CHUNK_MAX_TOKENS,
    ENABLE_SEMANTIC_CHUNKER,
    PARENT_MAX_TOKENS,
    SEMANTIC_BREAK_THRESHOLD,
    SEMANTIC_DEVICE,
    SEMANTIC_EMBED_MODEL,
    SEMANTIC_MIN_TOKENS,
)


@dataclass
class ParentChunk:
    """父块（按完整 Markdown 章节或自然段切分）."""

    id: str
    heading: str
    content: str
    source: str
    title: str


@dataclass
class DoclingChunks:
    """层级父子切块结果.

    Attributes:
        chunks: 子块 Document 列表（每个子块带 page_content + metadata）。
        markdown: 仅 PDF 解析路径返回完整 markdown（扫描版 PDF 上传后复用）；
                  md/txt 直接切块路径为 None。
        parents: 父块列表（按完整 Markdown 章节或自然段切分）。
        images: 普通 PDF 解析时提取的嵌入图片元数据，形如
                ``{"temp_path", "temp_dir", "key", "page", "ext", "sha256", "caption"}``；
                扫描版 PDF / md / txt 路径为空列表。主进程负责落盘到存储后端。
    """

    chunks: list[Document] = field(default_factory=list)
    markdown: str | None = None
    parents: list[ParentChunk] = field(default_factory=list)
    images: list[dict] = field(default_factory=list)


class HierarchicalChunker:
    """将文档按完整章节/自然段切为父块，再递归切分子块."""

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
        category: str = "",
    ) -> DoclingChunks:
        """解析文件并返回层级父子切块结果。"""
        path = Path(file_path)
        source = source or path.name
        title = title or Path(source).stem

        from finance_rag.src.rag.ingestion import get_parser

        result = get_parser().parse(path, source=source, title=title)
        return self._build_chunks(
            result.markdown.strip(), source, title,
            is_pdf=path.suffix.lower() == ".pdf",
            images=result.images,
            category=category,
        )

    def chunk_markdown(
        self,
        markdown: str,
        *,
        source: str,
        title: str = "",
        is_pdf: bool = True,
        category: str = "",
    ) -> DoclingChunks:
        """从 markdown 文本直接切块（扫描版 PDF 经 LlamaParse 后走此路径）。

        与 parse_and_chunk_from_text 的区别：保留 markdown（is_pdf=True 时），
        便于后续上传解析后的 .md 文件到对象存储。
        """
        if not markdown or not markdown.strip():
            raise ValueError(f"解析结果为空：{source}")
        return self._build_chunks(markdown.strip(), source, title, is_pdf=is_pdf)

    def _build_chunks(
        self,
        markdown: str,
        source: str,
        title: str,
        *,
        is_pdf: bool = False,
        images: list[dict] | None = None,
        category: str = "",
    ) -> DoclingChunks:
        """内部方法：对 markdown 文本进行层级切块。"""
        if not markdown:
            raise ValueError(f"解析结果为空：{source}")

        # 内容清洗：规则过滤（页码/版权/导航等噪声）+ 文档内段落级去重（固定流程）
        markdown = self._clean_markdown(markdown)
        if not markdown or not markdown.strip():
            raise ValueError(f"清洗后内容为空：{source}")

        # PDF 等无标题结构的文档：注入 Markdown 标题以改善切块质量
        if not re.search(r"^#{1,6}\s+", markdown, re.MULTILINE):
            markdown = _inject_heading_structure(markdown)

        if category in COMPLIANCE_CATEGORIES:
            parents = _split_compliance_articles(markdown, source, title)
            if parents is None:
                parents = self._split_into_parents(markdown, source, title)
        else:
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
            markdown=markdown if is_pdf else None,
            parents=parents,
            images=list(images or []),
        )

    @staticmethod
    def _clean_markdown(markdown: str) -> str:
        """规则清洗 + 文档内段落级 SimHash 去重（固定流程，不可关闭）。"""
        from finance_rag.src.rag.ingestion.cleaner import (
            TextCleaner,
            deduplicate_paragraphs,
        )

        text = TextCleaner().clean(markdown)
        return deduplicate_paragraphs(text)

    def _split_into_parents(
        self, markdown: str, source: str, title: str
    ) -> list[ParentChunk]:
        """按完整 Markdown 章节或自然段切分父块。

        ``parent_max_tokens`` 保留为兼容参数，但不再拆分结构完整的父块；
        子块长度由 ``_split_parent_into_children`` 控制。
        """
        heading_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
        matches = list(heading_pattern.finditer(markdown))

        parents: list[ParentChunk] = []

        if not matches:
            # 无标题文档按空行分隔的自然段切分，不按长度拆散段落。
            paragraphs = re.split(r"\n\s*\n+", markdown)
            for idx, paragraph in enumerate(paragraphs):
                chunk = paragraph.strip()
                if not chunk:
                    continue
                chunk_id = hashlib.sha256(
                    f"{source}:parent:{idx}:".encode("utf-8")
                ).hexdigest()
                parents.append(
                    ParentChunk(
                        id=chunk_id,
                        heading="",
                        content=chunk,
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



def _split_compliance_articles(
    markdown: str, source: str, title: str
) -> list[ParentChunk] | None:
    """按独立行条款编号切分合规文档，条款内部层级保持完整。"""
    article_re = re.compile(
        r"^(?:\s*#{1,6}\s*)?(?P<label>第\s*(?:[一二三四五六七八九十百千万零〇两0-9０-９]+(?:\.[0-9０-９]+)*)\s*条)"
        r"(?:\s*[:：.、-]?\s*(?P<heading>.*))?$",
        re.MULTILINE,
    )
    dotted_re = re.compile(
        r"^\s*(?P<label>[0-9０-９]+(?:\.[0-9０-９]+)+)\s*[、.．:：-]?\s*(?P<heading>.+)?$",
        re.MULTILINE,
    )
    matches = list(article_re.finditer(markdown))
    if not matches:
        dotted = list(dotted_re.finditer(markdown))
        depths = [m.group("label").count(".") + 1 for m in dotted]
        if dotted and max(set(depths), key=depths.count) == 2:
            matches = [m for m in dotted if m.group("label").count(".") + 1 == 2]
    if not matches:
        return HierarchicalChunker._split_into_parents  # type: ignore[return-value]

    parents: list[ParentChunk] = []
    starts = [m.start() for m in matches]
    if starts[0] > 0 and markdown[:starts[0]].strip():
        preamble = markdown[:starts[0]].strip()
        parents.append(_make_parent(preamble, "", source, title, len(parents)))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
        content = markdown[match.start():end].strip()
        label = match.group("label").strip()
        heading = (match.groupdict().get("heading") or "").strip()
        parents.append(_make_parent(content, f"{label} {heading}".strip(), source, title, len(parents)))
    return parents


def _make_parent(
    content: str, heading: str, source: str, title: str, index: int
) -> ParentChunk:
    chunk_id = hashlib.sha256(
        f"{source}:parent:{index}:{heading}".encode("utf-8")
    ).hexdigest()
    return ParentChunk(
        id=chunk_id,
        heading=heading,
        content=content,
        source=source,
        title=title,
    )


def _inject_heading_structure(markdown: str) -> str:
    """为无 Markdown 标题的中文文档注入标题结构，改善切块质量。

    识别以下模式并转换为 Markdown 标题：
    - ``第X章`` / ``第X节`` → ``##`` / ``###``
    - ``第X条``（法规条款，支持中文/阿拉伯数字编号）→ ``####``
    - ``综　述`` / ``综述`` → ``##``
    - ``一、`` ``二、`` 等中文编号 → ``###``
    - ``（一）`` ``（二）`` 等 → ``####``
    """
    lines = markdown.split("\n")
    result: list[str] = []

    # 中文章节编号（法规条文同时支持阿拉伯数字编号，如"第1条"）
    ch_num = "一二三四五六七八九十百千万"
    chapter_re = re.compile(rf"^第[{ch_num}]+章\s*(.*)")
    section_re = re.compile(rf"^第[{ch_num}]+节\s*(.*)")
    article_re = re.compile(rf"^第[{ch_num}0-9]+条\s*(.*)")
    cn_item_re = re.compile(rf"^([{ch_num}]+)、\s*(.*)")
    paren_re = re.compile(rf"^（[{ch_num}]+）\s*(.*)")

    for line in lines:
        stripped = line.strip()

        # 第X章 → ##
        m = chapter_re.match(stripped)
        if m:
            result.append(f"## {stripped}")
            continue

        # 第X节 → ###
        m = section_re.match(stripped)
        if m:
            result.append(f"### {stripped}")
            continue

        # 第X条 → ####（法规条款，合规审查的检索/引用单位）
        m = article_re.match(stripped)
        if m:
            result.append(f"#### {stripped}")
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


# 向后兼容别名
DoclingHybridChunker = HierarchicalChunker


_SENTENCE_RE = re.compile(r"[^。！？；\n]+[。！？；]?|\n+")


def _split_sentences(text: str) -> list[str]:
    """将文本切分为句子列表（保留句末标点）。"""
    sentences = [m.group(0) for m in _SENTENCE_RE.finditer(text)]
    return [s for s in sentences if s.strip()]


class SemanticChunker(HierarchicalChunker):
    """语义分块：用本地句子嵌入检测语义断点，按主题边界切分子块。

    * 相邻句子余弦相似度低于阈值且当前段已积累足够长度时切分；
    * 父块构建逻辑与层级切块一致（子块仍归属标题父块）；
    * 表格沿用原子单元保护，不参与语义切分；
    * 模型延迟加载，仅在启用该切块器时初始化。
    """

    def __init__(
        self,
        max_tokens: int = DOCLING_CHUNK_MAX_TOKENS,
        parent_max_tokens: int = PARENT_MAX_TOKENS,
        overlap: int = CHILD_OVERLAP_TOKENS,
        *,
        break_threshold: float = SEMANTIC_BREAK_THRESHOLD,
        min_tokens: int = SEMANTIC_MIN_TOKENS,
        embed_model: str = SEMANTIC_EMBED_MODEL,
        device: str = SEMANTIC_DEVICE,
    ) -> None:
        super().__init__(
            max_tokens=max_tokens,
            parent_max_tokens=parent_max_tokens,
            overlap=overlap,
        )
        self.break_threshold = break_threshold
        self.min_tokens = min_tokens
        self.embed_model = embed_model
        self.device = device
        self._model = None

    def _get_model(self):
        """延迟加载 sentence-transformer 模型（进程内复用）。"""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.embed_model, device=self.device)
        return self._model

    def _split_parent_into_children(self, content: str) -> list[str]:
        """文本单元走语义断点切分，表格保持原子。"""
        if not content.strip():
            return []

        units = _extract_atomic_units(content)
        pieces: list[str] = []
        for kind, text in units:
            if kind == "table":
                pieces.append(text.strip())
            else:
                pieces.extend(self._semantic_split(text))

        # 与小尺寸父子切块相同的合并逻辑：小片段合并，超长截断保护
        max_chars = self.max_tokens * 4
        chunks: list[str] = []
        current = ""
        for piece in pieces:
            if not piece.strip():
                continue
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

    def _semantic_split(self, text: str) -> list[str]:
        """基于相邻句嵌入相似度切分语义段。"""
        sentences = _split_sentences(text)
        if len(sentences) <= 1:
            return [text.strip()] if text.strip() else []

        import numpy as np

        model = self._get_model()
        embeddings = model.encode(
            sentences, normalize_embeddings=True, batch_size=16
        )

        segments: list[str] = []
        current = [sentences[0]]
        current_len = len(sentences[0])
        for i in range(1, len(sentences)):
            similarity = float(np.dot(embeddings[i - 1], embeddings[i]))
            if similarity < self.break_threshold and current_len >= self.min_tokens:
                segments.append("".join(current).strip())
                current = [sentences[i]]
                current_len = len(sentences[i])
            else:
                current.append(sentences[i])
                current_len += len(sentences[i])
        segments.append("".join(current).strip())

        return [s for s in segments if s]


def get_chunker() -> HierarchicalChunker:
    """根据配置开关返回对应切块器实例（语义 > 层级）。"""
    if ENABLE_SEMANTIC_CHUNKER:
        return SemanticChunker()
    return HierarchicalChunker()
