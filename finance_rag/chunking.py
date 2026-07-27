"""基于 Docling 的文档解析与 HybridChunker 切块。"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.documents import Document

from .config import DOCLING_CHUNK_MAX_TOKENS, DOCLING_CHUNK_TOKENIZER


@dataclass
class DoclingChunks:
    """可直接嵌入和检索的单层 Docling 切块结果。"""

    chunks: list[Document] = field(default_factory=list)
    markdown: str | None = None


class DoclingHybridChunker:
    """将受支持的文档转换为 DoclingDocument 并进行结构感知切块。"""

    def __init__(
        self,
        tokenizer_name: str = DOCLING_CHUNK_TOKENIZER,
        max_tokens: int = DOCLING_CHUNK_MAX_TOKENS,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError("DOCLING_CHUNK_MAX_TOKENS 必须大于 0")
        self.tokenizer_name = tokenizer_name
        self.max_tokens = max_tokens
        self._converter = None
        self._chunker = None

    def parse_and_chunk(
        self,
        file_path: str | Path,
        *,
        source: str = "",
        title: str = "",
    ) -> DoclingChunks:
        """解析文件，并返回 HybridChunker 生成的单层检索块。"""
        path = Path(file_path)
        source = source or path.name
        title = title or Path(source).stem

        try:
            converter = self._get_converter()
            if Path(source).suffix.lower() == ".txt":
                from docling.datamodel.base_models import InputFormat

                text = path.read_text(encoding="utf-8", errors="replace")
                result = converter.convert_string(text, format=InputFormat.MD, name=source)
            else:
                result = converter.convert(path)
            dl_doc = result.document
        except Exception as exc:
            raise ValueError(f"Docling 解析失败 {source}：{exc}") from exc

        markdown = dl_doc.export_to_markdown().strip()
        if not markdown:
            raise ValueError(f"Docling 解析结果为空：{source}")

        try:
            chunker = self._get_chunker()
            raw_chunks = list(chunker.chunk(dl_doc=dl_doc))
            documents: list[Document] = []
            for raw_chunk in raw_chunks:
                content = chunker.contextualize(raw_chunk).strip()
                if not content:
                    continue
                index = len(documents)
                chunk_id = hashlib.sha256(
                    f"{source}:docling:{index}:{content}".encode("utf-8")
                ).hexdigest()
                documents.append(Document(
                    page_content=content,
                    metadata={
                        "id": chunk_id,
                        "source": source,
                        "title": title,
                        "chunk": index,
                        "parent_id": "",
                    },
                ))
        except Exception as exc:
            raise ValueError(
                "Docling HybridChunker 切块失败 "
                f"{source}（tokenizer={self.tokenizer_name}）：{exc}"
            ) from exc

        if not documents:
            raise ValueError(f"Docling 切块结果为空：{source}")

        return DoclingChunks(
            chunks=documents,
            markdown=markdown if Path(source).suffix.lower() == ".pdf" else None,
        )

    def _get_converter(self):
        if self._converter is None:
            from docling.document_converter import DocumentConverter

            self._converter = DocumentConverter()
        return self._converter

    def _get_chunker(self):
        if self._chunker is None:
            try:
                from docling.chunking import HybridChunker
                from docling_core.transforms.chunker.tokenizer.huggingface import (
                    HuggingFaceTokenizer,
                )
                from transformers import AutoTokenizer

                tokenizer = HuggingFaceTokenizer(
                    tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_name),
                    max_tokens=self.max_tokens,
                )
                self._chunker = HybridChunker(tokenizer=tokenizer, merge_peers=True)
            except Exception as exc:
                raise RuntimeError(
                    "无法初始化 Docling tokenizer "
                    f"{self.tokenizer_name}，请确认模型可下载或已缓存：{exc}"
                ) from exc
        return self._chunker