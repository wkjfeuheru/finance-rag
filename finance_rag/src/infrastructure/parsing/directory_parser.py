"""基于 LlamaIndex 的文档解析实现。

提供 ``SimpleDirectoryReader`` 通用解析与 ``LlamaParse`` 扫描版 PDF 解析。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llama_index.core.schema import Document

logger = logging.getLogger(__name__)


def _documents_to_markdown(documents: list[Document]) -> str:
    """将 LlamaIndex Document 列表拼接为 Markdown 文本。"""
    parts = []
    for doc in documents:
        text = getattr(doc, "text", "") or ""
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def parse_with_simple_reader(file_path: str | Path) -> str:
    """使用 ``SimpleDirectoryReader`` 解析文件，返回 Markdown 文本。

    Parameters
    ----------
    file_path :
        待解析文件路径。

    Returns
    -------
    str
        解析后的 Markdown 文本。
    """
    from llama_index.core import SimpleDirectoryReader

    path = Path(file_path)
    reader = SimpleDirectoryReader(input_files=[str(path)])
    documents = reader.load_data()
    return _documents_to_markdown(documents)


def parse_with_llama_parse(file_path: str | Path, api_key: str) -> str:
    """使用 ``LlamaParse`` 解析扫描版 PDF，返回 Markdown 文本。

    Parameters
    ----------
    file_path :
        待解析 PDF 文件路径。
    api_key :
        LlamaCloud API Key。

    Returns
    -------
    str
        解析后的 Markdown 文本。
    """
    from llama_parse import LlamaParse

    path = Path(file_path)
    parser = LlamaParse(api_key=api_key, result_type="markdown")
    documents = parser.load_data(str(path))
    return _documents_to_markdown(documents)
