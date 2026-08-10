"""文档解析模块。提供统一的 get_parser() 入口。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ParseResult:
    """解析结果。"""
    markdown: str


class Parser:
    """统一文档解析器，根据文件扩展名选择解析策略。"""

    def parse(self, path: str | Path, *, source: str = "", title: str = "") -> ParseResult:
        """解析文件并返回 Markdown 文本。

        对于 .md / .txt 文件直接读取原文件；
        对于 .pdf 等需要 LlamaIndex 解析的格式，延迟导入 directory_parser。
        """
        path = Path(path)
        ext = path.suffix.lower()

        if ext in (".md", ".txt", ".markdown"):
            markdown = path.read_text(encoding="utf-8")
            return ParseResult(markdown=markdown)

        # PDF 及其他格式：使用 LlamaIndex SimpleDirectoryReader 解析
        from .directory_parser import parse_with_simple_reader

        markdown = parse_with_simple_reader(str(path))
        return ParseResult(markdown=markdown)


def get_parser() -> Parser:
    """获取解析器单例。"""
    return Parser()
