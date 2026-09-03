"""Document parsing entry point backed exclusively by local MinerU."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .mineru_parser import ParseResult, parse_with_mineru


@dataclass
class Parser:
    """Thin stable facade for the local MinerU adapter."""

    def parse(self, path: str | Path, *, source: str = "", title: str = "") -> ParseResult:
        del title
        return parse_with_mineru(path, source=source or Path(path).name)


def get_parser() -> Parser:
    return Parser()


__all__ = ["ParseResult", "Parser", "get_parser"]
