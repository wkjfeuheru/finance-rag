"""Local MinerU document parsing adapter using the Python API."""
from __future__ import annotations

import logging
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from finance_rag.src.core.config import (
    MINERU_BACKEND,
    MINERU_FORMULA_ENABLE,
    MINERU_KEEP_ARTIFACTS_ON_ERROR,
    MINERU_METHOD,
    MINERU_OUTPUT_DIR,
    MINERU_TABLE_ENABLE,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ParseResult:
    markdown: str
    parser: str = "mineru"
    parser_version: str = "unknown"
    stats: dict[str, int | str] = field(default_factory=dict)


class MinerUParseError(RuntimeError):
    """MinerU was unavailable or did not produce valid output."""


def _find_markdown(root: Path) -> Path | None:
    candidates = sorted(root.rglob("*.md"), key=lambda p: (len(p.parts), str(p)))
    return candidates[0] if candidates else None


def _parse_binary_document(input_path: Path, output_dir: Path) -> Path | None:
    """Invoke MinerU in-process and return the generated Markdown path."""
    from mineru.cli.common import do_parse, read_fn

    suffix = input_path.suffix.lower().lstrip(".")
    document_name = input_path.stem
    do_parse(
        output_dir=str(output_dir),
        pdf_file_names=[document_name],
        pdf_bytes_list=[read_fn(input_path, suffix)],
        p_lang_list=["ch"],
        formula_enable=MINERU_FORMULA_ENABLE,
        table_enable=MINERU_TABLE_ENABLE,
        backend=MINERU_BACKEND,
        parse_method=MINERU_METHOD,
        f_draw_layout_bbox=False,
        f_draw_span_bbox=False,
        f_dump_md=True,
        f_dump_middle_json=False,
        f_dump_model_output=False,
        f_dump_orig_pdf=False,
        f_dump_content_list=False,
        image_analysis=False,
    )
    return _find_markdown(output_dir / document_name / MINERU_METHOD)


def parse_with_mineru(path: str | Path, *, source: str = "") -> ParseResult:
    """Parse one document with local MinerU and return its Markdown output."""
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"待解析文件不存在：{input_path}")
    if input_path.suffix.lower() in {".md", ".markdown", ".txt"}:
        text = input_path.read_text(encoding="utf-8", errors="replace")
        return ParseResult(markdown=text, stats={"input_type": "text"})

    Path(MINERU_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix="mineru-", dir=MINERU_OUTPUT_DIR))
    keep_dir = False
    try:
        markdown_path = _parse_binary_document(input_path, work_dir)
        if markdown_path is None:
            raise MinerUParseError(f"MinerU 未生成 Markdown 输出：{source or input_path.name}")
        markdown = markdown_path.read_text(encoding="utf-8", errors="replace").strip()
        if not markdown:
            raise MinerUParseError(f"MinerU 输出为空：{source or input_path.name}")
        from mineru.version import __version__

        return ParseResult(
            markdown=markdown,
            parser_version=__version__,
            stats={"input_type": input_path.suffix.lower(), "output": str(markdown_path)},
        )
    except Exception as exc:
        keep_dir = MINERU_KEEP_ARTIFACTS_ON_ERROR
        if isinstance(exc, MinerUParseError):
            raise
        raise MinerUParseError(
            f"MinerU 解析失败（source={source or input_path.name}）：{exc}"
        ) from exc
    finally:
        if keep_dir:
            logger.warning("保留 MinerU 失败产物：%s", work_dir)
        else:
            shutil.rmtree(work_dir, ignore_errors=True)


__all__ = ["MinerUParseError", "ParseResult", "parse_with_mineru"]
