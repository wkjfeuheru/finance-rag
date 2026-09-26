"""Local MinerU document parsing adapter using the Python API."""
from __future__ import annotations

import json
import logging
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from finance_rag.src.core.config import (
    MINERU_BACKEND,
    MINERU_FORMULA_ENABLE,
    MINERU_KEEP_ARTIFACTS_ON_ERROR,
    MINERU_METHOD,
    MINERU_OUTPUT_DIR,
    MINERU_TABLE_ENABLE,
)
from finance_rag.src.rag.ingestion.pdf_assets import (
    ContentBlock,
    ImageAsset,
    extract_blocks,
    extract_images,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ParseResult:
    markdown: str
    parser: str = "mineru"
    parser_version: str = "unknown"
    stats: dict[str, int | str] = field(default_factory=dict)
    # 内嵌图片与分页文本块：Markdown 里拿不到，页码溯源与图片索引都依赖它们。
    images: tuple[ImageAsset, ...] = ()
    blocks: tuple[ContentBlock, ...] = ()


class MinerUParseError(RuntimeError):
    """MinerU was unavailable or did not produce valid output."""


def _find_markdown(root: Path) -> Path | None:
    candidates = sorted(root.rglob("*.md"), key=lambda p: (len(p.parts), str(p)))
    return candidates[0] if candidates else None


_CONTENT_LIST_NAME = "content_list.json"
# content_list 的分块类型字段：正文优先，表格与代码体兜底，最后才用图注 / 表注。
_BLOCK_TEXT_KEYS = ("text", "table_body", "code_body")
_BLOCK_CAPTION_KEYS = ("img_caption", "table_caption")


def _find_content_list(markdown_path: Path) -> Path | None:
    """定位 MinerU 的 ``content_list.json``（与 Markdown 同级或上一层）。

    MinerU 不同 backend 的产物层级略有差异，因此两级都找一遍，
    并优先取层级最浅的那个。
    """
    candidates: list[Path] = []
    for root in (markdown_path.parent, markdown_path.parent.parent):
        if root.is_dir():
            candidates.extend(root.rglob(_CONTENT_LIST_NAME))
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: (len(p.parts), str(p)))[0]


def _blocks_from_content_list(payload: Any) -> tuple[ContentBlock, ...]:
    """把 MinerU content_list 转成 ``(文本, 页码)`` 序列。

    content_list 的``page_idx``是 0-based，这里统一转成 1-based 对外暴露。
    对字段缺失 / 类型不符 / 空文本一律跳过，不抛异常——页码宁可缺失，不可乱填。
    """
    if not isinstance(payload, list):
        return ()
    blocks: list[ContentBlock] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        raw_page = item.get("page_idx")
        if isinstance(raw_page, bool) or not isinstance(raw_page, int) or raw_page < 0:
            continue
        text = ""
        for key in _BLOCK_TEXT_KEYS:
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                text = value.strip()
                break
        if not text:
            for key in _BLOCK_CAPTION_KEYS:
                value = item.get(key)
                if isinstance(value, list):
                    joined = " ".join(
                        str(part).strip() for part in value if str(part).strip()
                    )
                    if joined:
                        text = joined
                        break
        if not text:
            continue
        blocks.append(ContentBlock(text=text, page=raw_page + 1))
    return tuple(blocks)


def _load_content_list(markdown_path: Path) -> Any | None:
    content_list_path = _find_content_list(markdown_path)
    if content_list_path is None:
        return None
    try:
        return json.loads(content_list_path.read_text(encoding="utf-8", errors="replace"))
    except Exception as exc:  # noqa: BLE001 - 产物异常只降级，不影响入库
        logger.warning("读取 content_list 失败（%s）：%s", content_list_path, exc)
        return None


def _extract_blocks(
    markdown_path: Path, input_path: Path
) -> tuple[tuple[ContentBlock, ...], str]:
    """优先用 MinerU 自己的分块（与 Markdown 对齐更好），否则退回 PDF 文本块。

    返回 ``(blocks, 来源标记)``，来源写入 ``stats`` 便于排障。
    """
    blocks = _blocks_from_content_list(_load_content_list(markdown_path))
    if blocks:
        return blocks, "content_list"
    blocks = extract_blocks(input_path)
    return blocks, "pdf_text" if blocks else "none"


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
        # 打开 content_list：分页文本块是页码级溯源的唯一对齐来源。
        f_dump_content_list=True,
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
        # 注意：必须在 finally 删除 work_dir 之前读取 content_list。
        blocks, blocks_source = _extract_blocks(markdown_path, input_path)
        images = extract_images(input_path)
        from mineru.version import __version__

        return ParseResult(
            markdown=markdown,
            parser_version=__version__,
            stats={
                "input_type": input_path.suffix.lower(),
                "output": str(markdown_path),
                "blocks_source": blocks_source,
                "block_count": len(blocks),
                "image_count": len(images),
            },
            images=images,
            blocks=blocks,
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
