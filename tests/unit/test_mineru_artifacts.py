"""MinerU 产物定位：文件名带文档名前缀，且同目录还有 v2 版本。

这两个坑都来自真实产物实测：用精确名 ``content_list.json`` 找不到任何文件
（真实文件名是 ``H3_AP..._content_list.json``），于是静默退化成 pymupdf 页文本，
页码归属几乎全为 0；而 ``*content_list.json`` 又不能把嵌套结构的 v2 选中。
"""

from pathlib import Path

from finance_rag.src.rag.ingestion.mineru_parser import (
    _blocks_from_content_list,
    _find_content_list,
)


def _write(path: Path, text: str = "{}") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_finds_content_list_with_document_name_prefix(tmp_path):
    auto = tmp_path / "H3_AP1" / "auto"
    markdown = _write(auto / "H3_AP1.md", "# t")
    expected = _write(auto / "H3_AP1_content_list.json")

    assert _find_content_list(markdown) == expected


def test_does_not_pick_the_v2_variant(tmp_path):
    auto = tmp_path / "doc" / "auto"
    markdown = _write(auto / "doc.md", "# t")
    flat = _write(auto / "doc_content_list.json")
    _write(auto / "doc_content_list_v2.json", "[]")

    assert _find_content_list(markdown) == flat


def test_returns_none_when_no_content_list(tmp_path):
    auto = tmp_path / "doc" / "auto"
    markdown = _write(auto / "doc.md", "# t")

    assert _find_content_list(markdown) is None


def test_searches_one_level_up_as_well(tmp_path):
    markdown = _write(tmp_path / "doc" / "auto" / "deep" / "doc.md", "# t")
    expected = _write(tmp_path / "doc" / "auto" / "doc_content_list.json")

    assert _find_content_list(markdown) == expected


def test_chart_caption_counts_as_block_text():
    """真实产物里图表注解字段是 chart_caption，不是 img_caption。"""
    payload = [
        {"type": "chart", "chart_caption": ["图表1 盈利预测"], "page_idx": 2},
        {"type": "text", "text": "正文", "page_idx": 0},
    ]

    blocks = _blocks_from_content_list(payload)

    assert [(b.text, b.page) for b in blocks] == [("图表1 盈利预测", 3), ("正文", 1)]
