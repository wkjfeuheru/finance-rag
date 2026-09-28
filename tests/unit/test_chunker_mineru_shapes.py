"""MinerU 实际产物形态带来的两个切块层缺陷（均来自真实研报实测）。

1. **页码归属几乎全为 0**：层级切块会给标题注入 ``#``，而解析层 content_list 里
   同一段文本没有这些标记，前缀匹配在第 1 个字符就断开（实测 0/16）。
2. **HTML 表没被当表格**：MinerU 对研报的财务表 / 盈利预测表输出 ``<table>``，
   而原实现只认 Markdown 竖线表——这些表既进不了 PostgreSQL，也做不了
   「表头 + 首行」索引，还会被递归切分器从标签中间切成 ``<td colspan=`` 碎片。
"""

from finance_rag.src.rag.ingestion.chunker import (
    HierarchicalChunker,
    _extract_atomic_units,
    _table_payload,
    _table_summary,
)
from finance_rag.src.rag.ingestion.pdf_assets import ContentBlock

_HTML_TABLE = (
    "<table><tr><td colspan=\"6\">资产负债表(亿)</td></tr>"
    "<tr><td></td><td>2024A</td><td>2025A</td></tr>"
    "<tr><td>货币资金</td><td>33.22</td><td>41.10</td></tr>"
    "<tr><td>资本公积</td><td>14.87</td><td>14.87</td></tr></table>"
)


def _chunk(text: str, block_text: str, page: int = 3):
    chunker = HierarchicalChunker(max_tokens=512, parent_max_tokens=1)
    return chunker.chunk_markdown(
        text, source="s.pdf", blocks=(ContentBlock(text=block_text, page=page),)
    )


def test_heading_marker_does_not_break_page_attribution():
    """``# 标题`` 与 content_list 的裸标题必须能对上。"""
    result = _chunk(
        "## 国机汽车，2026 年 H1 点评，利润增速同比超 20%\n\n正文。\n",
        "国机汽车，2026 年 H1 点评，利润增速同比超 20%",
    )

    pages = [c.metadata["start_page"] for c in result.chunks]
    assert pages == [3] * len(pages)


def test_decorative_bullet_is_ignored_when_matching():
    result = _chunk("## ◼ 走势比较\n\n内容。\n", "◼ 走势比较")

    assert result.chunks[0].metadata["start_page"] == 3


def test_unrelated_text_still_gets_no_page():
    """放宽匹配不等于乱填：无关文本仍然留 0。"""
    result = _chunk("## 完全无关的另一段内容。\n", "另一个主题的段落")

    assert result.chunks[0].metadata["start_page"] == 0


def test_html_table_is_treated_as_atomic_unit():
    units = _extract_atomic_units(f"前文。\n\n{_HTML_TABLE}\n\n后文。")

    kinds = [kind for kind, _ in units]
    assert kinds == ["text", "table", "text"]
    assert units[1][1].startswith("<table")


def test_markdown_and_html_tables_can_coexist():
    md_table = "| 年份 | 营收 |\n| --- | --- |\n| 2026E | 100 |"
    units = _extract_atomic_units(f"{md_table}\n\n{_HTML_TABLE}")

    assert [kind for kind, _ in units] == ["table", "table"]


def test_html_table_payload_and_summary():
    rows = _table_payload(_HTML_TABLE)

    # 空单元格被丢弃，标题行保留可见文字
    assert rows[0] == ["资产负债表(亿)"]
    assert ["货币资金", "33.22", "41.10"] in rows
    summary = _table_summary(_HTML_TABLE)
    # 摘要要带真实数字：单格大标题行被跳过，取「年份表头 + 首行数据」
    assert "货币资金" in summary
    assert "33.22" in summary
    assert "资本公积" not in summary


def test_html_table_is_not_split_across_chunks():
    chunker = HierarchicalChunker(max_tokens=8, parent_max_tokens=1)
    result = chunker.chunk_markdown(
        f"## 财务指标\n\n{_HTML_TABLE}\n", source="s.pdf", title="t"
    )

    table_children = [c for c in result.chunks if c.metadata["block_type"] == "table"]
    assert len(table_children) == 1
    assert "货币资金" in table_children[0].page_content      # 索引里有首行数据
    assert "资本公积" not in table_children[0].page_content   # 表体不进索引
    assert "<td" not in table_children[0].page_content        # 不是 HTML 碎片

    assert len(result.tables) == 1
    table = result.tables[0]
    assert table.markdown.startswith("|")                    # 统一成 Markdown
    assert "<td" not in table.markdown
    assert table.row_count >= 2


def test_no_html_fragments_leak_into_text_chunks():
    """回归：表体曾被切成 ``colspan=`` 之类的碎片混进正文块。"""
    chunker = HierarchicalChunker(max_tokens=8, parent_max_tokens=1)
    result = chunker.chunk_markdown(
        f"## 财务指标\n\n{_HTML_TABLE}\n\n## 风险提示\n\n宏观经济风险。\n",
        source="s.pdf",
        title="t",
    )

    text_chunks = [c for c in result.chunks if c.metadata["block_type"] == "text"]
    joined = "\n".join(c.page_content for c in text_chunks)
    assert "<td" not in joined
    assert "colspan" not in joined
