"""答案上下文的证据标注（报告日期 / 页码 / 块类型）。

「整合数据」场景里，分析师必须能判断两件事：这个数字出自哪一天的报告、
在原文第几页。因此上下文块头必须带这些信息，且表格不能被上下文截断规则
砍掉——表格恰恰是数据密度最高的块。
"""

from finance_rag.src.services.chat_service import build_context, merge_docs


def _doc(**overrides) -> dict:
    base = {
        "content": "燃机订单超预期。",
        "title": "贵州茅台2026中报点评",
        "source": "600519-2026中报点评.pdf",
        "category": "investment_research",
        "date": "2026-08-15",
        "block_type": "text",
        "start_page": 4,
        "end_page": 5,
        "score": 1.0,
    }
    base.update(overrides)
    return base


def test_context_header_carries_date_and_page():
    context, _ = build_context([_doc()])

    assert "报告日期：2026-08-15" in context
    assert "页码：4-5" in context
    assert "燃机订单超预期。" in context


def test_context_omits_missing_date_and_page_without_faking_them():
    """页码/日期缺失时不能编一个出来——留空比填错好。"""
    context, _ = build_context([_doc(date="", start_page=0, end_page=0)])

    assert "报告日期" not in context
    assert "页码" not in context


def test_context_single_page_is_not_rendered_as_range():
    context, _ = build_context([_doc(start_page=7, end_page=7)])

    assert "页码：7" in context
    assert "页码：7-7" not in context


def test_context_labels_table_and_image_blocks():
    table, _ = build_context([_doc(block_type="table")])
    image, _ = build_context([_doc(block_type="image")])

    assert "类型：表格" in table
    assert "类型：图片" in image


def test_table_content_is_not_truncated_to_text_limit():
    """表格按 8000 字符上限注入，普通文本仍是 2000。"""
    table_body = "| 年份 | 营收 |\n" + "\n".join(
        f"| 2026E | 第{i}行 |" for i in range(400)
    )
    assert len(table_body) > 2000

    table_context, _ = build_context([_doc(block_type="table", content=table_body)])
    text_context, _ = build_context([_doc(block_type="text", content=table_body)])

    assert "第399行" in table_context
    assert "第399行" not in text_context


def test_truncated_table_is_announced_in_context():
    table_body = "| 年份 | 营收 |\n" + "\n".join(
        f"| 2026E | 第{i}行 |" for i in range(2000)
    )

    context, sources = build_context(
        [_doc(block_type="table", content=table_body, truncated=True)]
    )

    assert "表已截断" in context
    assert sources[0]["truncated"] is True


def test_sources_surface_page_block_type_and_metadata():
    _, sources = build_context(
        [_doc(block_type="table", security_code="600519", industry_l1="食品饮料")]
    )

    source = sources[0]
    assert source["start_page"] == 4
    assert source["end_page"] == 5
    assert source["block_type"] == "table"
    assert source["security_code"] == "600519"
    assert source["industry_l1"] == "食品饮料"


def test_merge_docs_breaks_score_ties_by_newer_date_first():
    merged = merge_docs(
        [_doc(date="2023-01-01", content="旧报告")],
        [_doc(date="2026-08-15", content="新报告")],
        top_k=2,
    )

    assert [doc["content"] for doc in merged] == ["新报告", "旧报告"]


def test_merge_docs_keeps_score_as_primary_key():
    merged = merge_docs(
        [_doc(date="2023-01-01", content="高分旧报告", score=1.0)],
        [_doc(date="2026-08-15", content="低分新报告", score=0.4)],
        top_k=2,
    )

    assert [doc["content"] for doc in merged] == ["高分旧报告", "低分新报告"]
