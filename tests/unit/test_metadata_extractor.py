"""研报元数据抽取的回归测试。

这一层的失败模式是**静默漏召**而非报错：过滤走精确匹配，一旦把行业写成
「白酒」而词表里是「白酒Ⅱ」，`industry_l1 == "食品饮料"` 的过滤就会悄悄
少召回一批文档，而分析师不会知道。因此这里重点锁住词表校验行为。

所有用例都显式传 ``enable_llm``，避免测试意外触发真实模型调用。
"""

from finance_rag.src.rag.ingestion import metadata_extractor
from finance_rag.src.rag.ingestion.metadata_extractor import (
    apply_metadata,
    extract_metadata,
    load_industry_taxonomy,
)


def _llm(payload: dict):
    """替换 LLM 抽取，让校验逻辑可以被独立测试。"""
    return lambda **_: dict(payload)


def test_filename_regex_extracts_code_name_and_broker():
    meta = extract_metadata(
        "贵州茅台(600519)2026中报点评-中信证券-20260815.pdf",
        "贵州茅台2026中报点评",
        "",
        enable_llm=False,
    )

    assert meta.security_code == "600519"
    assert meta.security_name == "贵州茅台"
    assert meta.broker == "中信证券"
    assert meta.report_type == "个股"
    assert meta.meta_source == "regex"


def test_industry_outside_taxonomy_is_dropped(monkeypatch):
    """「白酒」是三级叫法，申万二级是「白酒Ⅱ」——非法值必须丢弃并标记待确认。"""
    monkeypatch.setattr(
        metadata_extractor,
        "_llm_extract",
        _llm({"industry_l1": "白酒", "industry_l2": "白酒"}),
    )

    meta = extract_metadata("研报.pdf", "研报", "正文", enable_llm=True)

    assert meta.industry_l1 == ""
    assert meta.industry_l2 == ""
    assert meta.needs_review is True


def test_industry_l2_must_belong_to_selected_l1(monkeypatch):
    monkeypatch.setattr(
        metadata_extractor,
        "_llm_extract",
        _llm({"industry_l1": "食品饮料", "industry_l2": "半导体"}),
    )

    meta = extract_metadata("研报.pdf", "研报", "正文", enable_llm=True)

    assert meta.industry_l1 == "食品饮料"   # 一级合法，保留
    assert meta.industry_l2 == ""            # 二级不属于该一级，丢弃
    assert meta.needs_review is True


def test_valid_industry_passes_and_marks_llm_source(monkeypatch):
    monkeypatch.setattr(
        metadata_extractor,
        "_llm_extract",
        _llm(
            {
                "security_code": "600519",
                "security_name": "贵州茅台",
                "industry_l1": "食品饮料",
                "industry_l2": "白酒Ⅱ",
                "report_type": "个股",
            }
        ),
    )

    meta = extract_metadata("研报.pdf", "研报", "正文", enable_llm=True)

    assert (meta.industry_l1, meta.industry_l2) == ("食品饮料", "白酒Ⅱ")
    assert meta.meta_source == "llm"
    assert meta.needs_review is False


def test_regex_values_are_kept_when_llm_returns_nothing(monkeypatch):
    monkeypatch.setattr(metadata_extractor, "_llm_extract", _llm({}))

    meta = extract_metadata(
        "宁德时代(300750)深度报告-华泰证券-20260901.pdf", "", "", enable_llm=True
    )

    assert meta.security_code == "300750"
    assert meta.broker == "华泰证券"
    assert meta.needs_review is True   # 行业缺失：行业研报的兜底召回路径失效


def test_regex_wins_over_llm_for_deterministic_fields(monkeypatch):
    """文件名能确定性拿到的字段，不允许被模型改写。"""
    monkeypatch.setattr(
        metadata_extractor,
        "_llm_extract",
        _llm({"security_code": "000001", "broker": "别的券商"}),
    )

    meta = extract_metadata(
        "贵州茅台(600519)点评-中信证券.pdf", "", "", enable_llm=True
    )

    assert meta.security_code == "600519"
    assert meta.broker == "中信证券"


def test_invalid_code_and_report_type_are_dropped(monkeypatch):
    monkeypatch.setattr(
        metadata_extractor,
        "_llm_extract",
        _llm({"security_code": "60051", "report_type": "策略"}),
    )

    meta = extract_metadata("研报.pdf", "研报", "正文", enable_llm=True)

    assert meta.security_code == ""
    assert meta.report_type == ""
    assert meta.needs_review is True


def test_macro_report_type_comes_from_filename():
    meta = extract_metadata(
        "2026年宏观展望-中金公司-20261101.pdf", "", "", enable_llm=False
    )

    assert meta.report_type == "宏观"
    assert meta.security_code == ""


def test_industry_report_type_comes_from_filename():
    meta = extract_metadata(
        "白酒行业深度报告-招商证券-20260901.pdf", "", "", enable_llm=False
    )

    assert meta.report_type == "行业"
    assert meta.broker == "招商证券"


def test_disabled_llm_leaves_industry_empty_and_needs_review():
    """无模型环境仍可入库，但必须把行业缺失显式标出来。"""
    meta = extract_metadata("贵州茅台(600519)点评.pdf", "", "", enable_llm=False)

    assert meta.industry_l1 == ""
    assert meta.meta_source == "regex"
    assert meta.needs_review is True


def test_taxonomy_ships_full_sw2021_enum():
    taxonomy = load_industry_taxonomy()

    assert len(taxonomy) == 31
    assert sum(len(v) for v in taxonomy.values()) == 134
    assert "白酒Ⅱ" in taxonomy["食品饮料"]
    assert "半导体" in taxonomy["电子"]


def test_llm_extraction_failure_degrades_to_regex(monkeypatch):
    def _boom(**_):
        raise RuntimeError("模型不可用")

    monkeypatch.setattr(metadata_extractor, "_llm_extract", _boom)

    meta = extract_metadata(
        "贵州茅台(600519)点评-中信证券.pdf", "", "", enable_llm=True
    )

    assert meta.security_code == "600519"
    assert meta.industry_l1 == ""
    assert meta.meta_source == "regex"


# --- 真实模型返回形状：langchain 的 AIMessage，不是字符串 ----------------------


class _FakeAIMessage:
    """模拟 langchain ``AIMessage``：``str()`` 会带上元数据，``.content`` 才是正文。"""

    def __init__(self, content) -> None:
        self.content = content
        self.additional_kwargs = {"refusal": None}
        self.response_metadata = {"model_name": "qwen-turbo", "finish_reason": "stop"}

    def __str__(self) -> str:  # pragma: no cover - 仅用于复现真实 str() 行为
        return (
            f"content={self.content!r} additional_kwargs={self.additional_kwargs} "
            f"response_metadata={self.response_metadata}"
        )


def test_parse_json_payload_unwraps_langchain_message():
    """实测踩过的坑：把 AIMessage 直接塞进解析器，首尾花括号切出来不是合法 JSON。"""
    message = _FakeAIMessage(
        '{"security_code": "600335", "security_name": "国机汽车", '
        '"report_type": "个股", "broker": "太平洋证券", '
        '"industry_l1": "汽车", "industry_l2": "汽车零部件"}'
    )

    payload = metadata_extractor._parse_json_payload(message)

    assert payload["security_code"] == "600335"
    assert payload["industry_l2"] == "汽车零部件"


def test_parse_json_payload_handles_segmented_content():
    message = _FakeAIMessage([{"type": "text", "text": '{"report_type": "宏观"}'}])

    assert metadata_extractor._parse_json_payload(message) == {"report_type": "宏观"}


def test_llm_extraction_end_to_end_from_message(monkeypatch):
    """整条链路：模型返回消息对象 → 校验后落进元数据。"""
    monkeypatch.setattr(
        metadata_extractor,
        "_llm_extract",
        lambda **_: metadata_extractor._parse_json_payload(
            _FakeAIMessage(
                '{"security_code": "600335", "security_name": "国机汽车", '
                '"report_type": "个股", "industry_l1": "汽车"}'
            )
        ),
    )

    meta = extract_metadata("H3_AP202609261829925702_1.pdf", "", "首页正文", enable_llm=True)

    assert meta.security_code == "600335"
    assert meta.industry_l1 == "汽车"
    assert meta.meta_source == "llm"
    assert meta.needs_review is False


def test_apply_metadata_merges_without_mutating_input():
    original = {"category": "investment_research", "date": "2026-08-15"}
    extracted = extract_metadata(
        "贵州茅台(600519)2026中报点评-中信证券.pdf", "", "", enable_llm=False
    )

    merged = apply_metadata(original, extracted)

    assert merged["category"] == "investment_research"   # 原有字段保留
    assert merged["security_code"] == "600519"
    assert merged["meta_source"] == "regex"
    assert "security_code" not in original               # 不污染入参
