"""拒答判定的回归测试。

为什么值得单独锁：负样本拒答正确率是硬阈值（=1.00），而它依赖一个**关键词代理**。
真实模型很少说"无法回答"，更常说"检索到的内容**未提供**…""现有知识库中暂未检索到…"。
关键词表漏一个词，就会把正确拒答判成幻觉作答，把达标指标算成不达标——18 篇语料
基线上真实发生过（首轮把 2/2 算成了 1/2）。这里把两类易错点固定下来：

* 常见拒答措辞必须被识别（含"未提供/未提及/未包含"）；
* 关键词不得用于**正例**：一句正常作答里出现"未包含"不能被当成拒答。
"""

from __future__ import annotations

import pytest

from finance_rag.src.eval.runner import (
    REJECTION_KEYWORDS,
    negative_rejection_score,
)


@pytest.mark.parametrize(
    "answer",
    [
        "抱歉，我无法给出可靠回答：检索到的文档内容不足以支撑该问题。",
        "检索到的内容未提供宁德时代2026年上半年储能电池出货量的具体数据。",
        "现有知识库中暂未检索到贵州茅台2026年中报的营业总收入相关数据。",
        "知识库中未收录该公司的财务数据。",
        "资料未提及该指标。",
        "",
        "   ",
    ],
)
def test_declines_score_as_rejection(answer: str) -> None:
    assert negative_rejection_score(answer) == 1.0


@pytest.mark.parametrize(
    "answer",
    [
        "2026H1 公司实现营收 76.5 亿元，同比+10.8%。[1]",
        "公司评级为买入（首次）。[2]",
    ],
)
def test_substantive_answers_are_not_rejections(answer: str) -> None:
    assert negative_rejection_score(answer) == 0.0


def test_keyword_proxy_must_not_be_applied_to_positive_answers() -> None:
    """正例反例：一段正常作答因为句中出现"未包含"被关键词法误判为拒答。

    这正是脚本里正例只看生产策略标志（``answer_rejected``）的原因；
    本测试固定这个反例，避免以后有人图省事把关键词法用到正例上。
    """
    real_answer = (
        "从检索内容看，个股深度报告的判断侧重在公司层面。[2] "
        "而检索到的宏观周报部分仅显示标题，未包含具体判断内容。"
    )
    assert negative_rejection_score(real_answer) == 1.0, "关键词法对正例确实会误判"
    # 因此正例判定必须换成生产策略标志：该答案 answer_rejected=False -> 不作拒答处理
    assert REJECTION_KEYWORDS, "关键词表不应为空"
