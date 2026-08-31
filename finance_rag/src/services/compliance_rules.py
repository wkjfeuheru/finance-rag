"""合规红线规则引擎（确定性匹配，不依赖 LLM）。

合规审查中的「红线」必须是硬规则：命中即给出结论与法规依据，避免 LLM 幻觉。
本模块提供规则数据结构与关键词匹配函数，供合规审查链路在生成前调用，
实现「先规则、后模型」的确定性兜底。

说明：``RED_LINE_RULES`` 为内置示例规则，接入实际业务时应替换为
企业真实的合规红线清单（可改为从配置文件/数据库加载）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RedLineRule:
    """单条合规红线规则。

    Attributes:
        rule_id: 规则唯一标识。
        subject: 适用主体（机构 / 业务 / 产品）。
        behavior: 违规行为描述。
        legal_basis: 法规依据（法规名 + 条款）。
        keywords: 触发关键词，文本命中任一关键词即视为命中该规则。
        severity: 严重程度（high / medium）。
    """

    rule_id: str
    subject: str
    behavior: str
    legal_basis: str
    keywords: tuple[str, ...]
    severity: str = "high"

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "subject": self.subject,
            "behavior": self.behavior,
            "legal_basis": self.legal_basis,
            "keywords": list(self.keywords),
            "severity": self.severity,
        }


# ---------------------------------------------------------------------------
# 内置示例红线规则（接入实际业务时替换为真实红线清单）
# ---------------------------------------------------------------------------

RED_LINE_RULES: tuple[RedLineRule, ...] = (
    RedLineRule(
        rule_id="RL-001",
        subject="金融机构",
        behavior="未经批准从事金融业务",
        legal_basis="《中华人民共和国银行业监督管理法》",
        keywords=("未经批准", "未取得许可", "无证经营", "擅自从事金融业务", "未持牌"),
        severity="high",
    ),
    RedLineRule(
        rule_id="RL-002",
        subject="金融业务",
        behavior="非法集资或变相吸收公众存款",
        legal_basis="《防范和处置非法集资条例》",
        keywords=("非法集资", "吸收公众存款", "变相吸收", "保本保息", "承诺高额回报"),
        severity="high",
    ),
    RedLineRule(
        rule_id="RL-003",
        subject="金融机构",
        behavior="未履行反洗钱义务",
        legal_basis="《中华人民共和国反洗钱法》",
        keywords=("未履行反洗钱", "未按规定报告可疑交易", "客户身份识别不到位", "反洗钱缺失"),
        severity="high",
    ),
    RedLineRule(
        rule_id="RL-004",
        subject="证券业务",
        behavior="内幕交易或操纵市场",
        legal_basis="《中华人民共和国证券法》",
        keywords=("内幕交易", "操纵市场", "利用未公开信息", "老鼠仓", "拉抬股价"),
        severity="high",
    ),
    RedLineRule(
        rule_id="RL-005",
        subject="金融产品",
        behavior="虚假宣传或误导投资者",
        legal_basis="《中华人民共和国证券法》及相关监管规定",
        keywords=("虚假宣传", "误导投资者", "夸大收益", "隐瞒风险", "承诺收益"),
        severity="medium",
    ),
)


def match_red_lines(text: str) -> list[dict[str, Any]]:
    """对文本做红线关键词匹配，返回命中的规则列表（按定义顺序）。

    Args:
        text: 待审查文本（用户问题或待审文档内容）。

    Returns:
        命中的规则字典列表，每项含 rule_id / subject / behavior /
        legal_basis / keywords / severity。
    """
    if not text or not text.strip():
        return []
    hits: list[dict[str, Any]] = []
    for rule in RED_LINE_RULES:
        if any(keyword in text for keyword in rule.keywords):
            hits.append(rule.to_dict())
    return hits


def has_red_line(text: str) -> bool:
    """快速判断文本是否命中任意红线（供链路短路判断）。"""
    return bool(match_red_lines(text))
