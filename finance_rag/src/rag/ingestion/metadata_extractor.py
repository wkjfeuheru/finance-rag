"""研报元数据抽取：正则优先 → LLM 回退 → 受控词表校验。

链路设计要点（改动前请先读）：

- **正则可解释、可复现**，研报文件名高度规范（``名称(代码)…-券商-日期``），
  能确定性拿到的字段绝不交给 LLM。
- **行业只由 LLM 抽取**，但必须落在申万 2021 版受控词表内，并校验
  「二级属于所选一级」。理由：检索过滤走精确匹配（``==`` / ``in``），
  LLM 若输出「白酒」（申万二级实为「白酒Ⅱ」），过滤会**静默漏召**。
- **抽不到就留空**。空值不参与过滤，因此不会误收窄召回；
  非法值一律丢弃并把 ``needs_review`` 置真，交给人工在文档页修正。
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

REPORT_TYPES: tuple[str, ...] = ("个股", "行业", "宏观")

# A 股 6 位代码首位只可能是 0/3/4/6/8（深/沪/北）。
# 收紧到前缀而不是 ``\d{6}``，可以挡住把日期片段（如 ``202608``）当成代码。
_SECURITY_CODE_RE = re.compile(r"^(?:0|3|4|6|8)\d{5}$")


def is_valid_security_code(value: Any) -> bool:
    """校验是否为合法的 A 股 6 位代码。"""
    return bool(_SECURITY_CODE_RE.fullmatch(str(value or "").strip()))

_TAXONOMY_PATH = (
    Path(__file__).resolve().parents[4] / "assets" / "taxonomy" / "sw_industry.json"
)

# ``贵州茅台(600519)`` / ``贵州茅台（600519）``
_NAME_CODE_RE = re.compile(r"([\u4e00-\u9fffA-Za-z0-9*]{2,12})[（(](\d{6})[)）]")
# 兜底：裸 6 位代码（前后不能再接数字，避免命中日期里的片段）
_BARE_CODE_RE = re.compile(r"(?<!\d)(\d{6})(?!\d)")
# ``-中信证券-`` / ``_中信证券`` 等
_BROKER_RE = re.compile(r"([\u4e00-\u9fff]{2,6}证券)")
# 文件名里的 8 位日期（如 20260815）不应被当成证券代码
_DATE_RE = re.compile(r"(?<!\d)(20\d{6})(?!\d)")

_FIELD_LIMITS = {
    "security_code": 32,
    "security_name": 64,
    "industry_l1": 32,
    "industry_l2": 32,
    "report_type": 16,
    "broker": 64,
}

_LLM_PROMPT = """你是金融研报元数据抽取器。请只输出 JSON，不要解释、不要代码块。

字段与取值要求：
- security_code：6 位 A 股代码；行业研报与宏观研报留空字符串。
- security_name：证券简称；无法确定留空字符串。
- report_type：只能取 "个股" / "行业" / "宏观" 之一。
- broker：发布机构名称，如 "中信证券"；无法确定留空字符串。
- industry_l1：**必须**从下面的一级行业清单中原样选择，不得改写、不得自创；无法确定留空字符串。
- industry_l2：**必须**是所填 industry_l1 对应二级清单中的一项；无法确定留空字符串。

一级行业 -> 二级行业清单：
{taxonomy}

研报文件名：{filename}
研报正文开头：
{head}

输出 JSON 模板：{{"security_code": "", "security_name": "", "report_type": "",
"broker": "", "industry_l1": "", "industry_l2": ""}}"""


@dataclass(frozen=True)
class ExtractedMetadata:
    """一篇研报的元数据抽取结果。"""

    security_code: str = ""
    security_name: str = ""
    industry_l1: str = ""
    industry_l2: str = ""
    report_type: str = ""
    broker: str = ""
    meta_source: str = "regex"
    needs_review: bool = False

    def as_metadata(self) -> dict[str, Any]:
        """并入文档级 metadata 的字典形式。"""
        return {
            "security_code": self.security_code,
            "security_name": self.security_name,
            "industry_l1": self.industry_l1,
            "industry_l2": self.industry_l2,
            "report_type": self.report_type,
            "broker": self.broker,
            "meta_source": self.meta_source,
            "needs_review": self.needs_review,
        }


@lru_cache(maxsize=1)
def load_industry_taxonomy(path: str | Path | None = None) -> dict[str, frozenset[str]]:
    """加载申万一级 -> 二级受控词表。

    这是**词表**而不是「代码 -> 行业」映射表：行业值由 LLM 抽取后在此校验。
    文件缺失或损坏时返回空词表（此时所有行业值都会被丢弃并标记待确认，
    而不是放行任意自由文本）。
    """
    target = Path(path) if path is not None else _TAXONOMY_PATH
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
        industries = payload["industries"]
        return {
            str(level1): frozenset(str(item) for item in level2s)
            for level1, level2s in industries.items()
        }
    except Exception as exc:  # noqa: BLE001 - 词表不可用不应让入库失败
        logger.warning("加载申万行业词表失败（%s）：%s", target, exc)
        return {}


def _clip(value: Any, field: str) -> str:
    return str(value or "").strip()[: _FIELD_LIMITS[field]]


def _regex_security(filename: str, title: str) -> tuple[str, str]:
    """从文件名/标题解析 ``(证券代码, 证券简称)``。"""
    scopes = [filename, title]
    for scope in scopes:
        match = _NAME_CODE_RE.search(scope or "")
        if match:
            return match.group(2), match.group(1)
    for scope in scopes:
        stripped = _DATE_RE.sub(" ", scope or "")
        match = _BARE_CODE_RE.search(stripped)
        if match:
            return match.group(1), ""
    return "", ""


def _regex_broker(filename: str, title: str) -> str:
    for scope in (filename, title):
        match = _BROKER_RE.search(scope or "")
        if match:
            return match.group(1)
    return ""


def _regex_report_type(filename: str, title: str, security_code: str) -> str:
    """报告类型只用文件名/标题判定。

    正文开头常常罗列一堆股票代码，用它判类型会把行业研报误判成个股研报。
    """
    text = f"{filename or ''} {title or ''}"
    if "宏观" in text:
        return "宏观"
    if "行业" in text:
        return "行业"
    if security_code:
        return "个股"
    return ""


def _model_text(raw: Any) -> str:
    """取出模型返回的**文本内容**。

    langchain 的 ``model.invoke`` 返回的是 ``AIMessage`` 而不是字符串；直接
    ``str(AIMessage)`` 会把 ``additional_kwargs=...``、``response_metadata=...``
    一起带上，于是「首尾花括号之间」切出来的根本不是合法 JSON，解析必然失败。
    实测这会让**每一篇**文档的元数据都为空（研报文件名是哈希串，正则兜不住），
    而失败是静默的：只表现为 ``meta_source='regex'``、``needs_review=True``。
    """
    if raw is None:
        return ""
    content = getattr(raw, "content", None)
    if content is None and isinstance(raw, dict):
        content = raw.get("content")
    if content is None:
        return str(raw)
    if isinstance(content, list):
        # 部分模型返回分段内容：[{"type": "text", "text": "..."}]
        return " ".join(
            str(part.get("text", "")) for part in content if isinstance(part, dict)
        )
    return str(content)


def _parse_json_payload(raw: Any) -> dict[str, Any]:
    """从模型输出里取出 JSON 对象（容忍消息对象、```json 代码块与前后噪声）。"""
    if isinstance(raw, dict):
        return raw
    text = _model_text(raw).strip()
    if not text:
        return {}
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return {}
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _llm_extract(*, filename: str, head: str, taxonomy: dict[str, frozenset[str]]) -> dict[str, Any]:
    """让轻量模型抽取字段。任何失败都返回空字典，由调用方降级。"""
    from finance_rag.src.core.config import get_model, get_rewrite_model

    model = get_rewrite_model() or get_model()
    if model is None:
        return {}
    listing = "\n".join(
        f"- {level1}：{'、'.join(sorted(level2s))}" for level1, level2s in sorted(taxonomy.items())
    )
    prompt = _LLM_PROMPT.format(
        taxonomy=listing or "（词表不可用，全部留空）",
        filename=filename or "",
        head=(head or "")[:2000],
    )
    return _parse_json_payload(model.invoke(prompt))


def _validate(
    payload: dict[str, Any], taxonomy: dict[str, frozenset[str]]
) -> tuple[dict[str, str], list[str]]:
    """校验并清洗抽取结果；返回 ``(合法字段, 被丢弃的字段名)``。"""
    cleaned: dict[str, str] = {}
    rejected: list[str] = []

    code = _clip(payload.get("security_code"), "security_code")
    if code:
        if is_valid_security_code(code):
            cleaned["security_code"] = code
        else:
            rejected.append("security_code")

    name = _clip(payload.get("security_name"), "security_name")
    if name:
        cleaned["security_name"] = name

    report_type = _clip(payload.get("report_type"), "report_type")
    if report_type:
        if report_type in REPORT_TYPES:
            cleaned["report_type"] = report_type
        else:
            rejected.append("report_type")

    broker = _clip(payload.get("broker"), "broker")
    if broker:
        cleaned["broker"] = broker

    level1 = _clip(payload.get("industry_l1"), "industry_l1")
    if level1:
        if level1 in taxonomy:
            cleaned["industry_l1"] = level1
            level2 = _clip(payload.get("industry_l2"), "industry_l2")
            if level2:
                if level2 in taxonomy[level1]:
                    cleaned["industry_l2"] = level2
                else:
                    # 二级不属于所选一级：丢二级、留一级，并标记待确认
                    rejected.append("industry_l2")
        else:
            rejected.append("industry_l1")

    return cleaned, rejected


def extract_metadata(
    filename: str,
    title: str = "",
    markdown_head: str = "",
    *,
    enable_llm: bool = True,
) -> ExtractedMetadata:
    """抽取一篇研报的元数据。

    ``enable_llm=False`` 时只走正则（行业必然为空、``needs_review`` 为真），
    让无模型环境仍可入库，同时把缺失显式暴露出来而不是装作正常。
    """
    taxonomy = load_industry_taxonomy()

    code, name = _regex_security(filename, title)
    values: dict[str, str] = {}
    if code:
        values["security_code"] = code
    if name:
        values["security_name"] = name
    broker = _regex_broker(filename, title)
    if broker:
        values["broker"] = broker
    report_type = _regex_report_type(filename, title, code)
    if report_type:
        values["report_type"] = report_type

    rejected: list[str] = []
    used_llm = False
    if enable_llm:
        try:
            payload = _llm_extract(filename=filename, head=markdown_head, taxonomy=taxonomy)
        except Exception as exc:  # noqa: BLE001 - 模型不可用不得阻断入库
            logger.warning("研报元数据 LLM 抽取失败，降级为正则结果：%s", exc)
            payload = {}
        if payload:
            cleaned, rejected = _validate(payload, taxonomy)
            # 正则拿到的值优先（可解释、可复现），LLM 只补空缺
            for key, value in cleaned.items():
                if not values.get(key):
                    values[key] = value
                    used_llm = True

    needs_review = bool(rejected)
    if not values.get("industry_l1"):
        # 行业缺失：行业/宏观研报靠 industry 维度兜底召回，缺了这条路就断了
        needs_review = True
    if not values.get("security_code") and values.get("report_type") == "个股":
        needs_review = True

    return ExtractedMetadata(
        security_code=values.get("security_code", ""),
        security_name=values.get("security_name", ""),
        industry_l1=values.get("industry_l1", ""),
        industry_l2=values.get("industry_l2", ""),
        report_type=values.get("report_type", ""),
        broker=values.get("broker", ""),
        meta_source="llm" if used_llm else "regex",
        needs_review=needs_review,
    )


def metadata_field_validators() -> dict[str, Callable[[str], bool]]:
    """研报元数据字段的取值校验器（入库、API 修正、检索过滤共用一套规则）。

    集中在一处的原因：这三条路径都以「精确匹配」的方式使用这些值，
    任何一条放宽，都会造成静默漏召或写入不可过滤的值。
    """
    taxonomy = load_industry_taxonomy()
    return {
        "security_code": is_valid_security_code,
        "security_name": lambda value: 0 < len(value) <= 64,
        "industry_l1": lambda value: value in taxonomy,
        "industry_l2": lambda value: any(value in children for children in taxonomy.values()),
        "report_type": lambda value: value in REPORT_TYPES,
        "broker": lambda value: 0 < len(value) <= 64,
    }


def validate_metadata_field(field: str, value: Any) -> str:
    """校验单个元数据字段值；非法或空返回空串。

    返回空串而不是抛异常：调用方（入库/API）按「留空」处理，
    空值不参与过滤，因此不会误收窄召回。
    """
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    validator = metadata_field_validators().get(field)
    if validator is None or not validator(text):
        return ""
    return text


def apply_metadata(
    metadata: dict[str, Any] | None, extracted: ExtractedMetadata
) -> dict[str, Any]:
    """把抽取结果并入文档级 metadata（不改动调用方传入的字典）。"""
    merged = dict(metadata or {})
    merged.update(extracted.as_metadata())
    return merged


__all__ = [
    "REPORT_TYPES",
    "ExtractedMetadata",
    "apply_metadata",
    "extract_metadata",
    "is_valid_security_code",
    "load_industry_taxonomy",
    "metadata_field_validators",
    "validate_metadata_field",
]
