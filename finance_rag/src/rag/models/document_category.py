"""文档分类常量与元数据工具。"""

from __future__ import annotations

from typing import Any

CATEGORY_META_KEY = "category"

DOCUMENT_CATEGORIES = {
    "investment_research": "投研类",
    "compliance_risk": "合规风控类",
    "business_operations": "业务运营类",
    "management": "管理类",
}


def merge_category_into_metadata(
    metadata: dict[str, Any] | None,
    category: str = "",
) -> dict[str, Any]:
    """将分类合并进元数据，保证分类字段存在。"""
    merged = dict(metadata or {})
    if category:
        merged[CATEGORY_META_KEY] = category
    elif CATEGORY_META_KEY not in merged:
        merged[CATEGORY_META_KEY] = ""
    return merged
