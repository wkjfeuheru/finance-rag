"""查询复杂度分类与动态 K 解析。

规则分类器（零 LLM 成本）：
* simple：短问题、无复杂连接词、单个问号；
* complex：长问题 / 多复杂提示词（对比、区别、流程、步骤等）/ 多问号；
* normal：其余。

``resolve_dynamic_k`` 仅在 ``DYNAMIC_K`` 开启时生效，按档位返回
（k, rerank_top_n）。用户显式指定 k 时应由调用方跳过本函数。
"""

from __future__ import annotations

from finance_rag.src.core.config import DYNAMIC_K, DYNAMIC_K_MAP

COMPLEX_HINTS = (
    "对比", "区别", "差异", "比较", "以及", "分别", "步骤", "流程",
    "影响", "原因", "为什么", "如何", "有哪些", "关系", "优缺点",
)


def classify_query_complexity(query: str) -> str:
    """返回 "simple" | "normal" | "complex"。"""
    q = (query or "").strip()
    marks = q.count("？") + q.count("?")
    hints = sum(1 for hint in COMPLEX_HINTS if hint in q)

    if len(q) <= 15 and hints == 0 and marks <= 1:
        return "simple"
    if len(q) > 40 or hints >= 2 or marks >= 2:
        return "complex"
    return "normal"


def resolve_dynamic_k(
    query: str,
    k: int,
    rerank_top_n: int,
) -> tuple[int, int]:
    """按问题复杂度返回召回参数；DYNAMIC_K 关闭时原样返回。"""
    if not DYNAMIC_K:
        return k, rerank_top_n
    profile = DYNAMIC_K_MAP.get(classify_query_complexity(query)) or {}
    return (
        int(profile.get("k", k)),
        int(profile.get("rerank_top_n", rerank_top_n)),
    )
