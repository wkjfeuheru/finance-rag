"""Agent 状态模型。"""

from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict):
    """LangGraph Agent 在各节点之间传递的 canonical 状态。"""

    query: str
    history: list[dict[str, str]]
    rewritten_query: str
    keywords: list[str]
    sub_queries: list[str]
    docs: list[dict[str, Any]]
    reflection_round: int
    answer: str
    sources: list[dict[str, Any]]
    error: str


__all__ = ["AgentState"]
