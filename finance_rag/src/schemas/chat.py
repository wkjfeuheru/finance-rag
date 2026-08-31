"""聊天相关的 Pydantic 请求/响应模型。"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str = Field(..., description="user | assistant")
    content: str


class ChatRequest(BaseModel):
    query: str
    history: list[ChatMessage] = Field(default_factory=list)
    use_rerank: bool = True
    k: int | None = Field(
        default=None, ge=1, le=20,
        description="召回数量；为空时用默认值（开启动态 K 时按问题复杂度自动调整）",
    )
    rerank_top_n: int | None = Field(default=None, ge=1, le=10, description="重排序截断数")
    filters: dict | None = Field(default=None, description="元数据过滤条件")


class SourceInfo(BaseModel):
    index: int
    title: str
    source: str
    category: str = ""
    collection: str = ""
    chunk: int | None = None
    score: float
    preview: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceInfo] = []
    rewritten_query: str
    citation_validation: dict | None = None
    answer_rejected: bool = Field(
        default=False, description="拒答策略触发（检索为空或引用校验低分）"
    )
    low_confidence: bool = Field(
        default=False, description="回答置信度偏低（引用校验分数低于阈值）"
    )
