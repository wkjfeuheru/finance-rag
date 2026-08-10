"""聊天相关的 Pydantic 请求/响应模型。"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str = Field(..., description="user | assistant")
    content: str


class ChatRequest(BaseModel):
    query: str
    history: list[ChatMessage] = Field(default_factory=list)
    use_rewrite: bool | None = None
    use_rerank: bool = True
    k: int = Field(default=5, ge=1, le=20)
    rerank_top_n: int = Field(default=3, ge=1, le=10)
    strategy: str = Field(default="default", description="default | optimized")
    filters: dict | None = Field(default=None, description="元数据过滤条件")


class SourceInfo(BaseModel):
    index: int
    title: str
    source: str
    chunk: int | None = None
    score: float
    preview: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceInfo] = []
    rewritten_query: str
    citation_validation: dict | None = None
