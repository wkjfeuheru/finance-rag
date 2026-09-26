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
    infer_filters: bool = Field(
        default=True,
        description=(
            "true 时由 LLM 根据问题自动推断过滤条件；"
            "false 时只用显式传入的 filters——前端清除过滤 chip 后靠它生效"
        ),
    )


class SourceInfo(BaseModel):
    index: int
    title: str
    source: str
    category: str = ""
    collection: str = ""
    chunk: int | None = None
    score: float
    rerank_score: float | None = None
    preview: str
    # 证据定位：前端据此做页码跳转、元数据 chip 与「表已截断」提示
    date: str = ""
    block_type: str = "text"
    start_page: int = 0
    end_page: int = 0
    image_key: str = ""
    truncated: bool = False
    security_code: str = ""
    industry_l1: str = ""
    report_type: str = ""


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
