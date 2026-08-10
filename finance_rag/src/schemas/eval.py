"""评估相关的 Pydantic 请求/响应模型。"""

from __future__ import annotations

from pydantic import BaseModel, Field


class EvaluateStrategyRequest(BaseModel):
    """策略评估请求。"""
    use_dense_only: bool = Field(default=False, description="True 时只走稠密向量检索")
    use_rerank: bool = Field(default=False, description="是否启用 BGE 重排序")
    rerank_top_n: int = Field(default=3, ge=1, le=10, description="重排序返回结果数")
    k: int = Field(default=5, ge=1, le=20, description="检索深度")


class EvaluateSingleQueryRequest(BaseModel):
    """单条查询评估请求。"""
    query: str = Field(..., min_length=1, max_length=2000, description="手动输入的测试查询")
    use_dense_only: bool = Field(default=False, description="True 时只走稠密向量检索")
    use_rerank: bool = Field(default=False, description="是否启用 BGE 重排序")
    rerank_top_n: int = Field(default=3, ge=1, le=10, description="重排序返回结果数")
    k: int = Field(default=5, ge=1, le=20, description="检索深度")


class BatchEvaluateRequest(BaseModel):
    """批量评估请求。"""
    tier: str = Field(default="all", description="测试层级: L1 | L2 | L3 | all")
    use_dense_only: bool = Field(default=False, description="True 时只走稠密向量检索")
    use_rerank: bool = Field(default=False, description="是否启用 BGE 重排序")
    rerank_top_n: int = Field(default=3, ge=1, le=10)
    k: int = Field(default=5, ge=1, le=20)
