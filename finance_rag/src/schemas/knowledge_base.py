"""知识库管理相关的 Pydantic 请求/响应模型。

知识库 = 逻辑类别视图：所有类别共用同一个 Milvus 集合（finance_kb），
类别值写入文档的 ``category`` 标量字段。
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class KnowledgeBaseCreate(BaseModel):
    """创建知识库类别请求。"""

    name: str = Field(
        ...,
        description="类别值（中文/字母/数字/下划线/连字符，1-32 位），写入文档分类字段",
    )
    display_name: str = ""
    description: str = ""


class KnowledgeBaseUpdate(BaseModel):
    """更新知识库类别请求。

    ``name`` 为可选的新类别值：传入且与当前值不同时执行改名，
    服务端会联动迁移集合内该类别下所有文档的分类字段。
    """

    name: str | None = Field(default=None, description="新类别值（可选，改名时传入）")
    display_name: str | None = None
    description: str | None = None


class KnowledgeBaseInfo(BaseModel):
    """知识库类别信息。"""

    name: str
    display_name: str
    description: str
    created_at: str
    document_count: int
    chunk_count: int
