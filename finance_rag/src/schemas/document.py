"""文档管理相关的 Pydantic 请求/响应模型。"""

from __future__ import annotations

from pydantic import BaseModel


class AsyncUploadResponse(BaseModel):
    task_id: str
    filename: str


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    filename: str
    result: dict | None = None
    error: str | None = None
