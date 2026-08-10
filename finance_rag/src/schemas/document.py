"""文档管理相关的 Pydantic 请求/响应模型。"""

from __future__ import annotations

from pydantic import BaseModel


class UploadResponse(BaseModel):
    filename: str
    source: str
    title: str
    chunk_count: int
    parent_count: int
    size_mb: float


class UploadFailure(BaseModel):
    filename: str
    error: str


class BatchUploadResponse(BaseModel):
    total: int
    success_count: int
    failure_count: int
    successes: list[UploadResponse]
    failures: list[UploadFailure]


class AsyncUploadResponse(BaseModel):
    task_id: str
    filename: str


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    filename: str
    result: dict | None = None
    error: str | None = None
