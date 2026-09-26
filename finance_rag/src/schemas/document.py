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
    # 入库流水线阶段进度（parsed / chunked / embedded / written）
    progress: dict | None = None


class DocumentMetadataPatch(BaseModel):
    """研报元数据人工修正请求（只带需要改的字段；空串表示清空）。

    行业取值必须是申万标准名（如「食品饮料」「白酒Ⅱ」），
    取值校验在服务层统一执行，非法值会被拒绝而不是静默写空。
    """

    security_code: str | None = None
    security_name: str | None = None
    industry_l1: str | None = None
    industry_l2: str | None = None
    report_type: str | None = None
    broker: str | None = None
