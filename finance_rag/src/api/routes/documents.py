"""文档管理 API 路由（上传 / 列表 / 删除 / 知识库统计 / 任务状态）。"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile

from finance_rag.src.utils.audit import log as audit_log
from finance_rag.src.utils.metrics import document_ops
from finance_rag.src.api.dependencies import get_current_user
from finance_rag.src.services.document_service import get_document_manager
from finance_rag.src.services.task_service import get_task_manager
from finance_rag.src.schemas.document import (
    AsyncUploadResponse,
    TaskStatusResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# --- 文档批量上传（异步：立即返回 task_id，后台解析入库） ---

@router.post("/documents/upload", response_model=list[AsyncUploadResponse])
async def upload_documents(
    current_user: Annotated[str, Depends(get_current_user)],
    files: list[UploadFile] = File(...),
    category: str = Form(""),
    kb: str = Form(""),
):
    """上传文档到知识库（支持 .md / .txt / .pdf）。

    异步处理：立即返回 task_id 列表，前端轮询 /tasks/{task_id} 查询进度。
    category 为可选分类（investment_research/compliance_risk/business_operations/management）。
    kb 为目标知识库集合名（空 = 默认知识库）。
    """
    dm = get_document_manager(kb)
    if not files:
        raise HTTPException(status_code=400, detail="No files were uploaded")

    try:
        results = await dm.upload_documents_async(files, category=category)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("异步上传提交失败：%s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        for file in files:
            await file.close()

    document_ops.labels(operation="upload").inc(len(results))
    audit_log(
        "upload",
        user=current_user,
        resource=",".join(r.get("filename", "") for r in results),
        detail=f"异步提交 {len(results)} 个文件（category={category or '未分类'}）",
    )
    return [AsyncUploadResponse(**item) for item in results]


# --- 异步上传（别名，与 /documents/upload 行为一致） ---

@router.post("/documents/upload-async", response_model=list[AsyncUploadResponse])
async def upload_documents_async(
    current_user: Annotated[str, Depends(get_current_user)],
    files: list[UploadFile] = File(...),
    category: str = Form(""),
    kb: str = Form(""),
):
    """异步上传文档：立即返回 task_id，后台解析入库，前端轮询状态。"""
    dm = get_document_manager(kb)
    if not files:
        raise HTTPException(status_code=400, detail="No files were uploaded")

    results = []
    try:
        for file in files:
            try:
                result = await dm.upload_document_async(file, category=category)
                results.append(AsyncUploadResponse(**result))
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc))
            except Exception as exc:
                raise HTTPException(status_code=500, detail=str(exc))
    finally:
        for file in files:
            await file.close()

    audit_log(
        "upload_async",
        user=current_user,
        resource=",".join(r.filename for r in results),
        detail=f"异步提交 {len(results)} 个文件",
    )
    return results


# --- 任务状态查询 ---

@router.get("/tasks/{task_id}")
async def get_task_status(
    task_id: str,
    current_user: Annotated[str, Depends(get_current_user)],
):
    """查询异步上传任务的状态。"""
    tm = get_task_manager()
    task = tm.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    return TaskStatusResponse(
        task_id=task.id,
        status=task.status.value,
        filename=task.filename,
        result=task.result,
        error=task.error,
    )


# --- 文档列表 ---

@router.get("/documents")
async def list_documents(
    current_user: Annotated[str, Depends(get_current_user)],
    kb: str = Query(""),
    include_versions: bool = Query(
        False, description="true 时返回各文档的历史版本明细"
    ),
):
    """列出指定知识库所有文档（kb 为空 = 默认知识库）。"""
    dm = get_document_manager(kb)
    return dm.list_documents(include_versions=include_versions)


# --- 文档删除 ---

@router.delete("/documents/{source:path}")
async def delete_document(
    source: str,
    current_user: Annotated[str, Depends(get_current_user)],
    kb: str = Query(""),
    version: str = Query("", description="指定删除的版本号；空 = 删除全部版本"),
):
    """删除指定文档的 Milvus 向量记录，保留本地原文件。

    ``version`` 为空时删除该文档的所有版本；指定时仅删除对应版本。
    """
    dm = get_document_manager(kb)
    try:
        result = dm.delete_document(source, version=version or None)
        document_ops.labels(operation="delete").inc()
        audit_log("delete", user=current_user, resource=source)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return result


# --- 知识库统计 ---

@router.get("/kb/stats")
async def kb_stats(
    current_user: Annotated[str, Depends(get_current_user)],
    kb: str = Query(""),
):
    """知识库统计信息（文档数/块数/集合状态）。"""
    dm = get_document_manager(kb)
    return dm.get_stats()
