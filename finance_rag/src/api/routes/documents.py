"""文档管理 API 路由（上传 / 列表 / 删除 / 知识库统计 / 任务状态）。"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from finance_rag.src.utils.audit import log as audit_log
from finance_rag.src.utils.metrics import document_ops
from finance_rag.src.api.dependencies import get_current_user
from finance_rag.src.application.document_service import get_document_manager
from finance_rag.src.application.task_service import get_task_manager, TaskStatus
from finance_rag.src.schemas.document import (
    UploadResponse,
    UploadFailure,
    BatchUploadResponse,
    AsyncUploadResponse,
    TaskStatusResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# --- 文档批量上传 ---

@router.post("/documents/upload", response_model=BatchUploadResponse)
async def upload_documents(
    current_user: Annotated[str, Depends(get_current_user)],
    files: list[UploadFile] = File(...),
):
    """上传文档到知识库（支持 .md / .txt / .pdf）。"""
    dm = get_document_manager()
    if not files:
        raise HTTPException(status_code=400, detail="No files were uploaded")

    successes = []
    failures = []
    try:
        results = await dm.upload_documents(files)
        for file, result in zip(files, results):
            if isinstance(result, BaseException):
                if not isinstance(result, ValueError):
                    logger.error(
                        "Document upload failed: %s",
                        result,
                        exc_info=(type(result), result, result.__traceback__),
                    )
                failures.append({
                    "filename": file.filename or "untitled",
                    "error": str(result),
                })
            else:
                successes.append(result)
    finally:
        for file in files:
            await file.close()

    document_ops.labels(operation="upload").inc(len(successes))
    if failures:
        document_ops.labels(operation="upload").inc(len(failures))

    if successes:
        audit_log(
            "upload",
            user=current_user,
            resource=",".join(s.get("filename", "") for s in successes[:5]),
            detail=f"成功 {len(successes)} 个文件",
        )
    if failures:
        audit_log(
            "upload",
            user=current_user,
            result="failure",
            detail=f"失败 {len(failures)} 个",
        )

    return BatchUploadResponse(
        total=len(files),
        success_count=len(successes),
        failure_count=len(failures),
        successes=[UploadResponse(**item) for item in successes],
        failures=[UploadFailure(**item) for item in failures],
    )


# --- 异步上传 ---

@router.post("/documents/upload-async", response_model=list[AsyncUploadResponse])
async def upload_documents_async(
    current_user: Annotated[str, Depends(get_current_user)],
    files: list[UploadFile] = File(...),
):
    """异步上传文档：立即返回 task_id，后台解析入库，前端轮询状态。"""
    dm = get_document_manager()
    if not files:
        raise HTTPException(status_code=400, detail="No files were uploaded")

    results = []
    try:
        for file in files:
            try:
                result = await dm.upload_document_async(file)
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
):
    """列出知识库所有文档。"""
    dm = get_document_manager()
    return dm.list_documents()


# --- 文档删除 ---

@router.delete("/documents/{source:path}")
async def delete_document(
    source: str,
    current_user: Annotated[str, Depends(get_current_user)],
):
    """删除指定文档的 Milvus 向量记录，保留本地原文件。"""
    dm = get_document_manager()
    try:
        result = dm.delete_document(source)
        document_ops.labels(operation="delete").inc()
        audit_log("delete", user=current_user, resource=source)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return result


# --- 知识库统计 ---

@router.get("/kb/stats")
async def kb_stats(
    current_user: Annotated[str, Depends(get_current_user)],
):
    """知识库统计信息（文档数/块数/集合状态）。"""
    dm = get_document_manager()
    return dm.get_stats()
