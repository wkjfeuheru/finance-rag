"""文档管理 API 路由（上传 / 列表 / 删除 / 知识库统计 / 任务状态）。"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Response,
    UploadFile,
)

from finance_rag.src.core.config import PAGE_RENDER_DPI
from finance_rag.src.core.exceptions import IngestionRejectedError
from finance_rag.src.infrastructure.storage import get_storage
from finance_rag.src.utils.audit import log as audit_log
from finance_rag.src.utils.metrics import document_ops
from finance_rag.src.api.dependencies import get_current_user
from finance_rag.src.services.document_service import get_document_manager
from finance_rag.src.services.task_service import get_task_manager
from finance_rag.src.schemas.document import (
    AsyncUploadResponse,
    DocumentMetadataPatch,
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
    except IngestionRejectedError as exc:
        # 入库流水线背压（队列满）或正在关闭：属于「暂时不可用」，让客户端稍后重试。
        # 已入队的文件在重试时会命中内容指纹而被增量跳过，不会重复入库。
        logger.warning("异步上传被流水线拒绝：%s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
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
            except IngestionRejectedError as exc:
                # 队列满 / 流水线关闭：整批返回 503 让客户端退避重试；
                # 已入队文件在重试时命中内容指纹被增量跳过，不会重复入库。
                logger.warning("异步上传被流水线拒绝：%s", exc)
                raise HTTPException(status_code=503, detail=str(exc)) from exc
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
        progress=task.progress,
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


# --- 研报元数据人工修正 ---

@router.patch("/documents/{source:path}/metadata")
async def patch_document_metadata(
    source: str,
    payload: DocumentMetadataPatch,
    current_user: Annotated[str, Depends(get_current_user)],
    kb: str = Query(""),
):
    """人工修正研报元数据（不重新嵌入）。

    抽取链路的最后一道兜底：正则与模型都会抽错，抽错又没有任何补救手段，
    元数据维度就形同虚设。取值非法返回 422 而不是静默写空。
    """
    dm = get_document_manager(kb)
    fields = payload.model_dump(exclude_unset=True)
    if not fields:
        raise HTTPException(status_code=400, detail="未提供任何待更新字段")
    try:
        result = dm.update_document_metadata(source, fields)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("研报元数据更新失败：%s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if result.get("missing"):
        raise HTTPException(status_code=404, detail=f"未找到文档：{source}")
    audit_log(
        "update_metadata",
        user=current_user,
        resource=source,
        detail=f"字段：{', '.join(sorted(fields))}",
    )
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


# --- 原文页渲染（页码级溯源） ---

@router.get("/documents/{source:path}/page/{page}")
async def render_document_page(
    source: str,
    page: int,
    current_user: Annotated[str, Depends(get_current_user)],
):
    """把原文 PDF 的第 ``page`` 页（1-based）渲染成 PNG。

    分析师验证一条结论的成本必须足够低：拿到页码就能直接看到那一页。
    这里按需渲染、不落临时文件（PDF 字节直接从存储后端读入内存）。
    """
    if page < 1:
        raise HTTPException(status_code=400, detail="页码从 1 开始")

    storage = get_storage()
    try:
        payload = await storage.download(f"docs/{source}")
    except Exception as exc:  # noqa: BLE001 - 存储后端异常统一按「找不到原文」处理
        logger.warning("读取原文失败（%s）：%s", source, exc)
        raise HTTPException(status_code=404, detail=f"未找到原文：{source}") from exc

    try:
        import pymupdf

        document = pymupdf.open(stream=payload, filetype="pdf")
    except Exception as exc:  # noqa: BLE001 - 非 PDF（如 .md）不能渲染
        logger.info("原文不是可渲染的 PDF（%s）：%s", source, exc)
        raise HTTPException(status_code=400, detail="原文不是可渲染的 PDF") from exc

    try:
        if page > document.page_count:
            raise HTTPException(
                status_code=404,
                detail=f"页码超出范围（共 {document.page_count} 页）",
            )
        pixmap = document[page - 1].get_pixmap(dpi=PAGE_RENDER_DPI)
        return Response(content=pixmap.tobytes("png"), media_type="image/png")
    finally:
        document.close()
