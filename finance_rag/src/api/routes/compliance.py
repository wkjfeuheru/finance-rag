"""合规审查 API 路由（问答 + 文档审查，非流式）。"""

from __future__ import annotations

import asyncio
import logging
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from finance_rag.src.api.dependencies import get_current_user
from finance_rag.src.core.config import PDF_PARSE_TIMEOUT_SECONDS
from finance_rag.src.services.task_service import TaskStatus, get_task_manager
from finance_rag.src.services.compliance_service import (
    compliance_document_review,
    compliance_review,
    extract_upload_text,
)
from finance_rag.src.schemas.chat import SourceInfo
from finance_rag.src.schemas.compliance import (
    ComplianceFinding,
    ComplianceReviewRequest,
    ComplianceReviewResponse,
    DocumentReviewRequest,
    DocumentReviewResponse,
    FlaggedItem,
    RegulationEvidence,
)

logger = logging.getLogger(__name__)

router = APIRouter()
_background_review_tasks: set[asyncio.Task] = set()


@router.post("/review", response_model=ComplianceReviewResponse)
async def review(
    req: ComplianceReviewRequest,
    current_user: Annotated[str, Depends(get_current_user)],
):
    """合规问答：判断用户描述的业务/行为是否合规，给出法规依据与审计留痕。"""
    history = req.history or []
    try:
        result = await compliance_review(
            req.query,
            history,
            use_rerank=req.use_rerank,
            k=req.k,
            rerank_top_n=req.rerank_top_n,
            filters=req.filters,
        )
    except Exception as exc:
        from finance_rag.src.core.exceptions import friendly_message

        logger.exception("合规问答异常：%s", exc)
        raise HTTPException(status_code=500, detail=friendly_message(exc))

    return ComplianceReviewResponse(
        answer=result["answer"],
        sources=[SourceInfo(**s) for s in result["sources"]],
        red_lines=result["red_lines"],
        risk_level=result.get("risk_level"),
        action=result.get("action"),
        reason=result.get("reason"),
        suggestions=result.get("suggestions", []),
        evidence=result.get("evidence", []),
        citation_validation=result.get("citation_validation"),
        clause_validation=result.get("clause_validation"),
        answer_rejected=bool(result.get("answer_rejected", False)),
        low_confidence=bool(result.get("low_confidence", False)),
        report_version=result.get("report_version", 2),
        review_id=result.get("review_id"),
        findings=[ComplianceFinding(**f) for f in result.get("findings", [])],
        regulation_evidence=[RegulationEvidence(**e) for e in result.get("regulation_evidence", [])],
        evidence_coverage=result.get("evidence_coverage", 0.0),
        audit=result["audit"],
    )


@router.post("/document-review", response_model=DocumentReviewResponse)
async def document_review(
    req: DocumentReviewRequest,
    current_user: Annotated[str, Depends(get_current_user)],
):
    """文档合规审查：对待审文档做红线匹配 + 检索法规 + 总体评估。"""
    try:
        result = await compliance_document_review(
            req.content,
            use_rerank=req.use_rerank,
            k=req.k,
            rerank_top_n=req.rerank_top_n,
        )
    except Exception as exc:
        from finance_rag.src.core.exceptions import friendly_message

        logger.exception("文档合规审查异常：%s", exc)
        raise HTTPException(status_code=500, detail=friendly_message(exc))

    return DocumentReviewResponse(
        flagged_items=[FlaggedItem(**item) for item in result["flagged_items"]],
        red_lines=result["red_lines"],
        assessment=result["assessment"],
        sources=[SourceInfo(**s) for s in result["sources"]],
        risk_level=result.get("risk_level"),
        action=result.get("action"),
        reason=result.get("reason"),
        suggestions=result.get("suggestions", []),
        evidence=result.get("evidence", []),
        report_version=result.get("report_version", 2),
        review_id=result.get("review_id"),
        findings=[ComplianceFinding(**f) for f in result.get("findings", [])],
        regulation_evidence=[RegulationEvidence(**e) for e in result.get("regulation_evidence", [])],
        evidence_coverage=result.get("evidence_coverage", 0.0),
        audit=result["audit"],
    )


async def _run_document_review_task(task_id: str, filename: str, data: bytes) -> None:
    tm = get_task_manager()
    tm.update(task_id, status=TaskStatus.PROCESSING)
    try:
        tm.update(task_id, status=TaskStatus.PROCESSING, result={"stage": "parsing"})
        text = await asyncio.wait_for(
            asyncio.to_thread(extract_upload_text, filename, data),
            timeout=PDF_PARSE_TIMEOUT_SECONDS,
        )
        if not text or not text.strip():
            raise ValueError("文档内容为空，无法审查")
        tm.update(task_id, status=TaskStatus.PROCESSING, result={"stage": "reviewing"})
        result = await asyncio.wait_for(
            compliance_document_review(text),
            timeout=PDF_PARSE_TIMEOUT_SECONDS,
        )
        tm.update(task_id, status=TaskStatus.COMPLETED, result={"stage": "completed", "report": result})
    except asyncio.TimeoutError:
        message = f"文档审查超时（超过 {PDF_PARSE_TIMEOUT_SECONDS:g} 秒），请缩短文档或稍后重试"
        logger.error("异步合规文档审查超时：%s", task_id)
        tm.update(task_id, status=TaskStatus.FAILED, error=message)
    except asyncio.CancelledError:
        tm.update(task_id, status=TaskStatus.FAILED, error="文档审查任务被取消")
        raise
    except Exception as exc:
        logger.exception("异步合规文档审查失败：%s", exc)
        tm.update(task_id, status=TaskStatus.FAILED, error=str(exc))


def _retain_review_task(task_id: str, task: asyncio.Task) -> None:
    """保存后台任务强引用，结束后释放，避免任务状态永久停在 processing。"""
    _background_review_tasks.add(task)

    def _on_done(done: asyncio.Task) -> None:
        _background_review_tasks.discard(done)
        if done.cancelled():
            get_task_manager().update(
                task_id, status=TaskStatus.FAILED, error="文档审查任务被取消"
            )

    task.add_done_callback(_on_done)


@router.post("/document-review-upload-async")
async def document_review_upload_async(
    current_user: Annotated[str, Depends(get_current_user)],
    file: UploadFile = File(...),
):
    """异步合规文件审查：立即返回 task_id，前端轮询 /api/tasks/{task_id}。"""
    try:
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="上传文件为空")
        task = get_task_manager().create(file.filename or "upload.txt")
        review_task = asyncio.create_task(
            _run_document_review_task(task.id, file.filename or "upload.txt", data)
        )
        _retain_review_task(task.id, review_task)
        return {"task_id": task.id, "status": task.status.value, "filename": task.filename}
    finally:
        await file.close()


@router.post("/document-review-upload", response_model=DocumentReviewResponse)
async def document_review_upload(
    current_user: Annotated[str, Depends(get_current_user)],
    file: UploadFile = File(...),
):
    """文档合规审查（上传文件）：上传 .md/.txt/.pdf/.docx，解析后做合规审查。

    注意：解析放入线程池执行，避免 PDF 解析阻塞事件循环导致全站无响应。
    """
    try:
        data = await file.read()
        text = await asyncio.wait_for(
            asyncio.to_thread(extract_upload_text, file.filename or "upload.txt", data),
            timeout=PDF_PARSE_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=408,
            detail=f"文档解析超时（超过 {PDF_PARSE_TIMEOUT_SECONDS:g} 秒），"
            "请改用异步审查接口或缩短文档",
        )
    except Exception as exc:
        from finance_rag.src.core.exceptions import friendly_message

        logger.exception("上传文档解析失败：%s", exc)
        raise HTTPException(status_code=400, detail=f"文档解析失败：{friendly_message(exc)}")
    finally:
        await file.close()

    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="文档内容为空，无法审查")

    try:
        result = await compliance_document_review(text)
    except Exception as exc:
        from finance_rag.src.core.exceptions import friendly_message

        logger.exception("上传文档合规审查异常：%s", exc)
        raise HTTPException(status_code=500, detail=friendly_message(exc))

    return DocumentReviewResponse(
        flagged_items=[FlaggedItem(**item) for item in result["flagged_items"]],
        red_lines=result["red_lines"],
        assessment=result["assessment"],
        sources=[SourceInfo(**s) for s in result["sources"]],
        risk_level=result.get("risk_level"),
        action=result.get("action"),
        reason=result.get("reason"),
        suggestions=result.get("suggestions", []),
        evidence=result.get("evidence", []),
        report_version=result.get("report_version", 2),
        review_id=result.get("review_id"),
        findings=[ComplianceFinding(**f) for f in result.get("findings", [])],
        regulation_evidence=[RegulationEvidence(**e) for e in result.get("regulation_evidence", [])],
        evidence_coverage=result.get("evidence_coverage", 0.0),
        audit=result["audit"],
    )
