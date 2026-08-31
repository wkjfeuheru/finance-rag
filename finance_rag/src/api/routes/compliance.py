"""合规审查 API 路由（问答 + 文档审查，非流式）。"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from finance_rag.src.api.dependencies import get_current_user
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


@router.post("/document-review-upload", response_model=DocumentReviewResponse)
async def document_review_upload(
    current_user: Annotated[str, Depends(get_current_user)],
    file: UploadFile = File(...),
):
    """文档合规审查（上传文件）：上传 .md/.txt/.pdf/.docx，解析后做合规审查。"""
    try:
        data = await file.read()
        text = extract_upload_text(file.filename or "upload.txt", data)
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
