"""合规审查相关的 Pydantic 请求/响应模型。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from finance_rag.src.schemas.chat import SourceInfo


class ComplianceReviewRequest(BaseModel):
    query: str
    history: list[dict] = Field(default_factory=list)
    use_rerank: bool = True
    k: int | None = Field(default=None, ge=1, le=20)
    rerank_top_n: int | None = Field(default=None, ge=1, le=10)
    filters: dict | None = Field(default=None, description="元数据过滤条件")


class EvidenceLocation(BaseModel):
    document_id: str
    segment_id: str
    quote: str
    char_start: int | None = None
    char_end: int | None = None
    line_start: int | None = None
    line_end: int | None = None
    page: int | None = None


class DocumentEvidence(BaseModel):
    location: EvidenceLocation
    matched_keywords: list[str] = Field(default_factory=list)
    match_spans: list[tuple[int, int]] = Field(default_factory=list)
    rule_ids: list[str] = Field(default_factory=list)
    validation_status: str = "verified"


class RegulationEvidence(BaseModel):
    evidence_id: str
    source_id: str
    chunk_id: str | None = None
    title: str = ""
    source: str = ""
    clause: str | None = None
    quote: str
    query: str = ""
    score: float = 0.0
    validation_status: str = "unverified"
    validation_issues: list[str] = Field(default_factory=list)


class ComplianceFinding(BaseModel):
    finding_id: str
    segment_id: str
    status: str
    risk_level: str
    summary: str
    reason: str
    suggestions: list[str] = Field(default_factory=list)
    rule_refs: list[str] = Field(default_factory=list)
    document_evidence: list[DocumentEvidence] = Field(default_factory=list)
    regulation_evidence: list[RegulationEvidence] = Field(default_factory=list)
    evidence_status: str = "insufficient"
    confidence: float | None = None


class ComplianceReport(BaseModel):
    report_version: int = 2
    review_id: str
    input_type: str
    risk_level: str | None = None
    action: str | None = None
    summary: str = ""
    reason: str | None = None
    suggestions: list[str] = Field(default_factory=list)
    findings: list[ComplianceFinding] = Field(default_factory=list)
    regulation_evidence: list[RegulationEvidence] = Field(default_factory=list)
    evidence_coverage: float = 0.0
    audit: dict[str, Any] = Field(default_factory=dict)


class ComplianceReviewResponse(BaseModel):
    answer: str
    sources: list[SourceInfo] = Field(default_factory=list)
    red_lines: list[dict] = Field(default_factory=list)
    risk_level: str | None = None
    action: str | None = None
    reason: str | None = None
    suggestions: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    citation_validation: dict | None = None
    clause_validation: dict | None = None
    answer_rejected: bool = False
    low_confidence: bool = False
    report_version: int = 2
    review_id: str | None = None
    findings: list[ComplianceFinding] = Field(default_factory=list)
    regulation_evidence: list[RegulationEvidence] = Field(default_factory=list)
    evidence_coverage: float = 0.0
    audit: dict = Field(default_factory=dict)


class DocumentReviewRequest(BaseModel):
    content: str = Field(..., description="待审文档正文")
    use_rerank: bool = True
    k: int | None = Field(default=None, ge=1, le=20)
    rerank_top_n: int | None = Field(default=None, ge=1, le=10)


class FlaggedItem(BaseModel):
    segment: str
    rules: list[dict]


class DocumentReviewResponse(BaseModel):
    flagged_items: list[FlaggedItem] = Field(default_factory=list)
    red_lines: list[dict] = Field(default_factory=list)
    assessment: str
    sources: list[SourceInfo] = Field(default_factory=list)
    risk_level: str | None = None
    action: str | None = None
    reason: str | None = None
    suggestions: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    report_version: int = 2
    review_id: str | None = None
    findings: list[ComplianceFinding] = Field(default_factory=list)
    regulation_evidence: list[RegulationEvidence] = Field(default_factory=list)
    evidence_coverage: float = 0.0
    audit: dict = Field(default_factory=dict)
