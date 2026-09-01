"""合规审查 Agent 服务层（独立链路，不影响现有通用问答）。

提供两类能力：
* :func:`compliance_review` —— 合规问答：检索法规 → 红线匹配 → LLM 合规判断 →
  引用校验 → 审计留痕；
* :func:`compliance_document_review` —— 文档审查：分段红线匹配 + 检索法规 +
  LLM 总体评估。

设计原则：**先规则（确定性红线）、后模型（LLM 判断）、引用校验兜底拒答**，
每一步产出审计信息，保证合规结论可回溯。
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import uuid
from pathlib import Path
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from finance_rag.src.core.config import COMPLIANCE_CATEGORIES, get_model
from finance_rag.src.agent.prompts.chat import (
    COMPLIANCE_DOC_REVIEW_PROMPT,
    COMPLIANCE_JUDGMENT_PROMPT,
)
from finance_rag.src.services.compliance_rules import match_red_lines

logger = logging.getLogger(__name__)

# 文档审查时用于检索法规的查询长度上限（取待审文档开头作为检索查询）
_DOC_QUERY_MAX_CHARS = 300
# 文档审查分段多查询的段数上限：优先检索命中红线的分段，超过则截断
_DOC_SEGMENT_QUERY_LIMIT = 5

# ---------------------------------------------------------------------------
# 三级风险结论模型（对齐文档第三部分「证据化审核」的 [通过]/[需人工审核]/[驳回]）
# ---------------------------------------------------------------------------

RISK_LEVEL_SAFE = "safe"  # 安全/通过
RISK_LEVEL_CONTROVERSIAL = "controversial"  # 争议性/需人工审核
RISK_LEVEL_UNSAFE = "unsafe"  # 不安全/驳回

# 风险等级 -> 处置动作映射
_RISK_TO_ACTION = {
    RISK_LEVEL_SAFE: "pass",
    RISK_LEVEL_CONTROVERSIAL: "manual_review",
    RISK_LEVEL_UNSAFE: "reject",
}

# 结构化审核报告的固定标签（用于解析 LLM 输出）
_REPORT_TAG_RISK = "风险判定"
_REPORT_TAG_REASON = "理由"
_REPORT_TAG_SUGGESTION = "修改建议"
_REPORT_TAG_EVIDENCE = "引用条款"


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _format_red_lines(red_lines: list[dict[str, Any]]) -> str:
    """将命中红线规则列表格式化为 Prompt 文本。"""
    if not red_lines:
        return "（无）"
    return "\n".join(
        f"- {r['rule_id']}：{r['behavior']}（依据：{r['legal_basis']}）"
        for r in red_lines
    )


def _parse_conclusion(answer: str) -> bool | None:
    """从旧版合规判断回答中解析结论（向后兼容回退，不再是主路径）。

    Returns:
        ``True``（合规）/ ``False``（不合规）/ ``None``（无法判断或解析失败）。
    """
    match = re.search(r"结论[：:]\s*(合规|不合规|无法判断)", answer or "")
    if not match:
        return None
    label = match.group(1)
    return {"合规": True, "不合规": False, "无法判断": None}[label]


# 结构化报告中风险标签 -> 三级风险枚举的归一化映射
# 顺序即匹配优先级：不安全类必须先于安全类判断，否则「不安全」会被「安全」子串误匹配
_RISK_LABEL_ORDER = (
    ("unsafe", RISK_LEVEL_UNSAFE),
    ("不安全", RISK_LEVEL_UNSAFE),
    ("驳回", RISK_LEVEL_UNSAFE),
    ("controversial", RISK_LEVEL_CONTROVERSIAL),
    ("争议性", RISK_LEVEL_CONTROVERSIAL),
    ("争议", RISK_LEVEL_CONTROVERSIAL),
    ("需人工审核", RISK_LEVEL_CONTROVERSIAL),
    ("需人工", RISK_LEVEL_CONTROVERSIAL),
    ("safe", RISK_LEVEL_SAFE),
    ("安全", RISK_LEVEL_SAFE),
    ("通过", RISK_LEVEL_SAFE),
)


def _risk_to_action(risk_level: str | None) -> str | None:
    """三级风险等级 -> 处置动作（pass / manual_review / reject）。"""
    if risk_level is None:
        return None
    return _RISK_TO_ACTION.get(risk_level)


def _parse_tag_block(text: str, tag: str) -> str:
    """从结构化报告文本中抽取指定标签后的正文（取该标签到下一标签之间）。"""
    labels = [
        _REPORT_TAG_RISK,
        _REPORT_TAG_REASON,
        _REPORT_TAG_SUGGESTION,
        _REPORT_TAG_EVIDENCE,
    ]
    pattern = re.compile(
        rf"{re.escape(tag)}\s*[：:]\s*(?P<body>.*?)(?=(?:"
        + "|".join(re.escape(l) for l in labels)
        + r")\s*[：:]|$)",
        re.DOTALL,
    )
    match = pattern.search(text or "")
    return match.group("body").strip() if match else ""


def _parse_risk_label(label_text: str) -> str | None:
    """把 LLM 输出的风险判定文本归一化为三级风险枚举，无法识别返回 None。"""
    label = (label_text or "").strip()
    if not label:
        return None
    for key, value in _RISK_LABEL_ORDER:
        if key in label:
            return value
    return None


def _parse_audit_report(answer: str) -> dict[str, Any]:
    """解析 LLM 输出的结构化审核报告，产出结构化字段。

    按固定标签（风险判定/理由/修改建议/引用条款）切分正文；解析失败时回退到
    旧版 ``_parse_conclusion``，并将 ``risk_level`` 置为 ``None``。

    Returns:
        ``{"risk_level", "reason", "suggestions", "evidence"}``。
    """
    risk_label = _parse_tag_block(answer, _REPORT_TAG_RISK)
    risk_level = _parse_risk_label(risk_label)

    # 结构化标签解析失败时，回退到旧版「结论：合规/不合规/无法判断」
    if risk_level is None:
        legacy = _parse_conclusion(answer)
        risk_level = (
            RISK_LEVEL_SAFE if legacy is True
            else RISK_LEVEL_UNSAFE if legacy is False
            else RISK_LEVEL_CONTROVERSIAL
        )

    reason = _parse_tag_block(answer, _REPORT_TAG_REASON) or answer
    suggestion_text = _parse_tag_block(answer, _REPORT_TAG_SUGGESTION)
    evidence_text = _parse_tag_block(answer, _REPORT_TAG_EVIDENCE)

    # 修改建议按顿号/换行拆分为列表，「无」或空则视为无建议
    suggestions: list[str] = []
    if suggestion_text and suggestion_text not in ("无", "无需", "无修改建议"):
        suggestions = [
            s.strip()
            for s in re.split(r"[、；;\n]", suggestion_text)
            if s.strip()
        ]

    evidence: list[str] = []
    if evidence_text:
        evidence = [s.strip() for s in re.split(r"[、；;\n]", evidence_text) if s.strip()]

    return {
        "risk_level": risk_level,
        "reason": reason,
        "suggestions": suggestions,
        "evidence": evidence,
    }


def _query_understanding(
    text: str,
    red_lines: list[dict[str, Any]] | None = None,
    keyword_source: str | None = None,
) -> tuple[str, list[str]]:
    """查询理解与路由：从待审文本生成「语义查询 + 关键词查询」。

    关键词来源：
    1. 命中红线规则中的敏感关键词（确定性来源）；
    2. 文本基础中文切词/长词抽取（简单启发式，可降级为空）。

    Args:
        text: 语义查询文本。
        red_lines: 命中的红线规则列表（用于提取确定性敏感词）。
        keyword_source: 关键词匹配的来源文本；默认取 ``text``。文档审查场景
            传全文，使红线关键词覆盖整篇文档而非仅语义查询片段。

    Returns:
        ``(semantic_query, keywords)``；关键词为空时检索链路自动降级为纯语义检索。
    """
    semantic_query = (text or "").strip()
    # 关键词匹配用的语料：默认为语义查询本身，可覆盖为全文以覆盖中后段内容
    source = keyword_source if keyword_source is not None else semantic_query

    keywords: list[str] = []
    # 来源一：命中红线规则的敏感关键词（在来源语料中实际出现才纳入）
    for rule in red_lines or []:
        for kw in rule.get("keywords", ()) or ():
            if kw and kw in source and kw not in keywords:
                keywords.append(kw)

    # 来源二：简单长词抽取（连续中文 4 字及以上的片段，作为候选关键词）
    if not keywords:
        for match in re.findall(r"[\u4e00-\u9fff]{4,}", source):
            if match not in keywords:
                keywords.append(match)

    # 关键词数量上限，避免噪声过多
    return semantic_query, keywords[:5]


def _split_segments(text: str) -> list[str]:
    """将待审文档按行切分为审查片段（空行丢弃）。"""
    return [line.strip() for line in (text or "").splitlines() if line.strip()]


def _select_query_segments(
    content: str,
    flagged_items: list[dict[str, Any]],
    *,
    limit: int = _DOC_SEGMENT_QUERY_LIMIT,
) -> list[str]:
    """选择用于检索法规的查询片段（文档审查）。

    优先取命中红线的片段——违规风险最高、最需要法规证据；命中片段为空时
    退化为文档开头截断。返回片段数量不超过 ``limit``，避免长文档产生过多检索。
    """
    flagged_segments = [
        item["segment"] for item in flagged_items if item.get("segment")
    ]
    if flagged_segments:
        return flagged_segments[:limit]
    return [(content.strip())[:_DOC_QUERY_MAX_CHARS]]


def _merge_unique_docs(
    docs: list[dict[str, Any]],
    top_k: int | None,
) -> list[dict[str, Any]]:
    """合并多段检索结果：按 content 去重（保留高分），按 score 降序取 top_k。"""
    if not docs:
        return []
    # 以 content 前 200 字符为去重键，同键保留 score 更高的一条
    by_key: dict[str, dict[str, Any]] = {}
    for doc in docs:
        key = (doc.get("content", "") or "")[:200]
        prev = by_key.get(key)
        if prev is None or (doc.get("score", 0.0) or 0.0) > (prev.get("score", 0.0) or 0.0):
            by_key[key] = doc
    merged = list(by_key.values())
    merged.sort(key=lambda d: d.get("score", 0.0), reverse=True)
    if top_k:
        merged = merged[:top_k]
    return merged



def _source_id(source: dict[str, Any]) -> str:
    """为检索来源生成稳定 ID，避免报告只能依赖展示序号。"""
    raw = f"{source.get('source', '')}|{source.get('chunk', '')}|{source.get('title', '')}"
    return "src-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _build_regulation_evidence(
    sources: list[dict[str, Any]],
    query: str,
) -> list[dict[str, Any]]:
    """将检索来源转换为可被 finding 引用的证据目录。"""
    evidence: list[dict[str, Any]] = []
    for index, source in enumerate(sources, start=1):
        quote = str(source.get("content") or source.get("preview") or "").strip()
        if not quote:
            continue
        evidence.append({
            "evidence_id": f"ev-{index:04d}",
            "source_id": _source_id(source),
            "chunk_id": str(source.get("chunk")) if source.get("chunk") is not None else None,
            "title": source.get("title", ""),
            "source": source.get("source", ""),
            "clause": next(iter(re.findall(r"第[一二三四五六七八九十百千万零两0-9]+条", quote)), None),
            "quote": quote[:2000],
            "query": query,
            "score": float(source.get("rerank_score", source.get("score", 0.0)) or 0.0),
            "validation_status": "unverified",
            "validation_issues": [],
        })
    return evidence


def _document_evidence(content: str, segment: str, segment_id: str, rules: list[dict[str, Any]]) -> dict[str, Any]:
    """构建并校验待审原文证据，所有 quote 必须来自原文。"""
    start = content.find(segment)
    line_start = content.count("\n", 0, max(start, 0)) + 1 if start >= 0 else None
    keywords = [
        keyword for rule in rules for keyword in rule.get("keywords", [])
        if keyword and keyword in segment
    ]
    spans = [(match.start(), match.end()) for keyword in keywords for match in re.finditer(re.escape(keyword), segment)]
    return {
        "location": {
            "document_id": "review-document",
            "segment_id": segment_id,
            "quote": segment,
            "char_start": start if start >= 0 else None,
            "char_end": start + len(segment) if start >= 0 else None,
            "line_start": line_start,
            "line_end": line_start,
            "page": None,
        },
        "matched_keywords": list(dict.fromkeys(keywords)),
        "match_spans": spans,
        "rule_ids": [rule.get("rule_id", "") for rule in rules],
        "validation_status": "verified" if start >= 0 else "insufficient",
    }


def _build_findings(
    content: str,
    flagged_items: list[dict[str, Any]],
    regulation_evidence: list[dict[str, Any]],
    report: dict[str, Any],
) -> list[dict[str, Any]]:
    """将规则命中转换为逐点 finding，并绑定可追溯证据。"""
    findings: list[dict[str, Any]] = []
    for index, item in enumerate(flagged_items, start=1):
        segment = item.get("segment", "")
        rules = item.get("rules", [])
        rule_ids = [rule.get("rule_id", "") for rule in rules]
        doc_evidence = _document_evidence(content, segment, f"seg-{index:04d}", rules)
        linked = [
            dict(ev, validation_status="verified")
            for ev in regulation_evidence
            if any(
                (rule.get("legal_basis", "") and rule["legal_basis"] in (ev.get("title", "") + ev.get("quote", "")))
                or (rule.get("behavior", "") and rule["behavior"] in ev.get("quote", ""))
                for rule in rules
            )
        ]
        if not linked and regulation_evidence:
            linked = [dict(regulation_evidence[0], validation_status="unverified", validation_issues=["未能将法规来源确定绑定到该规则"]) ]
        complete = doc_evidence["validation_status"] == "verified" and bool(linked) and all(
            ev.get("validation_status") == "verified" for ev in linked
        )
        findings.append({
            "finding_id": f"finding-{index:04d}",
            "segment_id": f"seg-{index:04d}",
            "status": "confirmed" if complete else "suspected",
            "risk_level": "unsafe" if complete else "controversial",
            "summary": "；".join(rule.get("behavior", "疑似违规") for rule in rules),
            "reason": report.get("reason", "命中合规规则，需结合法规证据核验"),
            "suggestions": report.get("suggestions", []),
            "rule_refs": rule_ids,
            "document_evidence": [doc_evidence],
            "regulation_evidence": linked,
            "evidence_status": "verified" if complete else "insufficient",
            "confidence": 1.0 if complete else 0.5,
        })
    return findings


def _aggregate_findings(findings: list[dict[str, Any]], fallback: str | None) -> tuple[str | None, str | None]:
    """按证据完整性聚合总体风险，避免未经验证的 finding 直接驳回。"""
    if any(f["status"] == "confirmed" and f["risk_level"] == RISK_LEVEL_UNSAFE for f in findings):
        return RISK_LEVEL_UNSAFE, "reject"
    if findings:
        return RISK_LEVEL_CONTROVERSIAL, "manual_review"
    return fallback, _risk_to_action(fallback)


def _compliance_filters(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """构造合规审查的元数据过滤条件，强制限定检索类别。

    ``category`` 由 ``COMPLIANCE_CATEGORIES`` 决定（默认仅 ``compliance_risk``），
    其他显式过滤条件（如 date）在原样保留的基础上合并，保证合规检索不越界。
    """
    merged = dict(extra or {})
    merged["category"] = list(COMPLIANCE_CATEGORIES)
    return merged


def extract_upload_text(filename: str, data: bytes) -> str:
    """从上传文件字节提取纯文本。

    md/txt/markdown 直接按 UTF-8 解码；pdf/docx 等保存临时文件后走统一解析器
    （``get_parser().parse``）。解析失败时向上抛出，由调用方转换为友好错误。
    """
    ext = Path(filename).suffix.lower()
    if ext in (".md", ".txt", ".markdown"):
        return data.decode("utf-8", errors="replace")

    import tempfile

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        from finance_rag.src.rag.ingestion import get_parser

        return get_parser().parse(tmp_path).markdown
    finally:
        Path(tmp_path).unlink(missing_ok=True)


async def _generate(prompt_text: str, variables: dict[str, Any]) -> str:
    """用主模型执行一次同步式生成（放入线程池避免阻塞事件循环）。"""
    if get_model() is None:
        raise RuntimeError("LLM 未配置（缺少 DEEPSEEK_API_KEY）")
    chain = (
        ChatPromptTemplate.from_messages([("human", prompt_text)])
        | get_model()
        | StrOutputParser()
    )

    def _invoke() -> str:
        return str(chain.invoke(variables) or "").strip()

    return await asyncio.to_thread(_invoke)


def _apply_compliance_refusal(
    risk_level: str | None,
    generic_validation: dict[str, Any] | None,
    clause_validation: dict[str, Any],
    source_count: int,
    rerank_scores: list[float] | None,
) -> tuple[bool, bool]:
    """合规场景严格拒答：在通用拒答基础上追加条款级校验。

    规则（优先级从高到低）：
    * 通用拒答（空检索 / 相关性不足 / 引用低分）→ 拒答；
    * 给出明确结论（safe/unsafe）但未引用任何条款（且要求条款）→ 低置信；
    * 引用了条款但命中比例低于阈值 → 拒答（疑似编造条款）。

    ``controversial``（需人工审核）不触发条款级严格校验，避免误杀灰色地带。
    """
    from finance_rag.src.core.config import (
        COMPLIANCE_CLAUSE_MIN_SCORE,
        COMPLIANCE_REQUIRE_CLAUSE,
    )
    from finance_rag.src.services.citation_validator import apply_refusal_policy

    rejected, low_conf = apply_refusal_policy(
        generic_validation, source_count, rerank_scores=rerank_scores
    )
    if rejected:
        return True, False

    # 仅当给出明确结论（safe / unsafe）时才做条款级严格校验
    if risk_level in (RISK_LEVEL_SAFE, RISK_LEVEL_UNSAFE):
        total_clauses = int(clause_validation.get("total_clauses", 0) or 0)
        if COMPLIANCE_REQUIRE_CLAUSE and total_clauses == 0:
            return False, True
        score = float(clause_validation.get("score", 1.0))
        if total_clauses > 0 and score < COMPLIANCE_CLAUSE_MIN_SCORE:
            return True, False

    return rejected, low_conf


# ---------------------------------------------------------------------------
# 合规问答
# ---------------------------------------------------------------------------

async def compliance_review(
    query: str,
    history: list[dict[str, str]] | None = None,
    *,
    use_rerank: bool = True,
    k: int | None = None,
    rerank_top_n: int | None = None,
    filters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """合规问答：判断用户描述的「业务/行为」是否合规，并给出法规依据。

    Returns:
        含 ``answer`` / ``sources`` / ``red_lines`` / ``risk_level`` / ``action`` /
        ``reason`` / ``suggestions`` / ``evidence`` / ``citation_validation`` /
        ``answer_rejected`` / ``low_confidence`` / ``audit`` 的结构化结果。
    """
    from finance_rag.src.services.chat_service import (
        build_context,
        rerank_scores,
        retrieve_pipeline,
    )
    from finance_rag.src.services.citation_validator import (
        REFUSAL_ANSWER,
        run_citation_validation,
        validate_clause_citations,
    )

    audit_stages: list[str] = []

    # 1. 红线预检（对用户问题本身，识别其描述的违规行为）
    red_lines = match_red_lines(query)
    audit_stages.append("red_line_matching")

    # 2. 查询理解与路由：生成语义查询 + 关键词查询
    semantic_query, keywords = _query_understanding(query, red_lines)
    audit_stages.append("query_understanding")

    # 3. 检索相关法规条款（关闭元数据过滤推断，保持合规检索的确定性）
    audit_stages.append("retrieving")
    pipeline = await retrieve_pipeline(
        semantic_query, history,
        use_rerank=use_rerank,
        k=k,
        rerank_top_n=rerank_top_n,
        filters=_compliance_filters(filters),
        infer_filters=False,
        keywords=keywords,
        collection_names=COMPLIANCE_CATEGORIES,
    )
    docs = pipeline["docs"]

    if not docs:
        if pipeline.get("retrieval_error"):
            return {
                "report_version": 2,
                "review_id": str(uuid.uuid4()),
                "answer": pipeline["retrieval_error"],
                "sources": [],
                "red_lines": red_lines,
                "risk_level": None,
                "action": None,
                "reason": None,
                "suggestions": [],
                "evidence": [],
                "citation_validation": None,
                "clause_validation": None,
                "answer_rejected": False,
                "low_confidence": False,
                "findings": [],
                "regulation_evidence": [],
                "evidence_coverage": 0.0,
                "audit": {
                    "stages": audit_stages,
                    "red_lines": red_lines,
                    "query_understanding": {"semantic_query": semantic_query, "keywords": keywords},
                },
            }
        return {
            "report_version": 2,
            "review_id": str(uuid.uuid4()),
            "answer": REFUSAL_ANSWER,
            "sources": [],
            "red_lines": red_lines,
            "risk_level": None,
            "action": _risk_to_action(RISK_LEVEL_UNSAFE),
            "reason": None,
            "suggestions": [],
            "evidence": [],
            "citation_validation": None,
            "clause_validation": None,
            "answer_rejected": True,
            "low_confidence": False,
            "findings": [],
            "regulation_evidence": [],
            "evidence_coverage": 0.0,
            "audit": {
                "stages": audit_stages,
                "red_lines": red_lines,
                "query_understanding": {"semantic_query": semantic_query, "keywords": keywords},
            },
        }

    context_text, sources = build_context(docs)

    # 4. LLM 结构化审核报告
    audit_stages.append("judgment")
    answer = await _generate(
        COMPLIANCE_JUDGMENT_PROMPT,
        {
            "query": query,
            "red_lines": _format_red_lines(red_lines),
            "context": context_text,
        },
    )
    report = _parse_audit_report(answer)
    risk_level = report["risk_level"]
    action = _risk_to_action(risk_level)

    # 5. 引用校验 + 条款级校验 + 严格拒答兜底
    audit_stages.append("citation_validation")
    audit_stages.append("clause_validation")
    validation = run_citation_validation(answer, sources)
    clause_validation = validate_clause_citations(answer, sources)
    rejected, low_conf = _apply_compliance_refusal(
        risk_level,
        validation,
        clause_validation,
        len(sources),
        rerank_scores(sources),
    )
    if rejected:
        answer = REFUSAL_ANSWER

    # 6. 构建逐点 finding 与法规证据目录（文本审查将用户问题视为唯一审查单元）
    flagged_items: list[dict[str, Any]] = (
        [{"segment": query, "rules": red_lines}] if red_lines else []
    )
    regulation_evidence = _build_regulation_evidence(sources, semantic_query)
    findings = _build_findings(query, flagged_items, regulation_evidence, report)
    # 拒答时降低确定性：绝不把未经受信的候选升级为 confirmed
    if rejected:
        for finding in findings:
            finding["status"] = "suspected"
            finding["evidence_status"] = "insufficient"
        aggregate_risk, aggregate_action = None, _risk_to_action(RISK_LEVEL_UNSAFE)
    else:
        aggregate_risk, aggregate_action = _aggregate_findings(findings, report["risk_level"])
        risk_level = aggregate_risk if aggregate_risk is not None else None
        action = aggregate_action
    evidence_coverage = (
        sum(1 for f in findings if f["evidence_status"] == "verified") / len(findings)
        if findings else 1.0
    )

    return {
        "report_version": 2,
        "review_id": str(uuid.uuid4()),
        "answer": answer,
        "sources": sources,
        "red_lines": red_lines,
        "risk_level": risk_level,
        "action": action,
        "reason": report["reason"],
        "suggestions": report["suggestions"],
        "evidence": report["evidence"],
        "citation_validation": validation,
        "clause_validation": clause_validation,
        "answer_rejected": rejected,
        "low_confidence": low_conf,
        "findings": findings,
        "regulation_evidence": regulation_evidence,
        "evidence_coverage": round(evidence_coverage, 4),
        "audit": {
            "stages": audit_stages,
            "red_lines": red_lines,
            "query_understanding": {"semantic_query": semantic_query, "keywords": keywords},
            "retrieved_sources": [s["source"] for s in sources],
            "risk_level": risk_level,
            "action": action,
            "reason": report["reason"],
            "suggestions": report["suggestions"],
            "evidence": report["evidence"],
            "clause_validation": clause_validation,
            "answer_rejected": rejected,
            "low_confidence": low_conf,
            "finding_count": len(findings),
            "evidence_coverage": round(evidence_coverage, 4),
        },
    }


# ---------------------------------------------------------------------------
# 文档审查
# ---------------------------------------------------------------------------

async def compliance_document_review(
    content: str,
    *,
    use_rerank: bool = True,
    k: int | None = None,
    rerank_top_n: int | None = None,
) -> dict[str, Any]:
    """文档合规审查：对待审文档做分段红线匹配 + 查询理解 + 检索法规 + LLM 结构化评估。

    Returns:
        含 ``flagged_items`` / ``red_lines`` / ``assessment`` / ``sources`` /
        ``risk_level`` / ``action`` / ``reason`` / ``suggestions`` / ``evidence`` /
        ``audit`` 的结构化结果。
    """
    from finance_rag.src.services.chat_service import (
        build_context,
        retrieve_pipeline,
    )

    audit_stages: list[str] = ["segmenting", "red_line_matching"]

    if not content or not content.strip():
        return {
            "flagged_items": [],
            "red_lines": [],
            "assessment": "待审文档为空。",
            "sources": [],
            "risk_level": None,
            "action": None,
            "reason": None,
            "suggestions": [],
            "evidence": [],
            "audit": {"stages": audit_stages},
        }

    # 1. 分段红线匹配：定位每个片段命中的红线规则
    seen_rule_ids: set[str] = set()
    red_lines: list[dict[str, Any]] = []
    flagged_items: list[dict[str, Any]] = []
    for segment in _split_segments(content):
        segment_hits = match_red_lines(segment)
        if segment_hits:
            flagged_items.append({"segment": segment, "rules": segment_hits})
            for rule in segment_hits:
                if rule["rule_id"] not in seen_rule_ids:
                    seen_rule_ids.add(rule["rule_id"])
                    red_lines.append(rule)

    # 2. 查询理解与路由：从全文提取关键词（覆盖中后段），语义查询按片段拆分
    audit_stages.append("query_understanding")
    query_segments = _select_query_segments(content, flagged_items)

    # 3. 分段多查询检索：对每个片段分别检索法规，合并去重证据集
    audit_stages.append("retrieving")
    all_docs: list[dict[str, Any]] = []
    query_understanding: list[dict[str, Any]] = []
    effective_k: int | None = k
    for segment in query_segments:
        semantic_query, keywords = _query_understanding(
            segment, red_lines, keyword_source=content
        )
        query_understanding.append(
            {"semantic_query": semantic_query, "keywords": keywords}
        )
        pipeline = await retrieve_pipeline(
            semantic_query,
            use_rerank=use_rerank,
            k=k,
            rerank_top_n=rerank_top_n,
            filters=_compliance_filters(),
            infer_filters=False,
            keywords=keywords,
            collection_names=COMPLIANCE_CATEGORIES,
        )
        all_docs.extend(pipeline["docs"])
        effective_k = pipeline["k"]

    docs = _merge_unique_docs(all_docs, effective_k)
    context_text, sources = build_context(docs)

    # 4. LLM 结构化评估报告
    audit_stages.append("assessment")
    assessment = await _generate(
        COMPLIANCE_DOC_REVIEW_PROMPT,
        {
            "red_lines": _format_red_lines(red_lines),
            "content": content,
            "context": context_text,
        },
    )
    report = _parse_audit_report(assessment)
    regulation_evidence = _build_regulation_evidence(sources, "；".join(query_segments))
    findings = _build_findings(content, flagged_items, regulation_evidence, report)
    aggregate_risk, aggregate_action = _aggregate_findings(findings, report["risk_level"])
    evidence_coverage = (
        sum(1 for finding in findings if finding["evidence_status"] == "verified") / len(findings)
        if findings else 1.0
    )
    risk_level = aggregate_risk
    action = aggregate_action

    return {
        "report_version": 2,
        "review_id": str(uuid.uuid4()),
        "flagged_items": flagged_items,
        "findings": findings,
        "regulation_evidence": regulation_evidence,
        "evidence_coverage": round(evidence_coverage, 4),
        "red_lines": red_lines,
        "assessment": assessment,
        "sources": sources,
        "risk_level": risk_level,
        "action": action,
        "reason": report["reason"],
        "suggestions": report["suggestions"],
        "evidence": report["evidence"],
        "audit": {
            "stages": audit_stages,
            "red_lines": red_lines,
            "flagged_count": len(flagged_items),
            "finding_count": len(findings),
            "query_understanding": query_understanding,
            "retrieved_sources": [s["source"] for s in sources],
            "risk_level": risk_level,
            "action": action,
            "reason": report["reason"],
            "suggestions": report["suggestions"],
            "evidence": report["evidence"],
            "evidence_coverage": round(evidence_coverage, 4),
        },
    }
