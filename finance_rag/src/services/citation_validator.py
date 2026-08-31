"""引用验证器：验证 LLM 回答中的引用标记是否与来源文档匹配。

同时提供拒答策略：
* 检索为空 / 引用校验分数过低（幻觉风险高）时拒绝回答（非流式替换答案；
  流式路径由调用方在已输出内容后追加警示）；
* 分数介于低置信阈值与拒答阈值之间时标记 ``low_confidence``。
"""

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np

from finance_rag.src.core.config import (
    CITATION_SIMILARITY_THRESHOLD,
    ENABLE_CITATION_VALIDATION,
    ENABLE_REFUSAL,
    LOW_CONFIDENCE_THRESHOLD,
    REFUSAL_MIN_RERANK_SCORE,
    REFUSAL_SCORE_THRESHOLD,
)

logger = logging.getLogger(__name__)

# 拒答固定文案（检索为空 / 引用校验失败时使用）
REFUSAL_ANSWER = (
    "抱歉，我无法给出可靠回答：检索到的文档内容不足以支撑该问题，"
    "请补充相关文档或换个问法。"
)

# 流式路径的尾部警示模板（内容已输出不可撤回）
STREAM_WARNING = (
    "\n\n⚠️ 提示：本回答的引用校验未通过（score={score:.2f}），"
    "内容仅供参考，请以原始文档为准。"
)


class CitationValidator:
    """验证回答中的 [1][2] 引用是否与来源内容匹配。

    使用嵌入语义相似度检查引用上下文是否确实存在于来源文档中。
    """

    def __init__(self, embed_fn):
        """初始化引用验证器。

        Args:
            embed_fn: 嵌入函数，接受文本返回 float 列表。
        """
        self._embed = embed_fn

    def validate(
        self,
        answer: str,
        sources: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """验证回答中的引用标记。

        Args:
            answer: LLM 生成的回答文本。
            sources: 检索到的来源文档列表（含 content 字段）。

        Returns:
            {"valid": bool, "issues": list[str], "score": float}
        """
        citations = list(re.finditer(r"\[(\d+)\]", answer))

        if not citations:
            return {"valid": True, "issues": [], "score": 1.0}

        issues: list[str] = []
        checked = 0

        for match in citations:
            ref_num = int(match.group(1))
            context = self._get_citation_context(answer, match.start())

            if ref_num < 1:
                issues.append(f"引用[{ref_num}]超出范围（引用编号必须从1开始）")
                continue
            if ref_num > len(sources):
                issues.append(
                    f"引用[{ref_num}]超出范围（仅有{len(sources)}个来源）"
                )
                continue

            source = sources[ref_num - 1]
            source_content = source.get("preview", "") or source.get("content", "")

            if not source_content:
                issues.append(f"引用[{ref_num}]的来源内容为空")
                continue

            if self._verify_in_source(context, source_content):
                checked += 1
            else:
                issues.append(f"引用[{ref_num}]的内容与来源不匹配")

        total = len(citations)
        score = checked / total if total > 0 else 1.0

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "score": round(score, 4),
            "total_citations": total,
            "verified_citations": checked,
        }

    @staticmethod
    def _get_citation_context(answer: str, pos: int, window: int = 100) -> str:
        """获取引用标记周围的上下文文本。"""
        start = max(0, pos - window)
        end = min(len(answer), pos + window)
        return answer[start:end]

    def _verify_in_source(self, claim: str, source_content: str) -> bool:
        """检查声明是否存在于来源中（基于嵌入语义相似度）。

        Args:
            claim: 引用上下文文本。
            source_content: 来源文档内容。

        Returns:
            相似度是否超过阈值。
        """
        try:
            claim_emb = np.array(self._embed(claim))
            source_emb = np.array(
                self._embed(source_content[:500])
            )
            similarity = float(
                np.dot(claim_emb, source_emb)
                / (np.linalg.norm(claim_emb) * np.linalg.norm(source_emb) + 1e-8)
            )
            return similarity >= CITATION_SIMILARITY_THRESHOLD
        except Exception as exc:
            logger.warning("引用验证嵌入计算失败：%s", exc)
            return True


def run_citation_validation(
    answer: str,
    sources: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """执行引用验证（依赖本地嵌入）；未开启或无可验证来源时返回 None。"""
    if not ENABLE_CITATION_VALIDATION or not sources:
        return None
    try:
        from finance_rag.src.infrastructure.vector_store.milvus_kb import get_knowledge_base

        kb = get_knowledge_base()
        embed_fn = kb.get_embeddings().embed_query
        return CitationValidator(embed_fn).validate(answer, sources)
    except Exception as exc:
        logger.warning("引用验证失败，跳过：%s", exc)
        return None


def apply_refusal_policy(
    validation: dict[str, Any] | None,
    source_count: int,
    *,
    rerank_scores: list[float] | None = None,
) -> tuple[bool, bool]:
    """根据引用验证结果与来源相关性决定拒答/低置信标记。

    Returns:
        ``(answer_rejected, low_confidence)``。

        * 检索为空（source_count == 0）→ 拒答；
        * 相关性不足（全部来源 rerank_score < REFUSAL_MIN_RERANK_SCORE）→ 拒答；
        * 存在引用且校验分数 < REFUSAL_SCORE_THRESHOLD → 拒答；
        * 分数介于 [REFUSAL_SCORE_THRESHOLD, LOW_CONFIDENCE_THRESHOLD) → 低置信；
        * 其余（含未开启拒答 / 未做验证 / 无引用）→ 正常返回。
    """
    if not ENABLE_REFUSAL:
        return False, False
    if source_count == 0:
        return True, False
    # 相关性守卫：重排开启时，全部来源相关性过低视为检索不足
    if rerank_scores is not None and len(rerank_scores) > 0:
        if max(rerank_scores) < REFUSAL_MIN_RERANK_SCORE:
            return True, False
    if validation is None:
        return False, False
    total = int(validation.get("total_citations", 0) or 0)
    if total == 0:
        return False, False
    score = float(validation.get("score", 1.0))
    if score < REFUSAL_SCORE_THRESHOLD:
        return True, False
    if score < LOW_CONFIDENCE_THRESHOLD:
        return False, True
    return False, False


# ---------------------------------------------------------------------------
# 条款级引用校验（合规场景：结论必须能回溯到具体「第X条」）
# ---------------------------------------------------------------------------

# 条款引用正则：匹配「第X条」，X 支持中文/阿拉伯数字编号
_CLAUSE_RE = re.compile(r"第[一二三四五六七八九十百千万0-9]+条")


def extract_clause_references(text: str) -> list[str]:
    """从文本中提取条款引用（如「第五条」「第12条」），去重保序。"""
    if not text:
        return []
    seen: list[str] = []
    for match in _CLAUSE_RE.finditer(text):
        clause = match.group(0)
        if clause not in seen:
            seen.append(clause)
    return seen




def validate_finding_evidence(
    finding: dict[str, Any],
    document_text: str,
    evidence_catalog: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """校验单个 finding 的原文证据和法规证据 ID。"""
    issues: list[str] = []
    document_evidence = finding.get("document_evidence", [])
    if not document_evidence:
        issues.append("缺少待审文档原文证据")
    for item in document_evidence:
        location = item.get("location", item)
        quote = location.get("quote", "")
        start = location.get("char_start")
        end = location.get("char_end")
        valid_quote = bool(quote and quote in document_text)
        if start is not None and end is not None:
            valid_quote = valid_quote and document_text[start:end] == quote
        if not valid_quote:
            issues.append(f"原文证据不在待审文档中：{quote[:40]}")
    regulation_ids = finding.get("regulation_evidence_ids", [])
    regulation_items = finding.get("regulation_evidence", [])
    regulation_ids = regulation_ids or [item.get("evidence_id") for item in regulation_items]
    if not regulation_ids:
        issues.append("缺少法规证据")
    for evidence_id in regulation_ids:
        evidence = evidence_catalog.get(evidence_id)
        if evidence is None:
            issues.append(f"法规证据 ID 不存在：{evidence_id}")
            continue
        quote = evidence.get("quote", "")
        if not quote:
            issues.append(f"法规证据原文为空：{evidence_id}")
    return {
        "valid": not issues,
        "status": "verified" if not issues else "insufficient",
        "issues": issues,
    }


def validate_clause_citations(
    answer: str,
    sources: list[dict[str, Any]],
) -> dict[str, Any]:
    """条款级引用校验：回答中引用的「第X条」是否存在于来源文档中。

    与通用引用校验（``CitationValidator``）互补：通用校验看 ``[N]`` 标记
    的语义匹配，本函数看「法规条款编号」是否真的出现在检索来源里，
    用于拦截编造/错引条款的幻觉。

    Returns:
        ``{"clause_citations", "verified_clauses", "total_clauses",
        "score", "issues"}``；未引用任何条款时 total_clauses 为 0、score 为 1.0。
    """
    clauses = extract_clause_references(answer)
    if not clauses:
        return {
            "clause_citations": [],
            "verified_clauses": 0,
            "total_clauses": 0,
            "score": 1.0,
            "issues": [],
        }

    # 拼接来源内容与标题，作为条款命中检索语料
    corpus = " ".join(
        f"{s.get('title', '')} {s.get('content', '') or s.get('preview', '')}"
        for s in sources
    )

    verified = 0
    issues: list[str] = []
    for clause in clauses:
        if clause in corpus:
            verified += 1
        else:
            issues.append(f"条款 {clause} 未在来源中找到")

    total = len(clauses)
    return {
        "clause_citations": clauses,
        "verified_clauses": verified,
        "total_clauses": total,
        "score": round(verified / total, 4),
        "issues": issues,
    }
