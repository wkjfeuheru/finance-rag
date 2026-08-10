"""引用验证器：验证 LLM 回答中的引用标记是否与来源文档匹配。"""

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np

from config.settings import CITATION_SIMILARITY_THRESHOLD

logger = logging.getLogger(__name__)


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
