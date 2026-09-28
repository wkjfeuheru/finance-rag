"""旧 Markdown 测试集到权威 JSONL 数据模型的只读迁移。"""

from __future__ import annotations

import re
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .dataset import EvaluationDataset
from .test_set import TestSetLoader

EvidenceResolver = Callable[[str], dict[str, Any] | None]


def migrate_markdown_dataset(
    path: str | Path,
    evidence_resolver: EvidenceResolver,
    *,
    kb_fingerprint: str,
    tier: str = "gold",
    review_status: str = "pending",
) -> EvaluationDataset:
    """迁移旧测试集；证据 source 以 chunk 在线记录为准，绝不拆分文件名。"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        entries = TestSetLoader(Path(path)).load_test_set()
    generated_at = datetime.now(timezone.utc).astimezone().isoformat()
    rows: list[dict[str, Any]] = []
    for index, entry in enumerate(entries, 1):
        evidence: list[dict[str, str]] = []
        for evidence_index, chunk_id in enumerate(entry.chunk_ids):
            actual = evidence_resolver(chunk_id)
            if not actual:
                raise ValueError(f"{entry.unique_id or index}: chunk 不存在：{chunk_id}")
            content = str(actual.get("content", ""))
            suggested = (
                entry.evidence_snippets[evidence_index]
                if evidence_index < len(entry.evidence_snippets) else ""
            )
            quote = suggested if suggested and suggested in content else content.strip()[:800]
            evidence.append({
                "source": str(actual.get("source", "")),
                "document_version": str(actual.get("version", "") or "current"),
                "chunk_id": str(chunk_id),
                "quote": quote,
            })
        negative = entry.question_type == "negative"
        facts = list(entry.required_facts)
        if not negative and not facts:
            facts = [
                item.strip() for item in re.split(r"[。！？；\n]", entry.ground_truth)
                if item.strip()
            ][:5]
        rows.append({
            "id": entry.unique_id or f"legacy-{index:04d}",
            "query": entry.query,
            "question_type": entry.question_type,
            "difficulty": entry.difficulty,
            "answer_type": entry.expected_answer_type,
            "reference_answer": "" if negative else entry.ground_truth,
            "required_facts": [] if negative else facts,
            "evidence": [] if negative else evidence,
            "expected_behavior": entry.expected_behavior,
            "tier": tier,
            "review_status": review_status,
            "reviewer": "",
            "review_note": "从旧 Markdown 迁移，需重新审核",
            "provenance": {
                "generator_model": "legacy-markdown-migration",
                "generated_at": generated_at,
                "kb_fingerprint": kb_fingerprint,
                "legacy_path": str(Path(path)),
            },
        })
    dataset = EvaluationDataset.from_dicts(rows)
    dataset.validate(evidence_resolver=evidence_resolver)
    return dataset
