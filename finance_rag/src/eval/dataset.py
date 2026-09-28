"""版本化 JSONL 评测集、质量门禁与冻结清单。"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

SCHEMA_VERSION = "1.0"
DEFAULT_QUOTAS = {
    "gold": {"single_hop": 30, "multi_hop": 15, "negative": 15},
    "silver": {"single_hop": 50, "multi_hop": 25, "negative": 15},
}
QUESTION_TYPES = {"single_hop", "multi_hop", "negative"}
TIERS = {"gold", "silver"}
REVIEW_STATUSES = {"pending", "approved", "rejected", "auto_pass"}
DIFFICULTIES = {"easy", "medium", "hard"}
ANSWER_TYPES = {"factual", "list", "comparison", "procedural"}


class DatasetValidationError(ValueError):
    """评测集结构或质量门禁失败。"""


@dataclass(frozen=True)
class EvidenceRef:
    source: str
    document_version: str
    chunk_id: str
    quote: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EvidenceRef":
        return cls(
            source=str(value.get("source", "")).strip(),
            document_version=str(value.get("document_version", "")).strip(),
            chunk_id=str(value.get("chunk_id", "")).strip(),
            quote=str(value.get("quote", "")).strip(),
        )


@dataclass(frozen=True)
class EvalCase:
    id: str
    query: str
    question_type: str
    difficulty: str
    answer_type: str
    reference_answer: str
    required_facts: tuple[str, ...]
    evidence: tuple[EvidenceRef, ...]
    expected_behavior: str
    tier: str
    review_status: str
    reviewer: str
    review_note: str
    provenance: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EvalCase":
        return cls(
            id=str(value.get("id", "")).strip(),
            query=str(value.get("query", "")).strip(),
            question_type=str(value.get("question_type", "")).strip(),
            difficulty=str(value.get("difficulty", "medium")).strip(),
            answer_type=str(value.get("answer_type", "factual")).strip(),
            reference_answer=str(value.get("reference_answer", "")).strip(),
            required_facts=tuple(str(item).strip() for item in value.get("required_facts", []) if str(item).strip()),
            evidence=tuple(EvidenceRef.from_dict(item) for item in value.get("evidence", [])),
            expected_behavior=str(value.get("expected_behavior", "")).strip(),
            tier=str(value.get("tier", "silver")).strip(),
            review_status=str(value.get("review_status", "pending")).strip(),
            reviewer=str(value.get("reviewer", "")).strip(),
            review_note=str(value.get("review_note", "")).strip(),
            provenance=dict(value.get("provenance") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["required_facts"] = list(self.required_facts)
        value["evidence"] = [asdict(item) for item in self.evidence]
        return value


@dataclass(frozen=True)
class DatasetAudit:
    valid: bool
    case_count: int
    checked_evidence: int
    errors: tuple[str, ...] = ()


EvidenceResolver = Callable[[str], dict[str, Any] | None]


class EvaluationDataset:
    """统一评测集接口：加载、校验、审核和冻结。"""

    def __init__(self, cases: Iterable[EvalCase], *, source_path: Path | None = None):
        self.cases = tuple(cases)
        self.source_path = source_path

    @classmethod
    def from_dicts(cls, rows: Iterable[dict[str, Any]]) -> "EvaluationDataset":
        return cls(EvalCase.from_dict(row) for row in rows)

    @classmethod
    def load(
        cls,
        path: str | Path,
        tier: str | None = None,
        review_status: str | None = None,
    ) -> "EvaluationDataset":
        path = Path(path)
        cases: list[EvalCase] = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    cases.append(EvalCase.from_dict(json.loads(line)))
                except (TypeError, json.JSONDecodeError) as exc:
                    raise DatasetValidationError(f"第 {line_number} 行不是有效 JSON：{exc}") from exc
        if tier is not None:
            cases = [case for case in cases if case.tier == tier]
        if review_status is not None:
            cases = [case for case in cases if case.review_status == review_status]
        dataset = cls(cases, source_path=path)
        dataset.validate()
        return dataset

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for case in self.cases:
                handle.write(json.dumps(case.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
        return path

    def validate(
        self,
        *,
        evidence_resolver: EvidenceResolver | None = None,
    ) -> DatasetAudit:
        errors: list[str] = []
        seen_ids: set[str] = set()
        seen_queries: set[str] = set()
        checked_evidence = 0
        for case in self.cases:
            prefix = case.id or "<missing-id>"
            if not case.id:
                errors.append("存在缺少 id 的条目")
            elif case.id in seen_ids:
                errors.append(f"{prefix}: id 重复")
            seen_ids.add(case.id)
            if not case.query:
                errors.append(f"{prefix}: 缺少问题")
            elif case.query in seen_queries:
                errors.append(f"{prefix}: 问题重复")
            seen_queries.add(case.query)
            if case.question_type not in QUESTION_TYPES:
                errors.append(f"{prefix}: question_type 非法")
            if case.tier not in TIERS:
                errors.append(f"{prefix}: tier 非法")
            if case.review_status not in REVIEW_STATUSES:
                errors.append(f"{prefix}: review_status 非法")
            if case.difficulty not in DIFFICULTIES:
                errors.append(f"{prefix}: difficulty 非法")
            if case.answer_type not in ANSWER_TYPES:
                errors.append(f"{prefix}: answer_type 非法")
            if case.question_type == "negative":
                if not case.expected_behavior:
                    errors.append(f"{prefix}: 负样本缺少 expected_behavior")
                if case.evidence:
                    errors.append(f"{prefix}: 负样本不应包含 evidence")
                continue
            if not case.reference_answer:
                errors.append(f"{prefix}: 正例缺少 reference_answer")
            if not case.required_facts:
                errors.append(f"{prefix}: 正例缺少 required_facts")
            required_evidence = 2 if case.question_type == "multi_hop" else 1
            if len(case.evidence) < required_evidence:
                errors.append(f"{prefix}: 证据数量不足")
            if case.question_type == "multi_hop" and len({item.source for item in case.evidence}) < 2:
                errors.append(f"{prefix}: 多跳证据必须来自不同文档")
            for evidence in case.evidence:
                if (
                    not evidence.source or not evidence.document_version
                    or not evidence.chunk_id or not evidence.quote
                ):
                    errors.append(f"{prefix}: evidence 缺少 source/document_version/chunk_id/quote")
                    continue
                if evidence_resolver is None:
                    continue
                checked_evidence += 1
                actual = evidence_resolver(evidence.chunk_id)
                if not actual:
                    errors.append(f"{prefix}: chunk 不存在：{evidence.chunk_id}")
                    continue
                if str(actual.get("source", "")) != evidence.source:
                    errors.append(f"{prefix}: chunk/source 不匹配：{evidence.chunk_id}")
                if evidence.quote not in str(actual.get("content", "")):
                    errors.append(f"{prefix}: quote 无法在 chunk 中定位：{evidence.chunk_id}")
        if errors:
            raise DatasetValidationError("；".join(errors))
        return DatasetAudit(True, len(self.cases), checked_evidence)

    def export_review(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "id", "tier", "review_status", "reviewer", "review_note",
            "question_type", "query", "reference_answer", "evidence_sources", "evidence_quotes",
        ]
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for case in self.cases:
                writer.writerow({
                    "id": case.id,
                    "tier": case.tier,
                    "review_status": case.review_status,
                    "reviewer": case.reviewer,
                    "review_note": case.review_note,
                    "question_type": case.question_type,
                    "query": case.query,
                    "reference_answer": case.reference_answer,
                    "evidence_sources": " | ".join(item.source for item in case.evidence),
                    "evidence_quotes": " | ".join(item.quote for item in case.evidence),
                })
        return path

    def import_review(self, path: str | Path) -> "EvaluationDataset":
        allowed = {"pending", "approved", "rejected", "auto_pass"}
        updates: dict[str, dict[str, str]] = {}
        with Path(path).open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                case_id = (row.get("id") or "").strip()
                status = (row.get("review_status") or "").strip()
                if case_id and status:
                    if status not in allowed:
                        raise DatasetValidationError(f"{case_id}: 审核状态非法：{status}")
                    updates[case_id] = {
                        "review_status": status,
                        "reviewer": (row.get("reviewer") or "").strip(),
                        "review_note": (row.get("review_note") or "").strip(),
                    }
        known = {case.id for case in self.cases}
        unknown = sorted(set(updates) - known)
        if unknown:
            raise DatasetValidationError(f"审核文件包含未知 id：{unknown}")
        return EvaluationDataset(
            (
                replace(case, **updates[case.id])
                if case.id in updates else case
                for case in self.cases
            ),
            source_path=self.source_path,
        )

    def freeze(
        self,
        output_path: str | Path,
        *,
        quotas: dict[str, dict[str, int]] | None = None,
        kb_fingerprint: str,
        dataset_version: str | None = None,
        max_source_share: float = 0.20,
    ) -> dict[str, Any]:
        self.validate()
        quotas = quotas or DEFAULT_QUOTAS
        selected: list[EvalCase] = []
        selected_source_counts: Counter[str] = Counter()
        for tier, type_quotas in quotas.items():
            for question_type, expected in type_quotas.items():
                eligible_status = "approved" if tier == "gold" else "auto_pass"
                candidates = sorted(
                    (
                        case for case in self.cases
                        if case.tier == tier
                        and case.question_type == question_type
                        and case.review_status == eligible_status
                    ),
                    key=lambda case: case.id,
                )
                if len(candidates) < expected:
                    if tier == "gold":
                        raise DatasetValidationError(
                            f"Gold {question_type} 需要 {expected} 条 approved，实际 {len(candidates)} 条"
                        )
                    raise DatasetValidationError(
                        f"Silver {question_type} 需要 {expected} 条 auto_pass，实际 {len(candidates)} 条"
                    )
                # 在满足各层配额的候选中优先选择当前引用次数较低的来源，
                # 让冻结集在来源上尽量均衡；无解时仍由下方严格 20% 门禁失败。
                remaining = list(candidates)
                for _ in range(expected):
                    def balance_key(case: EvalCase) -> tuple[float, float, str]:
                        sources = [item.source for item in case.evidence] or [""]
                        projected = selected_source_counts.copy()
                        projected.update(sources)
                        total = sum(projected.values()) or 1
                        return (
                            max(projected[source] / total for source in sources),
                            sum(projected[source] for source in sources) / len(sources),
                            case.id,
                        )

                    chosen = min(remaining, key=balance_key)
                    selected.append(chosen)
                    selected_source_counts.update(item.source for item in chosen.evidence)
                    remaining.remove(chosen)

        selected_ids = {case.id for case in selected}
        if len(selected_ids) != len(selected):
            raise DatasetValidationError("冻结选择包含重复 id")

        expected_total = sum(sum(values.values()) for values in quotas.values())
        if len(selected) != expected_total:
            raise DatasetValidationError(f"冻结数量错误：期望 {expected_total}，实际 {len(selected)}")

        fingerprints = {
            str(case.provenance.get("kb_fingerprint", ""))
            for case in selected
            if case.provenance.get("kb_fingerprint")
        }
        if fingerprints and fingerprints != {kb_fingerprint}:
            raise DatasetValidationError(
                f"知识库指纹漂移：数据集={sorted(fingerprints)} 当前={kb_fingerprint}"
            )

        source_counts = Counter(
            evidence.source
            for case in selected
            if case.question_type != "negative"
            for evidence in case.evidence
        )
        total_evidence = sum(source_counts.values())
        if total_evidence:
            over_limit = {
                source: count / total_evidence
                for source, count in source_counts.items()
                if count / total_evidence > max_source_share
            }
            if over_limit:
                raise DatasetValidationError(f"单文档证据占比超过 {max_source_share:.0%}：{over_limit}")

        frozen = EvaluationDataset(sorted(selected, key=lambda case: case.id))
        output_path = frozen.save(output_path)
        version = dataset_version or datetime.now(timezone.utc).strftime("%Y%m%d")
        counts = {
            tier: dict(Counter(case.question_type for case in selected if case.tier == tier))
            for tier in sorted(quotas)
        }
        content_hash = hashlib.sha256(output_path.read_bytes()).hexdigest()
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "dataset_version": version,
            "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
            "kb_fingerprint": kb_fingerprint,
            "dataset_sha256": content_hash,
            "case_count": len(selected),
            "counts": counts,
            "source_distribution": dict(sorted(source_counts.items())),
            "max_source_share": max_source_share,
        }
        manifest_path = output_path.with_suffix(".manifest.json")
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return manifest
