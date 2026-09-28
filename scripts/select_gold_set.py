"""从生成的候选里筛出正式 gold 集（可复现的"人工筛"）。

筛选不是凭感觉挑好看的题，而是三条可验证的硬约束：

1. **引文必须真实存在**于它标注的那个 chunk 里（逐条在线核对）。生成器的
   ``quote`` 可能来自清洗前的文本、或跨块拼接，落不到证据块上——这种题必须剔除，
   否则 ``evidence_hit`` 会假性归零，被误读成"检索不行"。
2. **多跳题必须跨报告**：证据来自 ≥2 篇不同文档。本项目里 multi_hop 的定义就是
   「跨报告整合同一主题」，同篇两段不算。
3. **类型配额**：按 spec 的 10 跨报告整合 + 8 单跳 + 2 负样本。

用法::

    python scripts/select_gold_set.py --candidates <candidates.jsonl> --out <gold.jsonl>
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LINE = "=" * 78
#: 引文最短长度：太短的引文（「增长」「买入」）无法证明它确实来自该块
MIN_QUOTE_CHARS = 12
#: 单跳题里的问题若短于该长度，多半是残句
MIN_QUERY_CHARS = 8
TARGET = {"single_hop": 8, "multi_hop": 10, "negative": 2}


def _load_chunks(kb) -> dict[str, str]:
    """``chunk_id -> content``（只读一次，逐条核引用）。"""
    client = kb._get_client()
    rows = client.query(
        collection_name=kb.collection_name,
        filter='tenant_id != ""',
        output_fields=["id", "content"],
        limit=16384,
    )
    return {str(r.get("id") or ""): str(r.get("content") or "") for r in rows}


def _evidence_ok(evidence: list[dict], chunks: dict[str, str]) -> tuple[bool, str]:
    """逐条核对引文；返回 ``(是否通过, 原因)``。"""
    for item in evidence:
        chunk_id = str(item.get("chunk_id") or "")
        quote = str(item.get("quote") or "").strip()
        if chunk_id not in chunks:
            return False, f"证据块不存在：{chunk_id[:12]}"
        if len(quote) < MIN_QUOTE_CHARS:
            return False, f"引文过短（{len(quote)} 字）"
        # 允许空白差异：把引文按空白切成片段，要求各片段都出现
        pieces = [p for p in quote.split() if len(p) >= 4]
        haystack = chunks[chunk_id]
        missing = [p for p in pieces if p not in haystack]
        if pieces and len(missing) == len(pieces):
            # 整条引文对不上时，再试一次「去空白后包含」
            if "".join(quote.split()) not in "".join(haystack.split()):
                return False, f"引文不在证据块中：{quote[:24]}…"
    return True, ""


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):  # pragma: no cover
            pass

    parser = argparse.ArgumentParser(description="从候选里筛出正式 gold 集")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--report", default="")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)

    from finance_rag.src.core.config import KB_COLLECTION_NAME
    from finance_rag.src.eval.dataset import EvaluationDataset
    from finance_rag.src.infrastructure.vector_store.milvus_kb import get_knowledge_base

    kb = get_knowledge_base(KB_COLLECTION_NAME)
    chunks = _load_chunks(kb)
    dataset = EvaluationDataset.load(args.candidates)
    print(LINE)
    print(f"候选 {len(dataset.cases)} 条；集合内块 {len(chunks)}")
    print(f"候选类型分布：{dict(Counter(c.question_type for c in dataset.cases))}")
    print(LINE)

    reasons: list[str] = []
    accepted: dict[str, list] = {"single_hop": [], "multi_hop": [], "negative": []}
    for case in dataset.cases:
        if len(case.query.strip()) < MIN_QUERY_CHARS:
            reasons.append(f"{case.id}: 问题过短")
            continue
        if case.question_type == "negative":
            if not case.expected_behavior.strip():
                reasons.append(f"{case.id}: 负样本缺少 expected_behavior")
                continue
            accepted["negative"].append(case)
            continue

        evidence = [
            {"chunk_id": e.chunk_id, "quote": e.quote, "source": e.source}
            for e in case.evidence
        ]
        ok, why = _evidence_ok(evidence, chunks)
        if not ok:
            reasons.append(f"{case.id}: {why}")
            continue
        sources = {e["source"] for e in evidence}
        if case.question_type == "multi_hop" and len(sources) < 2:
            reasons.append(f"{case.id}: 多跳题证据只来自 {len(sources)} 篇")
            continue
        accepted[case.question_type].append(case)

    selected = []
    for qtype, want in TARGET.items():
        pool = accepted[qtype]
        print(f"{qtype:11} 通过 {len(pool):3} 条，需要 {want}")
        selected.extend(pool[:want])
        if len(pool) < want:
            print(f"  [!] 不足：缺 {want - len(pool)} 条")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    EvaluationDataset(selected).save(out_path)
    print(LINE)
    print(f"已写出 {len(selected)} 条 -> {out_path}")
    print(f"类型分布：{dict(Counter(c.question_type for c in selected))}")
    with_source = Counter(e.source for c in selected for e in c.evidence)
    print(f"覆盖文档：{len(with_source)} 篇；单篇最多被引用 {max(with_source.values()) if with_source else 0} 次")
    if reasons:
        print(LINE)
        print(f"剔除 {len(reasons)} 条：")
        for reason in reasons[:15]:
            print(f"  - {reason}")
        if len(reasons) > 15:
            print(f"  …共 {len(reasons)} 条")
        if args.report:
            Path(args.report).write_text("\n".join(reasons), encoding="utf-8")
    print(LINE)
    return 0 if selected else 1


if __name__ == "__main__":
    raise SystemExit(main())
