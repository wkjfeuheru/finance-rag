"""从真实入库结果构建研报 gold 评测集（JSONL）。

为什么需要一个构建器而不是手写 gold 集：

* **证据必须是真实的**。手写 chunk id 极易写错或写到幻觉——一个 id 打错，
  `evidence_hit` 就会假性归零，最后被误读成"检索不行"。这里要求给出**引文片段**，
  由脚本在真实 chunk 里定位；定位不到就丢弃该题并打印出来，绝不编造。
* 题集要**可重建**。语料补进来、集合重建之后，同一份问题清单能重新产出 gold 集。

问题清单格式（`--questions`，JSON）::

    [
      {
        "id": "rr-single-001",
        "query": "国机汽车 2026 年 H1 的营业收入是多少？",
        "question_type": "single_hop",
        "answer_type": "factual",
        "reference_answer": "2026 年 H1 实现营收 153.62 亿元，同比 -8.71%。",
        "expected_behavior": "",
        "evidence": [
          {"source_hint": "H3_AP202609261829925702", "quote_hint": "153.62"}
        ]
      }
    ]

``question_type`` 为 ``negative`` 的题不需要 ``evidence``，但必须给
``expected_behavior``（例如"应拒答：语料中没有该公司数据"）。

用法::

    python scripts/build_gold_set.py --questions q.json --out finance_rag/src/eval/data/research_report_gold.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LINE = "=" * 78
DEFAULT_DIFFICULTY = "medium"
MIN_QUOTE_CHARS = 8
MAX_QUOTE_CHARS = 120
#: 引文提示词的最短长度。太短的提示（如「汽车」「评级」）会命中一大堆无关块，
#: 于是 gold 集的证据 id 变成随机结果——那等于自己伪造证据。
MIN_HINT_CHARS = 4
#: ``EvalCase.provenance`` 是 dict（``dataset.py`` 用 ``dict(...)`` 解析），
#: 写成字符串会在加载时抛 ValueError。这里统一用结构化来源描述。
PROVENANCE: dict[str, str] = {
    "corpus": "research-report-18-docs",
    "builder": "scripts/build_gold_set.py",
    "evidence": "quotes located in real chunks (unique match required)",
}


def _load_chunks(kb, sources: list[str] | None = None) -> dict[str, list[dict]]:
    """把集合里的块按 source 读出来（一次性读完，避免每题都查 Milvus）。"""
    client = kb._get_client()
    fields = ["id", "source", "content", "start_page", "block_type"]
    try:
        rows = client.query(
            collection_name=kb.collection_name,
            filter='tenant_id != ""',
            output_fields=fields + ["version"],
            limit=16384,
        )
    except Exception:
        # 旧 schema 没有 version 字段：不带它查询，后面按 source 回落
        rows = client.query(
            collection_name=kb.collection_name,
            filter='tenant_id != ""',
            output_fields=fields,
            limit=16384,
        )
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        if sources and row.get("source") not in sources:
            continue
        grouped.setdefault(str(row.get("source") or ""), []).append(row)
    return grouped


def _document_version(chunk: dict) -> str:
    """证据所属文档的版本标识。

    评测集校验要求 ``document_version`` 非空，用来把证据钉在某一次文档版本上。
    本语料的文件名（``H3_AP2026..._1.pdf``）不含 ``vN`` 标记，入库时
    ``extract_document_version`` 返回空串，因此这里回落到 ``source``——
    它是文档的真实标识，不是编造出来的版本号；代码里保留优先读 ``version``
    的分支，等语料真的带版本时会自动生效。
    """
    return str(chunk.get("version") or "").strip() or str(chunk.get("source") or "").strip()


def _find_evidence(grouped: dict[str, list[dict]], hint: dict) -> tuple[dict, str] | None:
    """在真实块里定位一条证据；返回 ``(chunk, quote)``，定位不到返回 None。

    ``candidates`` 参数用于把「命中了多个块」的情形暴露出来——命中唯一才是可信证据。
    """
    source_hint = str(hint.get("source_hint") or "")
    quote_hint = str(hint.get("quote_hint") or "").strip()
    if not quote_hint:
        return None
    hits: list[tuple[dict, str]] = []
    for source, chunks in grouped.items():
        if source_hint and source_hint not in source:
            continue
        for chunk in chunks:
            content = str(chunk.get("content") or "")
            position = content.find(quote_hint)
            if position < 0:
                continue
            # 引文取上下文窗口，便于人工复核；不改变原文
            start = max(0, position - 20)
            end = min(len(content), position + len(quote_hint) + 20)
            quote = content[start:end].strip()
            if len(quote) < MIN_QUOTE_CHARS:
                quote = content[max(0, position - 40) : position + 80].strip()
            hits.append((chunk, quote[:MAX_QUOTE_CHARS]))
    if len(hits) != 1:
        return None
    return hits[0]


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):  # pragma: no cover
            pass

    parser = argparse.ArgumentParser(description="从真实入库结果构建 gold 评测集")
    parser.add_argument("--questions", required=True, help="问题清单 JSON")
    parser.add_argument("--out", required=True, help="输出 JSONL")
    parser.add_argument("--report", help="未解析条目的报告文件（默认打印到终端）")
    parser.add_argument("--reviewer", default="implementation-agent")
    args = parser.parse_args(argv)

    from finance_rag.src.core.config import KB_COLLECTION_NAME
    from finance_rag.src.infrastructure.vector_store.milvus_kb import get_knowledge_base

    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    kb = get_knowledge_base(KB_COLLECTION_NAME)
    grouped = _load_chunks(kb)
    print(LINE)
    print(f"集合 {KB_COLLECTION_NAME}：{len(grouped)} 篇文档，"
          f"{sum(len(v) for v in grouped.values())} 个块")
    print(f"候选问题：{len(questions)}")
    print(LINE)

    accepted: list[dict] = []
    rejected: list[str] = []
    for item in questions:
        qid = item.get("id", "?")
        qtype = item.get("question_type", "single_hop")
        evidence_spec = item.get("evidence") or []

        if qtype == "negative":
            behavior = str(item.get("expected_behavior") or "").strip()
            if not behavior:
                rejected.append(f"{qid}: negative 题缺少 expected_behavior")
                continue
            accepted.append({
                "id": qid, "query": item["query"], "question_type": qtype,
                "difficulty": item.get("difficulty", DEFAULT_DIFFICULTY),
                "answer_type": item.get("answer_type", "factual"),
                "reference_answer": item.get("reference_answer", ""),
                "required_facts": item.get("required_facts", []),
                "evidence": [], "expected_behavior": behavior,
                "tier": item.get("tier", "gold"), "review_status": "approved",
                "reviewer": args.reviewer,
                "review_note": "negative 题按 expected_behavior 判定，无证据块",
                "provenance": dict(PROVENANCE, kind="negative"),
            })
            continue

        resolved: list[dict] = []
        problems: list[str] = []
        for hint in evidence_spec:
            hint_text = str(hint.get("quote_hint") or "").strip()
            if len(hint_text) < MIN_HINT_CHARS:
                problems.append(
                    f"引文提示词过短（{hint_text!r}）：会命中无关块，"
                    f"至少 {MIN_HINT_CHARS} 个字"
                )
                continue
            found = _find_evidence(grouped, hint)
            if found is None:
                problems.append(
                    f"引文 {hint_text!r} 未唯一命中任何块"
                    f"（source_hint={hint.get('source_hint')!r}）"
                )
                continue
            chunk, quote = found
            resolved.append({
                "source": str(chunk.get("source") or ""),
                "document_version": _document_version(chunk),
                "chunk_id": str(chunk.get("id") or ""),
                "quote": quote,
            })
        need = 1 if qtype == "single_hop" else 2
        if problems or len(resolved) < need:
            detail = "; ".join(problems) or f"仅解析到 {len(resolved)} 条证据（需要 {need}）"
            rejected.append(f"{qid}: {detail}")
            continue
        # multi_hop 在本项目里定义为「跨报告整合同一主题」，因此证据必须来自
        # 不同文档；同一篇里的两段不算跨报告，否则指标会虚高。
        if qtype == "multi_hop" and len({e["source"] for e in resolved}) < 2:
            rejected.append(
                f"{qid}: 证据全部来自同一篇（{resolved[0]['source']}），不构成跨报告"
            )
            continue
        accepted.append({
            "id": qid, "query": item["query"], "question_type": qtype,
            "difficulty": item.get("difficulty", DEFAULT_DIFFICULTY),
            "answer_type": item.get("answer_type", "factual"),
            "reference_answer": item.get("reference_answer", ""),
            "required_facts": item.get("required_facts", []),
            "evidence": resolved, "expected_behavior": "",
            "tier": item.get("tier", "gold"), "review_status": "approved",
            "reviewer": args.reviewer,
            "review_note": "证据由脚本在真实 chunk 中定位，引文为原文片段",
            "provenance": dict(PROVENANCE, kind="evidence"),
        })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # 不写 ``_meta`` 头：``EvaluationDataset.load`` 逐行当作 EvalCase 解析，
    # 多一行元数据会直接抛错。
    with out_path.open("w", encoding="utf-8", newline="\n") as handle:
        for case in accepted:
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")

    print(f"采纳 {len(accepted)} 题 -> {out_path}")
    by_type: dict[str, int] = {}
    for case in accepted:
        by_type[case["question_type"]] = by_type.get(case["question_type"], 0) + 1
    print(f"类型分布：{by_type}")
    if rejected:
        print(LINE)
        print(f"剔除 {len(rejected)} 题（证据不足，绝不编造）：")
        for line in rejected:
            print(f"  - {line}")
        if args.report:
            Path(args.report).write_text("\n".join(rejected), encoding="utf-8")
    print(LINE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
