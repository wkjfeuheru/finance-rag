"""在 gold 集上跑**生产问答链路**，测「引用可溯源率」与「负样本拒答正确率」。

为什么不能只用检索层评测：这两个指标衡量的是最终答案，
`evaluate.py retrieval run` 只走 ``expand_parents=False`` 的块级检索，
根本不生成答案。这里复用生产入口 ``chat_service.chat``，让查询改写、
重排、引用校验、拒答策略全部按线上默认值生效——否则测出来的不是线上行为。

口径（与 spec 一致）：

* **引用可溯源率** = 正例答案中「带 ``[N]`` 且通过 ``CitationValidator``」的比例。
  同时单独报出「正例里压根没有引用的条数」，否则 0 引用会被算成"可溯源"而虚高。
* **负样本拒答正确率** = 负例中正确拒答（拒答文案 / 拒答判定 / 空回答）的比例。

用法::

    python scripts/baseline_e2e.py --dataset finance_rag/src/eval/data/research_report_gold20.jsonl \
        --out scripts/results/research_report/18-docs-baseline/e2e
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LINE = "=" * 78
_CITATION_RE = re.compile(r"\[(\d+)\]")


def _has_citation(answer: str) -> bool:
    return bool(_CITATION_RE.search(answer or ""))


def _is_refusal(answer: str, rejected: bool) -> bool:
    """生产拒答策略是否拦下了这道题（``answer_rejected`` 或固定拒答文案）。

    这是**正例**该用的口径：正例要看的是"系统有没有误拒"，
    只看生产策略标志最干净。不能用关键词代理——实测中一个 266 字、
    正常作答并带 5 条引用的正例，只因为句子里出现"未包含"就被关键词判成拒答。
    """
    if rejected:
        return True
    from finance_rag.src.services.citation_validator import REFUSAL_ANSWER

    text = (answer or "").strip()
    if not text:
        return True
    head = REFUSAL_ANSWER.strip()[:20]
    return text == REFUSAL_ANSWER.strip() or text.startswith(head)


def _is_decline(answer: str, rejected: bool) -> bool:
    """**负例**该用的口径：模型是否拒绝作答（策略拒答，或自己改口说没有数据）。

    负例问的是"会不会编"，而生产拒答策略并不会为每个负例触发——
    实测两例负样本都是模型自己说"检索到的内容未提供…""暂未检索到…"，
    ``answer_rejected`` 为 False。所以这里必须在策略标志之外叠加关键词代理，
    同时把原始答案落盘供人工复核（关键词代理只是代理）。
    """
    if _is_refusal(answer, rejected):
        return True
    from finance_rag.src.eval.runner import negative_rejection_score

    return negative_rejection_score(answer or "") == 1.0


async def _run_case(case, chat) -> dict:
    started = time.perf_counter()
    try:
        result = await chat(case.query)
    except Exception as exc:  # 单题失败也要留痕，不能整批静默丢失
        return {
            "case_id": case.id,
            "question_type": case.question_type,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "answer": "",
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        }
    sources = result.get("sources") or []
    validation = result.get("citation_validation")
    answer = result.get("answer") or ""
    # gold 证据是否落进来源：检索层的独立交叉验证
    source_names = {str(item.get("source") or "") for item in sources}
    evidence_hit = any(e.source in source_names for e in case.evidence) if case.evidence else None
    return {
        "case_id": case.id,
        "question_type": case.question_type,
        "status": "completed",
        "answer": answer,
        "answer_chars": len(answer),
        "has_citation": _has_citation(answer),
        "citation_valid": None if validation is None else bool(validation.get("valid")),
        "citation_score": None if validation is None else validation.get("score"),
        "citation_total": None if validation is None else validation.get("total_citations"),
        "answer_rejected": bool(result.get("answer_rejected")),
        "low_confidence": bool(result.get("low_confidence")),
        "refused": _is_refusal(answer, bool(result.get("answer_rejected"))),
        "declined": _is_decline(answer, bool(result.get("answer_rejected"))),
        "source_count": len(sources),
        "source_names": sorted(source_names),
        "evidence_in_sources": evidence_hit,
        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
    }


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):  # pragma: no cover
            pass

    parser = argparse.ArgumentParser(description="生产链路端到端基线")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True, help="输出前缀（会生成 .jsonl 与 .txt）")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 题（调试用）")
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="复用已存在的 <out>.jsonl 原始答案重算汇总，不再调用 LLM。"
             "判定口径（如拒答关键词）完善后用它复评，避免重复烧钱。",
    )
    args = parser.parse_args(argv)

    from finance_rag.src.eval.dataset import EvaluationDataset
    from finance_rag.src.services.chat_service import chat

    dataset = EvaluationDataset.load(args.dataset)
    cases = list(dataset.cases)
    if args.limit:
        cases = cases[: args.limit]

    out_prefix = Path(args.out)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []

    if args.reuse:
        source = out_prefix.with_suffix(".jsonl")
        if not source.exists():
            print(f"--reuse 需要已存在的原始记录：{source}")
            return 1
        for line in source.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                # 用当前口径重算拒答，不改动已保存的答案
                record["refused"] = _is_refusal(
                    record.get("answer") or "", bool(record.get("answer_rejected"))
                )
                record["declined"] = _is_decline(
                    record.get("answer") or "", bool(record.get("answer_rejected"))
                )
                records.append(record)
        print(LINE)
        print(f"端到端复评：复用 {len(records)} 条原始答案（未调用 LLM）")
        print(LINE)
    else:
        print(LINE)
        print(f"端到端基线：{len(cases)} 题（生产 chat 链路默认参数）")
        print(LINE)
        for index, case in enumerate(cases, 1):
            record = asyncio.run(_run_case(case, chat))
            records.append(record)
            flag = "OK " if record["status"] == "completed" else "ERR"
            print(f"[{index:2}/{len(cases)}] {flag} {case.id:16} "
                  f"sources={record.get('source_count', 0):2} "
                  f"cite={record.get('citation_score')} "
                  f"rejected={record.get('answer_rejected')} "
                  f"{record.get('latency_ms', 0) / 1000:.1f}s", flush=True)

    positives = [r for r in records if r["question_type"] != "negative" and r["status"] == "completed"]
    negatives = [r for r in records if r["question_type"] == "negative" and r["status"] == "completed"]
    failures = [r["case_id"] for r in records if r["status"] != "completed"]

    traceable = [r for r in positives if r["has_citation"] and r["citation_valid"] is True]
    no_citation = [r for r in positives if not r["has_citation"]]
    invalid = [r for r in positives if r["has_citation"] and r["citation_valid"] is not True]
    refused_positives = [r for r in positives if r["refused"]]
    # 负样本口径用 declined（策略拒答 or 模型自己改口说没有数据）
    rejection_rate = (
        sum(1 for r in negatives if r["declined"]) / len(negatives) if negatives else None
    )
    traceable_rate = len(traceable) / len(positives) if positives else None
    # 排除"被拒答导致没有引用"的干扰口径：只在真的给出答案的正例里算可溯源率
    answered = [r for r in positives if not r["refused"]]
    traceable_rate_answered = (
        len([r for r in answered if r["has_citation"] and r["citation_valid"] is True]) / len(answered)
        if answered else None
    )
    evidence_rate = (
        sum(1 for r in positives if r["evidence_in_sources"]) / len(positives) if positives else None
    )

    out_prefix = Path(args.out)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    with out_prefix.with_suffix(".jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    lines = [
        LINE,
        "端到端基线（生产 chat 链路）",
        LINE,
        f"总题数：{len(records)}；失败：{len(failures)}{(' -> ' + ', '.join(failures)) if failures else ''}",
        f"正例：{len(positives)}；负例：{len(negatives)}",
        "",
        "-- 引用可溯源率（spec 口径：正例中带 [N] 且通过 CitationValidator）--",
        f"  可溯源 {len(traceable)} / {len(positives)} = {traceable_rate:.4f}"
        if traceable_rate is not None else "  无可统计正例",
        f"  其中：无引用 {len(no_citation)} 条；有引用但校验未通过 {len(invalid)} 条",
        f"  被拒答的正例（生产策略误拒） {len(refused_positives)} 条"
        + (f" -> {', '.join(r['case_id'] for r in refused_positives)}" if refused_positives else ""),
        f"  参考口径（只算真正作答的正例）：{traceable_rate_answered:.4f}"
        if traceable_rate_answered is not None else "  参考口径：无可统计",
        "",
        "-- 负样本拒答正确率（口径：生产策略拒答 或 模型自己改口说没有数据）--",
        f"  正确拒答 {sum(1 for r in negatives if r['declined'])} / {len(negatives)} = {rejection_rate:.4f}"
        if rejection_rate is not None else "  无负例",
        "  注：关键词代理判定，原始答案已落盘 JSONL，须人工复核。",
        "",
        "-- 交叉验证：gold 证据是否出现在来源里（含父块扩展的生产路径）--",
        f"  {evidence_rate:.4f}" if evidence_rate is not None else "  无正例",
        LINE,
    ]
    text = "\n".join(lines)
    out_prefix.with_suffix(".txt").write_text(text, encoding="utf-8")
    print(text)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
