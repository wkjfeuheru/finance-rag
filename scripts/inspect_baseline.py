"""把分层基线检查点里的证据排名摊平成一张表，用于定位"检索不到"的真实原因。

只看汇总的 recall@5 无法区分三种完全不同的失败：
``证据块根本没进候选``（召回问题）、``进了候选但被 top-k 截断``（排序问题）、
``返回的 id 与标注 id 不是同一层粒度``（口径问题）。这里把每题的证据 id 与
Top-5 实际返回 id 并排列出，一眼能分辨。

用法::

    python scripts/inspect_baseline.py --root scripts/results/research_report/18-docs-baseline
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):  # pragma: no cover
            pass

    parser = argparse.ArgumentParser(description="摊平分层基线的证据排名")
    parser.add_argument("--root", required=True)
    parser.add_argument("--layer", default="", help="只看某一层")
    parser.add_argument("--case", default="", help="只看某一题")
    parser.add_argument("--out", help="明细写入文件")
    args = parser.parse_args(argv)

    root = Path(args.root)
    run_dirs = sorted(p for p in root.rglob("*-L?") if p.is_dir())
    if not run_dirs:
        print(f"未找到分层运行目录：{root}")
        return 1

    lines: list[str] = []
    for run_dir in run_dirs:
        layer = run_dir.name.split("-")[-1]
        if args.layer and layer != args.layer.upper():
            continue
        records_path = run_dir / "records.jsonl"
        summary_path = run_dir / "summary.json"
        if not records_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
        metrics = summary.get("metrics", {})
        lines.append(
            f"### {layer}  n={summary.get('n')}  "
            f"recall@5={metrics.get('recall_at_5')}  mrr={metrics.get('mrr')}  "
            f"p@5={metrics.get('precision_at_5')}"
        )
        hit_rank_top3 = 0
        hit_any = 0
        total_evidence = 0
        for record in _load(records_path):
            case_id = record.get("case_id", "")
            if args.case and args.case not in case_id:
                continue
            diag = record.get("diagnostics") or {}
            ranks = diag.get("evidence_ranks") or {}
            top5 = diag.get("top5_chunk_ids") or []
            cand = diag.get("candidate_chunk_ids") or []
            ranks_text = ", ".join(
                f"{cid[:8]}:{'MISS' if r is None else r}" for cid, r in ranks.items()
            )
            found = [r for r in ranks.values() if r is not None]
            total_evidence += len(ranks)
            hit_any += len(found)
            hit_rank_top3 += sum(1 for r in found if r <= 3)
            lines.append(
                f"{case_id:16} {record.get('question_type',''):10} "
                f"证据排名[{ranks_text}]  候选深度={len(cand)}  "
                f"top5={[c[:8] for c in top5]}"
            )
        if total_evidence:
            lines.append(
                f"  -> 证据命中 {hit_any}/{total_evidence} "
                f"= {hit_any / total_evidence:.4f}；其中 rank<=3 占比 "
                f"{hit_rank_top3 / total_evidence:.4f}"
            )
        lines.append("")

    text = "\n".join(lines)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
