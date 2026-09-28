"""量化"模板块挤占 top-5"的诊断。

``recall@5`` 单独看无法区分两件事：

1. **真的没检索到**：Top-5 里没有一篇是证据所在文档；
2. **检索到了文档，但 top-5 被模板块占满**：``资料来源````免责声明````评级说明``
   ``目录`` 这类块与任何问题都"像"，把正文块挤出前 5。

第 2 种在真实使用里同样是 bug（喂给 LLM 的上下文全是免责声明），
但修复方向完全不同，所以必须分开计量。

用法::

    python scripts/diag_baseline.py --root scripts/results/research_report/18-docs-baseline
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: 模板/版式块的特征词：出现即视为"不含业务信息"
BOILERPLATE_PATTERNS = (
    "免责声明", "特别声明", "重要声明", "分析师声明", "分析师承诺", "证券分析师承诺",
    "投资评级说明", "评级说明", "评级体系", "投资评级标准", "行业投资评级",
    "资料来源", "数据来源", "地址：", "邮编：", "公司网址", "联系电话",
    "法律声明", "版权声明", "联系我们", "风险提示", "免责条款", "适当性管理办法",
    "分析师：", "执业证书", "登记编号", "SAC", "证书编号", "THANK YOU", "演示完毕",
    "谢谢观看", "图表目录", "目录",
)
_BOILER_RE = re.compile("|".join(re.escape(p) for p in BOILERPLATE_PATTERNS))
_IMG_RE = re.compile(r"^!\[\]\(images/|^\[图片\]")


def _is_boilerplate(chunk: dict) -> bool:
    content = str(chunk.get("content") or "")
    if chunk.get("block_type") == "image" or _IMG_RE.match(content.strip()):
        return True
    # 以模板特征开头或以之为主体（短块直接命中即算）
    if _BOILER_RE.search(content[:120]):
        return True
    return len(content.strip()) < 40 and bool(_BOILER_RE.search(content))


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):  # pragma: no cover
            pass

    parser = argparse.ArgumentParser(description="模板块挤占诊断")
    parser.add_argument("--root", required=True)
    parser.add_argument("--out")
    args = parser.parse_args(argv)

    from finance_rag.src.core.config import KB_COLLECTION_NAME
    from finance_rag.src.infrastructure.vector_store.milvus_kb import get_knowledge_base

    kb = get_knowledge_base(KB_COLLECTION_NAME)
    client = kb._get_client()
    rows = client.query(
        collection_name=kb.collection_name,
        filter='tenant_id != ""',
        output_fields=["id", "source", "content", "block_type"],
        limit=16384,
    )
    chunks = {str(r.get("id") or ""): r for r in rows}

    lines: list[str] = []
    for run_dir in sorted(p for p in Path(args.root).rglob("*-L?") if p.is_dir()):
        layer = run_dir.name.split("-")[-1]
        records_path = run_dir / "records.jsonl"
        if not records_path.exists():
            continue
        records = [
            json.loads(line)
            for line in records_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        evidence_total = doc_hit = exact_hit = rank3 = 0
        cand_hit = cand_total = 0
        slots = boiler_slots = 0
        kind = {"boilerplate": 0, "image": 0, "table": 0, "text": 0}
        for record in records:
            diag = record.get("diagnostics") or {}
            ranks = diag.get("evidence_ranks") or {}
            top5 = [c for c in (diag.get("top5_chunk_ids") or []) if c]
            candidates = {c for c in (diag.get("candidate_chunk_ids") or []) if c}
            top5_sources = {str(chunks.get(c, {}).get("source") or "") for c in top5}
            for cid, rank in ranks.items():
                evidence_total += 1
                if rank is not None:
                    exact_hit += 1
                    if rank <= 3:
                        rank3 += 1
                cand_total += 1
                if cid in candidates:
                    cand_hit += 1
                ev_source = str(chunks.get(cid, {}).get("source") or "")
                if ev_source and ev_source in top5_sources:
                    doc_hit += 1
            for cid in top5:
                slots += 1
                chunk = chunks.get(cid, {})
                btype = str(chunk.get("block_type") or "")
                if _is_boilerplate(chunk):
                    boiler_slots += 1
                    kind["boilerplate" if btype != "image" else "image"] += 1
                elif btype == "table":
                    kind["table"] += 1
                else:
                    kind["text"] += 1
        if not evidence_total:
            continue
        lines.append(f"### {layer}")
        lines.append(f"  证据总数                    {evidence_total}")
        lines.append(f"  exact top-5 命中率          {exact_hit / evidence_total:.4f}  ({exact_hit})")
        lines.append(f"  rank<=3 占比                {rank3 / evidence_total:.4f}  ({rank3})")
        lines.append(f"  候选池内命中率（召回上限）  {cand_hit / cand_total:.4f}  ({cand_hit}/{cand_total})")
        lines.append(f"  文档级 top-5 命中率         {doc_hit / evidence_total:.4f}  ({doc_hit})")
        lines.append(f"  top-5 槽位总数              {slots}")
        lines.append(f"  其中模板/图片块占槽         {boiler_slots / slots:.4f}  ({boiler_slots})")
        lines.append(f"  top-5 构成                  {kind}")
        lines.append("")

    text = "\n".join(lines)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
