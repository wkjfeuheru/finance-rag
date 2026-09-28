"""检查问题清单里的引文提示词在**完整块内容**中命中几个块。

为什么单独做这个工具：`dump_chunks.py` 导出的内容按 300 字符截断，
看着"只出现一次"的引文很可能在另一个块的正文深处也出现——那样 gold 集的
证据 id 就成了随机挑一个块，指标随之失真。这里读全量内容，把每个提示词的
命中块全部列出来，便于改成真正唯一的写法。

用法::

    python scripts/check_hints.py --questions finance_rag/src/eval/data/research_report_questions.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):  # pragma: no cover
            pass

    parser = argparse.ArgumentParser(description="检查引文提示词的命中块数量")
    parser.add_argument("--questions", required=True)
    parser.add_argument("--out", help="把命中明细写成 UTF-8 文件")
    args = parser.parse_args(argv)

    from finance_rag.src.core.config import KB_COLLECTION_NAME
    from finance_rag.src.infrastructure.vector_store.milvus_kb import get_knowledge_base

    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    kb = get_knowledge_base(KB_COLLECTION_NAME)
    client = kb._get_client()
    rows = client.query(
        collection_name=kb.collection_name,
        filter='tenant_id != ""',
        output_fields=["id", "source", "content", "start_page"],
        limit=16384,
    )

    lines: list[str] = []
    bad = 0
    for item in questions:
        hints = item.get("evidence") or []
        if not hints:
            continue
        lines.append(f"{item['id']}  ({item.get('question_type')})")
        for hint in hints:
            quote = str(hint.get("quote_hint") or "")
            source_hint = str(hint.get("source_hint") or "")
            hits: list[str] = []
            for row in rows:
                source = str(row.get("source") or "")
                if source_hint and source_hint not in source:
                    continue
                if quote and quote in str(row.get("content") or ""):
                    hits.append(f"{str(row.get('id'))[:8]}@p{row.get('start_page')}")
            mark = "OK " if len(hits) == 1 else ("MISS" if not hits else "DUP ")
            if len(hits) != 1:
                bad += 1
            lines.append(f"  [{mark}] hits={len(hits)} {quote!r} -> {', '.join(hits[:6])}")
        lines.append("")

    text = "\n".join(lines)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    print(f"非唯一/未命中的引文：{bad}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
