"""把集合里的真实块导出成可读文本，供人工挑选用作 gold 引文。

为什么需要这个脚本：gold 集的引文必须来自真实块，而"哪个短语在整篇里唯一出现"
只能看着真实原文判断。凭印象写引文（如「AI」「买入」）会命中几十个块，
构建器只能把整道题丢掉——最后 gold 集凑不满，指标也失去意义。

用法::

    python scripts/dump_chunks.py --out-dir _chunks
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXCERPT_CHARS = 300
_UNSAFE = re.compile(r"[^0-9A-Za-z\u4e00-\u9fff._-]+")


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):  # pragma: no cover
            pass

    parser = argparse.ArgumentParser(description="导出真实块内容供选引文")
    parser.add_argument("--out-dir", default="_chunks")
    parser.add_argument("--excerpt", type=int, default=EXCERPT_CHARS)
    args = parser.parse_args(argv)

    from finance_rag.src.core.config import KB_COLLECTION_NAME
    from finance_rag.src.infrastructure.vector_store.milvus_kb import get_knowledge_base

    kb = get_knowledge_base(KB_COLLECTION_NAME)
    client = kb._get_client()
    rows = client.query(
        collection_name=kb.collection_name,
        filter='tenant_id != ""',
        output_fields=["id", "source", "content", "start_page", "end_page", "block_type"],
        limit=16384,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("source") or ""), []).append(row)

    index_lines: list[str] = []
    for source in sorted(grouped):
        chunks = sorted(grouped[source], key=lambda r: (int(r.get("start_page") or 0), str(r.get("id"))))
        stem = _UNSAFE.sub("_", source)[:80]
        body = [f"### source={source}  chunks={len(chunks)}", ""]
        for row in chunks:
            content = str(row.get("content") or "").replace("\n", " ")
            page = f"p{row.get('start_page')}-{row.get('end_page')}"
            body.append(f"[{row.get('id')}] {page} {row.get('block_type')} len={len(content)}")
            body.append(f"    {content[: args.excerpt]}")
            body.append("")
        (out_dir / f"{stem}.txt").write_text("\n".join(body), encoding="utf-8")
        index_lines.append(f"{source}\t{len(chunks)}\t{stem}.txt")

    (out_dir / "_index.tsv").write_text("\n".join(index_lines), encoding="utf-8")
    print(f"导出 {len(grouped)} 篇 / {len(rows)} 个块 -> {out_dir}")
    for line in index_lines:
        print("  " + line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
