"""把 dump_chunks 的输出压成紧凑索引，便于快速定位可用引文。"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHUNK_DIR = ROOT / "_chunks"
HEAD = re.compile(r"^\[([0-9a-f]+)\]\s+(\S+)\s+(\S+)\s+len=(\d+)$")

limit = int(sys.argv[1]) if len(sys.argv) > 1 else 90
only = sys.argv[2] if len(sys.argv) > 2 else ""

out: list[str] = []
for path in sorted(CHUNK_DIR.glob("*.txt")):
    if only and only not in path.name:
        continue
    source = ""
    pending: tuple[str, str, str, str] | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("### source="):
            source = line.split("source=", 1)[1].split("  chunks=")[0]
            out.append("")
            out.append(f"### {source}")
            continue
        match = HEAD.match(line.strip())
        if match:
            pending = match.groups()  # type: ignore[assignment]
            continue
        if pending and line.startswith("    "):
            text = " ".join(line.strip().split())
            cid, page, btype, length = pending
            out.append(f"{cid[:8]} {page:>8} {btype:<16} {text[:limit]}")
            pending = None

target = CHUNK_DIR / (f"_compact_{only}.tsv" if only else "_compact.tsv")
target.write_text("\n".join(out), encoding="utf-8")
print(f"{target} lines={len(out)}")
