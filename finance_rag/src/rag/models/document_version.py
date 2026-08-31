"""从文件名或元数据提取文档版本。"""

from __future__ import annotations

import re
from pathlib import Path

_VERSION_RE = re.compile(r"[_\-（(]?v(\d+(?:\.\d+)*)\s*[)）]?\s*(?=\.[^.]*$|$)", re.IGNORECASE)


def extract_document_version(source: str, metadata: dict | None = None) -> str:
    """优先从元数据提取版本，否则从文件名后缀提取。"""
    if metadata:
        explicit = str(metadata.get("version") or "").strip()
        if explicit:
            return explicit

    match = _VERSION_RE.search(Path(source).stem)
    return match.group(1) if match else ""
