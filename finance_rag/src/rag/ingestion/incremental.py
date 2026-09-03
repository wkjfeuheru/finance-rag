"""Pure helpers for versioned chunk manifests and incremental indexing."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class ChunkFingerprint:
    chunk_key: str
    content_hash: str
    parent_id: str
    position: int
    metadata: dict[str, Any]


def content_hash(content: str) -> str:
    return hashlib.sha256(content.strip().encode("utf-8")).hexdigest()


def build_chunk_fingerprint(content: str, *, parent_id: str, position: int, metadata: dict[str, Any] | None = None) -> ChunkFingerprint:
    digest = content_hash(content)
    key = hashlib.sha256(f"{parent_id}:{digest}".encode("utf-8")).hexdigest()
    return ChunkFingerprint(key, digest, parent_id, position, dict(metadata or {}))


def diff_chunks(old: Iterable[ChunkFingerprint], new: Iterable[ChunkFingerprint]) -> dict[str, list[ChunkFingerprint]]:
    """Diff manifests; unchanged content is reusable even when position changes."""
    old_by_key = {item.chunk_key: item for item in old}
    new_by_key = {item.chunk_key: item for item in new}
    unchanged: list[ChunkFingerprint] = []
    changed: list[ChunkFingerprint] = []
    for key, item in new_by_key.items():
        previous = old_by_key.get(key)
        if previous and previous.content_hash == item.content_hash:
            unchanged.append(item)
        else:
            changed.append(item)
    removed = [item for key, item in old_by_key.items() if key not in new_by_key]
    return {"unchanged": unchanged, "changed": changed, "removed": removed}


def manifest_from_documents(documents: Iterable[Any]) -> list[ChunkFingerprint]:
    result = []
    for position, document in enumerate(documents):
        metadata = dict(getattr(document, "metadata", {}) or {})
        text = str(getattr(document, "page_content", "") or "")
        parent_id = str(metadata.get("parent_id", ""))
        result.append(build_chunk_fingerprint(text, parent_id=parent_id, position=position, metadata=metadata))
    return result


__all__ = ["ChunkFingerprint", "build_chunk_fingerprint", "content_hash", "diff_chunks", "manifest_from_documents"]
