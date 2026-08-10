"""Windows compatibility helpers for the ablation benchmark."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Any

from .ablation import source_files
from finance_rag.src.infrastructure.vector_store.milvus_kb import KnowledgeBase


def _ascii_staged_path(path: Path, staging_dir: Path) -> Path:
    """Copy an input to an ASCII-only path for Docling on Windows."""
    staging_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:16]
    target = staging_dir / f"document_{digest}{path.suffix.lower()}"
    if not target.exists() or target.stat().st_mtime_ns < path.stat().st_mtime_ns:
        shutil.copy2(path, target)
    return target


def prepare_collections_windows_safe(
    kbs: dict[str, KnowledgeBase], data_dir: Path,
    staging_dir: Path | None = None,
) -> dict[str, Any]:
    files = source_files(data_dir)
    if not files:
        raise ValueError(f"数据目录没有可入库文档：{data_dir}")
    staging_dir = staging_dir or data_dir.parent / "artifacts" / "benchmark_inputs"
    staged = {path: _ascii_staged_path(path, staging_dir) for path in files}
    result: dict[str, Any] = {}
    for chunking, kb in kbs.items():
        documents = []
        for original in files:
            documents.append(kb.add_document(
                staged[original], source=original.name, title=original.stem
            ))
        stats = kb.get_stats()
        expected = {path.name for path in files}
        actual = {item["source"] for item in kb.list_documents()}
        if expected - actual:
            raise RuntimeError(f"{chunking} 集合缺少来源：{sorted(expected - actual)}")
        if stats.get("chunk_count", 0) <= 0:
            raise RuntimeError(f"{chunking} 集合没有有效切块")
        result[chunking] = {
            "collection": kb.collection_name,
            "documents": documents,
            "stats": stats,
            "input_staging_dir": str(staging_dir),
        }
    return result
