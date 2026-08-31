"""文件哈希指纹存储，用于增量构建：跳过未变更文件的重复解析 + 嵌入。

以 JSON 文件持久化到 ``MILVUS_FINGERPRINT_PATH``，支持按 source 记录 SHA256、
SimHash、mtime、size、version 等元数据。写入采用「临时文件 + 原子替换」，
并加锁保护，避免并发更新互相覆盖或进程崩溃截断 JSON。
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from finance_rag.src.core.config import MILVUS_FINGERPRINT_PATH


class FingerprintStore:
    """文件哈希指纹存储。"""

    def __init__(self, store_path: str | None = None):
        self._path = Path(store_path) if store_path else Path(MILVUS_FINGERPRINT_PATH)
        self._data: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                self._data = json.loads(self._path.read_text("utf-8"))
            except (json.JSONDecodeError, OSError):
                self._data = {}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp_path.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2), "utf-8"
        )
        os.replace(tmp_path, self._path)

    def get(self, source: str) -> dict[str, Any] | None:
        """获取指定 source 的指纹记录，不存在返回 None。"""
        return self._data.get(source)

    def is_unchanged(self, source: str, file_hash: str) -> bool:
        """检查文件是否未变更（哈希一致）。"""
        record = self._data.get(source)
        return record is not None and record.get("sha256") == file_hash

    def is_changed_by_hash(self, source: str, content_hash: str) -> bool:
        """基于内容哈希判断文件是否变更。

        Returns:
            True 表示文件已变更或首次入库，False 表示未变更。
        """
        record = self.get(source)
        if record is None:
            return True  # 首次入库
        stored_hash = record.get("sha256", "")
        return stored_hash != content_hash

    def update(
        self,
        source: str,
        sha256: str,
        file_path: str,
        chunk_count: int,
        *,
        simhash: int | None = None,
        mtime: float | None = None,
        size: int | None = None,
        version: str | None = None,
    ) -> None:
        """更新（或创建）指纹记录；可选记录元数据检测与去重所需的扩展字段。"""
        record: dict[str, Any] = {
            "sha256": sha256,
            "file_path": file_path,
            "chunk_count": chunk_count,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        }
        if simhash is not None:
            record["simhash"] = simhash
        if mtime is not None:
            record["mtime"] = mtime
        if size is not None:
            record["size"] = size
        if version is not None:
            record["version"] = version
        with self._lock:
            self._data[source] = record
            self._save()

    def get_all_simhashes(self) -> dict[str, int]:
        """返回 {source: simhash}，仅包含已记录 simhash 的条目。"""
        return {
            source: record["simhash"]
            for source, record in self._data.items()
            if isinstance(record.get("simhash"), int)
        }

    def remove(self, source: str) -> None:
        """删除指纹记录。"""
        with self._lock:
            self._data.pop(source, None)
            self._save()

    @staticmethod
    def hash_file(file_path: str | Path) -> str:
        """计算文件 SHA256 哈希。"""
        sha = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                sha.update(chunk)
        return sha.hexdigest()


__all__ = ["FingerprintStore"]
