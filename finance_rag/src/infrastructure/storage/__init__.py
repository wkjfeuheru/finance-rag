"""存储模块 — 文件存储后端抽象层。

统一使用阿里云 OSS 对象存储，存储桶由外部基础设施预先创建。
"""

from __future__ import annotations

import logging
import threading

from .base import StorageBackend
from .oss import OSSStorageBackend

logger = logging.getLogger(__name__)

_storage: StorageBackend | None = None
_storage_lock = threading.Lock()


def get_storage() -> StorageBackend:
    """获取 OSS 存储后端单例（线程安全）。"""
    global _storage
    if _storage is not None:
        return _storage

    with _storage_lock:
        if _storage is None:
            _storage = OSSStorageBackend()
            logger.info("存储后端：%s", _storage.backend_name)
        return _storage
