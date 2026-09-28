"""存储模块 — 文件存储后端抽象层。

后端由 ``STORAGE_BACKEND`` 选择：

- ``local``：本地文件系统（``UPLOAD_DIR``，默认 ``<repo>/files``）
- ``oss`` / ``s3``：S3 兼容对象存储（``OSS_*`` 配置）

历史实现的这个函数**恒返回 OSS 后端**，``STORAGE_BACKEND`` 只在生产配置校验里被读到，
于是一个本地开发环境配了 ``STORAGE_BACKEND=local`` 仍会把原件往真实的
``oss-cn-hangzhou.aliyuncs.com`` 发——研报是授权材料，这个默认值不能留。
"""

from __future__ import annotations

import logging
import threading

from finance_rag.src.core.config import STORAGE_BACKEND

from .base import StorageBackend
from .local import LocalStorageBackend
from .oss import OSSStorageBackend

logger = logging.getLogger(__name__)

#: 选择本地后端的取值（其余一律按 S3 兼容对象存储处理）
_LOCAL_ALIASES = frozenset({"local", "file", "filesystem"})

_storage: StorageBackend | None = None
_storage_lock = threading.Lock()


def get_storage() -> StorageBackend:
    """按 ``STORAGE_BACKEND`` 获取存储后端单例（线程安全）。"""
    global _storage
    if _storage is not None:
        return _storage

    with _storage_lock:
        if _storage is None:
            if STORAGE_BACKEND.strip().lower() in _LOCAL_ALIASES:
                _storage = LocalStorageBackend()
            else:
                _storage = OSSStorageBackend()
            logger.info("存储后端：%s（STORAGE_BACKEND=%s）",
                        _storage.backend_name, STORAGE_BACKEND)
        return _storage


def reset_storage() -> None:
    """丢弃缓存的单例（切换后端或测试用）。"""
    global _storage
    with _storage_lock:
        _storage = None


__all__ = ["StorageBackend", "get_storage", "reset_storage"]
