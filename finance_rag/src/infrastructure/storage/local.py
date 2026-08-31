"""本地文件系统存储后端。

封装当前所有 ``Path.write_bytes / read_text / exists / unlink`` 操作，
提供与 StorageBackend 一致的异步接口。
"""

from __future__ import annotations

import logging
from pathlib import Path

from finance_rag.src.core.config import UPLOAD_DIR
from .base import StorageBackend

logger = logging.getLogger(__name__)


class LocalStorageBackend(StorageBackend):
    """本地文件系统存储后端。

    key 映射为 ``UPLOAD_DIR`` 下的相对子路径。
    """

    def __init__(self, root_dir: str | Path | None = None) -> None:
        self._root = Path(root_dir) if root_dir else Path(UPLOAD_DIR)
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def backend_name(self) -> str:
        return "local"

    def _resolve(self, key: str) -> Path:
        """将 key 解析为本地绝对路径。"""
        return self._root / key

    async def upload(self, key: str, data: bytes) -> None:
        path = self._resolve(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        logger.debug("本地上传：%s (%d bytes)", key, len(data))

    async def download(self, key: str) -> bytes:
        path = self._resolve(key)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在：{path}")
        return path.read_bytes()

    async def download_text(self, key: str, encoding: str = "utf-8") -> str:
        path = self._resolve(key)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在：{path}")
        return path.read_text(encoding=encoding)

    async def delete(self, key: str) -> None:
        path = self._resolve(key)
        if path.exists():
            path.unlink()
            logger.debug("本地删除：%s", key)

    async def exists(self, key: str) -> bool:
        return self._resolve(key).exists()

    async def download_to_path(self, key: str, local_path: Path) -> Path:
        """本地后端：key 已在此文件系统上，直接返回原路径。

        如果 key 对应的路径与 local_path 不同，则执行 copy2。
        """
        source = self._resolve(key)
        if not source.exists():
            raise FileNotFoundError(f"文件不存在：{source}")

        if source.resolve() == Path(local_path).resolve():
            return source

        import shutil
        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, local_path)
        return local_path

    def get_full_path(self, key: str) -> str:
        return str(self._resolve(key))

    # ------------------------------------------------------------------
    # 同步辅助方法（供非异步代码直接调用）
    # ------------------------------------------------------------------

    def upload_sync(self, key: str, data: bytes) -> None:
        path = self._resolve(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def download_sync(self, key: str) -> bytes:
        return self._resolve(key).read_bytes()

    def download_text_sync(self, key: str, encoding: str = "utf-8") -> str:
        return self._resolve(key).read_text(encoding=encoding)

    def exists_sync(self, key: str) -> bool:
        return self._resolve(key).exists()

    def delete_sync(self, key: str) -> None:
        path = self._resolve(key)
        if path.exists():
            path.unlink()

    def download_to_path_sync(self, key: str, local_path: Path) -> Path:
        """同步版本。"""
        source = self._resolve(key)
        if not source.exists():
            raise FileNotFoundError(f"文件不存在：{source}")
        if source.resolve() == Path(local_path).resolve():
            return source
        import shutil
        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, local_path)
        return local_path
