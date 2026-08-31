"""存储后端抽象基类。"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from pathlib import Path


class StorageBackend(ABC):
    """文件存储后端抽象接口。

    所有具体实现（Local / S3）必须实现以下方法。
    key 是存储标识符，对本地后端映射为相对 ``UPLOAD_DIR`` 的子路径，
    对 S3 后端映射为 bucket 中的对象 key。
    """

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """后端名称标识（如 'local' / 's3'）。"""
        ...

    @abstractmethod
    async def upload(self, key: str, data: bytes) -> None:
        """上传文件内容到指定 key。"""
        ...

    @abstractmethod
    async def download(self, key: str) -> bytes:
        """从指定 key 下载文件内容。"""
        ...

    @abstractmethod
    async def download_text(self, key: str, encoding: str = "utf-8") -> str:
        """从指定 key 下载文本文件内容。"""
        ...

    @abstractmethod
    async def delete(self, key: str) -> None:
        """删除指定 key 的文件。"""
        ...

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """检查指定 key 是否存在。"""
        ...

    async def get_hash(self, key: str, algorithm: str = "sha256") -> str:
        """获取指定 key 文件的内容哈希（默认 SHA256）。

        默认实现：下载全部内容后计算。子类可重写以使用服务端哈希。
        """
        data = await self.download(key)
        return hashlib.new(algorithm, data).hexdigest()

    @abstractmethod
    async def download_to_path(self, key: str, local_path: Path) -> Path:
        """将指定 key 的内容下载到本地文件路径，返回该路径。

        用于 PDF 版面解析等必须依赖本地文件的场景。
        """
        ...

    def delete_sync(self, key: str) -> None:
        """同步删除指定 key 的文件。对象不存在时静默忽略。"""
        raise NotImplementedError(f"{self.backend_name} 后端未实现 delete_sync")

    def get_full_path(self, key: str) -> str:
        """获取 key 的完整路径（仅本地后端有效）。

        OSS 后端抛出 NotImplementedError。
        """
        raise NotImplementedError(f"{self.backend_name} 后端不支持文件系统路径")
