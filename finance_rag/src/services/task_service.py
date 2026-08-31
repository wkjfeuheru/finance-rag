"""后台异步任务状态管理。

为 PDF 等耗时文档的上传解析提供异步任务跟踪能力：
上传接口立即返回 task_id，前端轮询状态直到完成。
"""

from __future__ import annotations

import enum
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


class TaskStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Task:
    id: str
    status: TaskStatus = TaskStatus.PENDING
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    filename: str = ""
    result: dict[str, Any] | None = None
    error: str | None = None


class TaskManager:
    """内存任务状态存储，适合单实例部署。"""

    def __init__(self) -> None:
        self._tasks: dict[str, Task] = {}

    def create(self, filename: str = "") -> Task:
        task_id = uuid.uuid4().hex[:12]
        task = Task(id=task_id, filename=filename)
        self._tasks[task_id] = task
        return task

    def get(self, task_id: str) -> Task | None:
        return self._tasks.get(task_id)

    def update(
        self,
        task_id: str,
        *,
        status: TaskStatus | None = None,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> Task | None:
        task = self._tasks.get(task_id)
        if task is None:
            return None
        if status is not None:
            task.status = status
        if result is not None:
            task.result = result
        if error is not None:
            task.error = error
        task.updated_at = time.time()
        return task

    def cleanup(self, max_age_seconds: float = 3600) -> int:
        """清理超过 max_age_seconds 的已完成/失败任务，返回清理数量。"""
        cutoff = time.time() - max_age_seconds
        to_delete = [
            tid
            for tid, t in self._tasks.items()
            if t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED)
            and t.updated_at < cutoff
        ]
        for tid in to_delete:
            del self._tasks[tid]
        return len(to_delete)


# 全局单例
_task_manager: TaskManager | None = None
_task_manager_lock = threading.Lock()


def get_task_manager() -> TaskManager:
    global _task_manager
    if _task_manager is None:
        with _task_manager_lock:
            if _task_manager is None:
                _task_manager = TaskManager()
    return _task_manager
