"""审计日志模块。

记录所有破坏性操作：文档上传、文档删除、策略评估、登录尝试。
输出为结构化 JSON 行，可被 ELK/Splunk 等日志平台采集。
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

logger = logging.getLogger("finance_rag.audit")

# 审计日志专用 handler（INFO 级别写入 audit.log）
_audit_handler: logging.Handler | None = None


def _ensure_handler() -> None:
    global _audit_handler
    if _audit_handler is not None:
        return
    from pathlib import Path
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    _audit_handler = logging.FileHandler(
        log_dir / "audit.log", encoding="utf-8"
    )
    _audit_handler.setLevel(logging.INFO)
    _audit_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_audit_handler)
    logger.propagate = False


def log(
    action: str,
    *,
    user: str = "anonymous",
    resource: str = "",
    detail: str = "",
    result: str = "success",
    **extra: Any,
) -> None:
    """记录一条审计日志。

    Args:
        action: 操作类型（upload, delete, evaluate, login, chat）
        user: 操作用户
        resource: 操作对象（文件名、source 等）
        detail: 补充信息
        result: 结果（success, failure, skipped）
        **extra: 其他自定义字段
    """
    _ensure_handler()
    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        "action": action,
        "user": user,
        "resource": resource,
        "detail": detail,
        "result": result,
        **extra,
    }
    logger.info(json.dumps(entry, ensure_ascii=False, default=str))
