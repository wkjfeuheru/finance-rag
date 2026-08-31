"""SSE 流式输出适配。"""

from __future__ import annotations

import json
from typing import Any


def encode_event(event: dict[str, Any]) -> dict[str, str]:
    """将业务事件编码为 sse-starlette 可直接发送的事件结构。"""
    return {
        "event": str(event["type"]),
        "data": json.dumps(event, ensure_ascii=False),
    }


__all__ = ["encode_event"]
