"""JSON 结构化日志配置与请求上下文。"""

from __future__ import annotations

import json
import logging
import time
from contextvars import ContextVar
from typing import Any

request_id_var: ContextVar[str] = ContextVar("request_id", default="")


def get_request_id() -> str:
    """获取当前请求的关联 ID。"""
    return request_id_var.get() or ""


class StructuredLogger:
    """输出统一 JSON 事件的轻量日志封装。"""

    def __init__(self, name: str):
        self._logger = logging.getLogger(name)

    def _log(self, level: int, event: str, **fields: Any) -> None:
        entry = {
            "event": event,
            "timestamp": time.time(),
            "logger": self._logger.name,
        }
        request_id = get_request_id()
        if request_id:
            entry["request_id"] = request_id
        entry.update(fields)
        self._logger.log(
            level,
            json.dumps(entry, ensure_ascii=False, default=str),
        )

    def knowledge_retrieved(
        self,
        faq_hits: int,
        rule_hits: int,
        product_hits: int,
        elapsed_ms: float,
        **extra: Any,
    ) -> None:
        self._log(
            logging.DEBUG,
            "knowledge_retrieved",
            faq_hits=faq_hits,
            rule_hits=rule_hits,
            product_hits=product_hits,
            elapsed_ms=round(elapsed_ms, 2),
            **extra,
        )

    def rag_rerank_start(self, candidate_count: int, **extra: Any) -> None:
        self._log(
            logging.DEBUG,
            "rag_rerank_start",
            candidate_count=candidate_count,
            **extra,
        )

    def rag_rerank_end(
        self,
        elapsed_ms: float,
        result_count: int,
        **extra: Any,
    ) -> None:
        self._log(
            logging.DEBUG,
            "rag_rerank_end",
            elapsed_ms=round(elapsed_ms, 2),
            result_count=result_count,
            **extra,
        )


agent_logger = StructuredLogger("finance_rag")
