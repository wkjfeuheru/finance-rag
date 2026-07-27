from __future__ import annotations

"""Structured JSON logging for the standalone RAG pipeline."""

import json
import logging
import time
from contextvars import ContextVar

request_id_var: ContextVar[str] = ContextVar("request_id", default="")


def set_request_id(request_id: str) -> None:
    request_id_var.set(request_id)


def get_request_id() -> str:
    return request_id_var.get() or ""


class StructuredLogger:
    def __init__(self, name: str):
        self._logger = logging.getLogger(name)

    def _log(self, level: int, event: str, **fields) -> None:
        entry = {
            "event": event,
            "timestamp": time.time(),
            "logger": self._logger.name,
        }
        req_id = get_request_id()
        if req_id:
            entry["request_id"] = req_id
        entry.update(fields)
        self._logger.log(level, json.dumps(entry, ensure_ascii=False, default=str))

    def knowledge_retrieved(self, faq_hits: int, rule_hits: int, product_hits: int, elapsed_ms: float, **extra) -> None:
        self._log(
            logging.DEBUG,
            "knowledge_retrieved",
            faq_hits=faq_hits,
            rule_hits=rule_hits,
            product_hits=product_hits,
            elapsed_ms=round(elapsed_ms, 2),
            **extra,
        )

    def rag_rerank_start(self, candidate_count: int, **extra) -> None:
        self._log(logging.DEBUG, "rag_rerank_start", candidate_count=candidate_count, **extra)

    def rag_rerank_end(self, elapsed_ms: float, result_count: int, **extra) -> None:
        self._log(
            logging.DEBUG,
            "rag_rerank_end",
            elapsed_ms=round(elapsed_ms, 2),
            result_count=result_count,
            **extra,
        )

    def rag_eval_result(self, metric: str, value: float, k: int, **extra) -> None:
        self._log(
            logging.INFO,
            "rag_eval_result",
            metric=metric,
            value=round(value, 4),
            k=k,
            **extra,
        )


agent_logger = StructuredLogger("finance_rag")
