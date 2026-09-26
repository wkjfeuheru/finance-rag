"""RAG 整表持久化存储（PostgreSQL）。

与 :mod:`finance_rag.src.rag.retrieval.parent_store` 同构：
``table_id -> 整表内容``，按集合隔离。检索侧据此把「表头 + 首行」的索引块
展开成完整表格，且**不受父块 3000 字符上限约束**。
"""

from __future__ import annotations

from typing import Any

from finance_rag.src.core.config import KB_COLLECTION_NAME
from finance_rag.src.infrastructure.relational_db.table_store import TableStoreRepository


class TableStore:
    """整表存储：``table_id -> {payload, markdown, ...}``。"""

    def __init__(self, collection: str = KB_COLLECTION_NAME) -> None:
        self._collection = collection
        self._repo = TableStoreRepository()

    def store_batch(self, items: list[dict[str, Any]]) -> None:
        self._repo.upsert_many(self._collection, items)

    def get(self, table_id: str) -> dict[str, Any] | None:
        return self._to_dict(self._repo.get_one(self._collection, table_id))

    def get_batch(self, table_ids: list[str]) -> list[dict[str, Any]]:
        rows = self._repo.get_many(self._collection, table_ids)
        return [self._to_dict(row) for row in rows]

    def delete_by_source(self, source: str) -> int:
        return self._repo.delete_by_source(self._collection, source)

    @staticmethod
    def _to_dict(row: dict[str, Any] | None) -> dict[str, Any] | None:
        if row is None:
            return None
        return {
            "id": row["id"],
            "parent_id": row.get("parent_id", ""),
            "payload": row.get("payload") or [],
            "markdown": row.get("markdown", ""),
            "row_count": int(row.get("row_count", 0) or 0),
            "source": row.get("source", ""),
            "heading_path": row.get("heading_path", ""),
            "start_page": int(row.get("start_page", 0) or 0),
        }
