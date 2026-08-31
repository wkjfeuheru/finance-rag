"""父块全文仓储（PostgreSQL）。"""

from __future__ import annotations

from typing import Any

from sqlalchemy import (
    Column,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    delete,
    insert,
    select,
)

from finance_rag.src.core.config import DATABASE_URL


metadata = MetaData()
parent_chunks = Table(
    "parent_chunks",
    metadata,
    # 复合主键 (collection, id)：同一父块在不同知识库（集合）下隔离存储
    Column("collection", String(64), primary_key=True),
    Column("id", String(64), primary_key=True),
    Column("content", Text, nullable=False),
    Column("heading", String(512), nullable=False, default=""),
    Column("source", String(512), nullable=False, default=""),
)


class ParentStoreRepository:
    """父块全文持久化到关系型数据库，按 ``collection`` 隔离。"""

    def __init__(self, database_url: str = DATABASE_URL) -> None:
        self.database_url = database_url
        self._engine = None

    def _get_engine(self):
        if self._engine is None:
            self._engine = create_engine(self.database_url, pool_pre_ping=True)
            metadata.create_all(self._engine, tables=[parent_chunks])
        return self._engine

    def upsert_many(self, collection: str, items: list[dict[str, Any]]) -> None:
        """批量写入/覆盖父块（等价 ``INSERT OR REPLACE``，事务内 delete+insert）。"""
        if not items:
            return
        ids = [str(item["id"]) for item in items]
        rows = [
            {
                "collection": collection,
                "id": str(item["id"]),
                "content": item["content"],
                "heading": item.get("heading", ""),
                "source": item.get("source", ""),
            }
            for item in items
        ]
        with self._get_engine().begin() as connection:
            connection.execute(
                delete(parent_chunks).where(
                    parent_chunks.c.collection == collection,
                    parent_chunks.c.id.in_(ids),
                )
            )
            connection.execute(insert(parent_chunks), rows)

    def get_one(self, collection: str, parent_id: str) -> dict[str, Any] | None:
        with self._get_engine().connect() as connection:
            row = connection.execute(
                select(parent_chunks).where(
                    parent_chunks.c.collection == collection,
                    parent_chunks.c.id == parent_id,
                )
            ).mappings().first()
        return dict(row) if row is not None else None

    def get_many(self, collection: str, parent_ids: list[str]) -> list[dict[str, Any]]:
        if not parent_ids:
            return []
        with self._get_engine().connect() as connection:
            rows = connection.execute(
                select(parent_chunks).where(
                    parent_chunks.c.collection == collection,
                    parent_chunks.c.id.in_(parent_ids),
                )
            ).mappings().all()
        return [dict(row) for row in rows]

    def delete_by_source(self, collection: str, source: str) -> int:
        with self._get_engine().begin() as connection:
            result = connection.execute(
                delete(parent_chunks).where(
                    parent_chunks.c.collection == collection,
                    parent_chunks.c.source == source,
                )
            )
        return result.rowcount
