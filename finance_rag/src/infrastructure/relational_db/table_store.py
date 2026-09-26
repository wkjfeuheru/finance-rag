"""整表仓储（PostgreSQL）。

向量库只存「表头 + 首行」作索引，**表体**存在这里：二维数组（JSONB）+ 原始
Markdown。这样做的好处是整表不受 :data:`PARENT_MAX_CHARS` 之类的父块上限约束。

与 :mod:`parent_store` 同构：复合主键 ``(collection, id)`` 做集合隔离，
建表走 ``metadata.create_all`` 自愈，写入是事务内 ``delete + insert`` 的 upsert。
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import (
    Column,
    Integer,
    JSON,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    delete,
    insert,
    select,
)
from sqlalchemy.dialects.postgresql import JSONB

from finance_rag.src.core.config import DATABASE_URL


metadata = MetaData()
table_chunks = Table(
    "table_chunks",
    metadata,
    # 复合主键 (collection, id)：同一张表在不同知识库（集合）下隔离存储
    Column("collection", String(64), primary_key=True),
    Column("id", String(64), primary_key=True),
    # 所属父块，便于按父块回溯上下文（与向量库里的 table_id 关联）
    Column("parent_id", String(64), nullable=False, default=""),
    # 二维数组形式的结构化表体；PostgreSQL 用 JSONB，其它方言降级为 JSON
    Column("payload", JSON().with_variant(JSONB, "postgresql"), nullable=False),
    # 原始 Markdown 表，供「查看完整表」原样展示
    Column("markdown", Text, nullable=False, default=""),
    Column("row_count", Integer, nullable=False, default=0),
    Column("source", String(512), nullable=False, default=""),
    Column("heading_path", String(1024), nullable=False, default=""),
    Column("start_page", Integer, nullable=False, default=0),
)


class TableStoreRepository:
    """整表持久化到关系型数据库，按 ``collection`` 隔离。"""

    def __init__(self, database_url: str = DATABASE_URL) -> None:
        self.database_url = database_url
        self._engine = None

    def _get_engine(self):
        if self._engine is None:
            self._engine = create_engine(self.database_url, pool_pre_ping=True)
            metadata.create_all(self._engine, tables=[table_chunks])
        return self._engine

    def upsert_many(self, collection: str, items: list[dict[str, Any]]) -> None:
        """批量写入/覆盖整表（等价 ``INSERT OR REPLACE``，事务内 delete+insert）。"""
        if not items:
            return
        ids = [str(item["id"]) for item in items]
        rows = [
            {
                "collection": collection,
                "id": str(item["id"]),
                "parent_id": str(item.get("parent_id", "")),
                "payload": item.get("payload") or [],
                "markdown": item.get("markdown", ""),
                "row_count": int(item.get("row_count", 0)),
                "source": item.get("source", ""),
                "heading_path": str(item.get("heading_path", ""))[:1024],
                "start_page": int(item.get("start_page", 0)),
            }
            for item in items
        ]
        with self._get_engine().begin() as connection:
            connection.execute(
                delete(table_chunks).where(
                    table_chunks.c.collection == collection,
                    table_chunks.c.id.in_(ids),
                )
            )
            connection.execute(insert(table_chunks), rows)

    def get_one(self, collection: str, table_id: str) -> dict[str, Any] | None:
        rows = self.get_many(collection, [table_id])
        return rows[0] if rows else None

    def get_many(self, collection: str, table_ids: list[str]) -> list[dict[str, Any]]:
        if not table_ids:
            return []
        with self._get_engine().connect() as connection:
            rows = connection.execute(
                select(table_chunks).where(
                    table_chunks.c.collection == collection,
                    table_chunks.c.id.in_(table_ids),
                )
            ).mappings().all()
        return [dict(row) for row in rows]

    def delete_by_source(self, collection: str, source: str) -> int:
        with self._get_engine().begin() as connection:
            result = connection.execute(
                delete(table_chunks).where(
                    table_chunks.c.collection == collection,
                    table_chunks.c.source == source,
                )
            )
        return result.rowcount
