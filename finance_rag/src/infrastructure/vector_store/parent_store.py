"""父块 SQLite KV 存储。

提供 parent_id -> 父块全文的持久化存储，支持单条/批量读写和按 source 删除。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from config.settings import PARENT_STORE_PATH

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS parent_chunks (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    heading TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL DEFAULT ''
)
"""


class ParentStore:
    """SQLite KV 存储：parent_id -> 父块全文。"""

    def __init__(self, db_path: str | Path | None = None) -> None:
        path = Path(db_path) if db_path else Path(PARENT_STORE_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = str(path)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(_CREATE_SQL)
            conn.commit()

    def store(self, parent_id: str, content: str, heading: str = "", source: str = "") -> None:
        """单条插入或替换。"""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO parent_chunks (id, content, heading, source) VALUES (?, ?, ?, ?)",
                (parent_id, content, heading, source),
            )
            conn.commit()

    def store_batch(self, items: list[dict[str, Any]]) -> None:
        """批量插入/替换。

        items 是 dict 列表，每个 dict 需包含 ``id``、``content``、``heading``、``source`` 键。
        """
        with sqlite3.connect(self._db_path) as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO parent_chunks (id, content, heading, source) VALUES (?, ?, ?, ?)",
                [
                    (
                        item["id"],
                        item["content"],
                        item.get("heading", ""),
                        item.get("source", ""),
                    )
                    for item in items
                ],
            )
            conn.commit()

    def get(self, parent_id: str) -> dict[str, str] | None:
        """查询单条，返回 dict（含 id/content/heading/source）或 None。"""
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT id, content, heading, source FROM parent_chunks WHERE id = ?",
                (parent_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "id": row[0],
            "content": row[1],
            "heading": row[2],
            "source": row[3],
        }

    def get_batch(self, parent_ids: list[str]) -> list[dict[str, str]]:
        """批量查询，返回 dict 列表。"""
        if not parent_ids:
            return []
        placeholders = ",".join("?" * len(parent_ids))
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                f"SELECT id, content, heading, source FROM parent_chunks WHERE id IN ({placeholders})",
                parent_ids,
            ).fetchall()
        return [
            {
                "id": row[0],
                "content": row[1],
                "heading": row[2],
                "source": row[3],
            }
            for row in rows
        ]

    def delete_by_source(self, source: str) -> int:
        """按 source 删除，返回删除的行数。"""
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM parent_chunks WHERE source = ?",
                (source,),
            )
            conn.commit()
            return cursor.rowcount
