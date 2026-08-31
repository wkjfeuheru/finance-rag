"""Knowledge-base metadata repository backed by PostgreSQL."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    Column,
    DateTime,
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
knowledge_bases = Table(
    "knowledge_bases",
    metadata,
    Column("name", String(32), primary_key=True),
    Column("display_name", String(128), nullable=False),
    Column("description", Text, nullable=False, default=""),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)


class KnowledgeBaseRepository:
    """Persist knowledge-base metadata in the relational database."""

    def __init__(self, database_url: str = DATABASE_URL) -> None:
        self.database_url = database_url
        self._engine = None

    def _get_engine(self):
        if self._engine is None:
            self._engine = create_engine(self.database_url, pool_pre_ping=True)
            metadata.create_all(self._engine, tables=[knowledge_bases])
        return self._engine

    def load(self) -> dict[str, Any]:
        """Return the registry-shaped payload used by the service layer."""
        with self._get_engine().connect() as connection:
            rows = connection.execute(
                select(knowledge_bases).order_by(knowledge_bases.c.created_at)
            ).mappings().all()
        return {
            "knowledge_bases": [
                {
                    "name": row["name"],
                    "display_name": row["display_name"],
                    "description": row["description"],
                    "created_at": (
                        row["created_at"].replace(tzinfo=timezone.utc)
                        if row["created_at"].tzinfo is None
                        else row["created_at"].astimezone(timezone.utc)
                    ).isoformat(),
                }
                for row in rows
            ],
            # 空表视为「未播种」，触发 _ensure_builtin_registered 自动补全内置类别
            "builtin_seeded": bool(rows),
        }

    def save(self, data: dict[str, Any]) -> None:
        """Replace registry metadata atomically in one database transaction."""
        now = datetime.now(timezone.utc)
        rows = []
        for entry in data.get("knowledge_bases", []):
            created_at = entry.get("created_at") or now.isoformat()
            parsed_created_at = datetime.fromisoformat(created_at)
            rows.append(
                {
                    "name": entry["name"],
                    "display_name": entry.get("display_name", entry["name"]),
                    "description": entry.get("description", ""),
                    "created_at": parsed_created_at,
                    "updated_at": now,
                }
            )

        with self._get_engine().begin() as connection:
            connection.execute(delete(knowledge_bases))
            if rows:
                connection.execute(insert(knowledge_bases), rows)
