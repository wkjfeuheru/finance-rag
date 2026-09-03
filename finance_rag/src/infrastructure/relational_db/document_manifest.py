"""Document/version/chunk manifest persistence for event-driven indexing."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import Column, DateTime, Integer, MetaData, String, Table, Text, create_engine, insert, select, update

from finance_rag.src.core.config import DATABASE_URL

metadata = MetaData()
documents = Table(
    "documents", metadata,
    Column("document_id", String(128), primary_key=True),
    Column("tenant_id", String(64), nullable=False),
    Column("source", String(512), nullable=False),
    Column("title", String(256), nullable=False),
    Column("current_version_id", String(128), nullable=True),
    Column("status", String(32), nullable=False, default="processing"),
    Column("category", String(64), nullable=False, default=""),
    Column("date", String(32), nullable=False, default=""),
    Column("object_key", String(1024), nullable=False, default=""),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    Column("deleted_at", DateTime(timezone=True), nullable=True),
)
document_versions = Table(
    "document_versions", metadata,
    Column("version_id", String(128), primary_key=True),
    Column("document_id", String(128), nullable=False),
    Column("object_version_id", String(256), nullable=False, default=""),
    Column("etag", String(256), nullable=False, default=""),
    Column("file_sha256", String(64), nullable=False),
    Column("status", String(32), nullable=False, default="processing"),
    Column("parser_version", String(128), nullable=False, default="mineru"),
    Column("chunk_count", Integer, nullable=False, default=0),
    Column("error", Text, nullable=False, default=""),
    Column("created_at", DateTime(timezone=True), nullable=False),
)
document_chunks = Table(
    "document_chunks", metadata,
    Column("chunk_id", String(128), primary_key=True),
    Column("document_id", String(128), nullable=False),
    Column("version_id", String(128), nullable=False),
    Column("chunk_key", String(128), nullable=False),
    Column("content_hash", String(64), nullable=False),
    Column("parent_id", String(128), nullable=False),
    Column("position", Integer, nullable=False),
    Column("state", String(32), nullable=False, default="active"),
)


class DocumentManifestRepository:
    """Small repository used by the incremental indexer and object events."""

    def __init__(self, database_url: str = DATABASE_URL) -> None:
        self._engine = create_engine(database_url, pool_pre_ping=True)

    def ensure_schema(self) -> None:
        metadata.create_all(self._engine, tables=[documents, document_versions, document_chunks])

    def get_document(self, document_id: str) -> dict[str, Any] | None:
        self.ensure_schema()
        with self._engine.connect() as connection:
            row = connection.execute(select(documents).where(documents.c.document_id == document_id)).mappings().first()
        return dict(row) if row else None

    def upsert_document(self, values: dict[str, Any]) -> None:
        self.ensure_schema()
        now = datetime.now(timezone.utc)
        payload = {"updated_at": now, **values}
        with self._engine.begin() as connection:
            existing = connection.execute(select(documents.c.document_id).where(documents.c.document_id == values["document_id"])).first()
            if existing:
                connection.execute(update(documents).where(documents.c.document_id == values["document_id"]).values(payload))
            else:
                connection.execute(insert(documents).values(payload))

    def create_version(self, values: dict[str, Any]) -> None:
        self.ensure_schema()
        with self._engine.begin() as connection:
            connection.execute(insert(document_versions).values(created_at=datetime.now(timezone.utc), **values))

    def set_version_status(self, version_id: str, status: str, *, error: str = "") -> None:
        self.ensure_schema()
        with self._engine.begin() as connection:
            connection.execute(update(document_versions).where(document_versions.c.version_id == version_id).values(status=status, error=error))

    def replace_chunks(self, rows: list[dict[str, Any]]) -> None:
        self.ensure_schema()
        if not rows:
            return
        version_id = rows[0]["version_id"]
        with self._engine.begin() as connection:
            connection.execute(document_chunks.delete().where(document_chunks.c.version_id == version_id))
            connection.execute(insert(document_chunks), rows)

    def list_documents(self, *, include_deleted: bool = False) -> list[dict[str, Any]]:
        self.ensure_schema()
        statement = select(documents).order_by(documents.c.updated_at.desc())
        if not include_deleted:
            statement = statement.where(documents.c.status != "deleted")
        with self._engine.connect() as connection:
            return [dict(row) for row in connection.execute(statement).mappings().all()]


__all__ = ["DocumentManifestRepository"]
