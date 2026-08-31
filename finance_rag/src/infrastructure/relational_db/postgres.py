"""PostgreSQL 会话和事务适配器。"""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from finance_rag.src.core.config import DATABASE_URL


class PostgresDatabase:
    """惰性创建的 PostgreSQL 数据库会话工厂。"""

    def __init__(self, database_url: str = DATABASE_URL) -> None:
        self.database_url = database_url
        self._session_factory: sessionmaker[Session] | None = None

    def _get_session_factory(self) -> sessionmaker[Session]:
        if self._session_factory is None:
            engine = create_engine(self.database_url, pool_pre_ping=True)
            self._session_factory = sessionmaker(bind=engine, expire_on_commit=False)
        return self._session_factory

    def session(self) -> Generator[Session, None, None]:
        """提供一次请求范围的会话，并在结束时释放资源。"""
        db = self._get_session_factory()()
        try:
            yield db
        finally:
            db.close()

    def healthcheck(self) -> Any:
        """执行轻量连接检查，供基础设施探针使用。"""
        with self._get_session_factory()() as db:
            return db.execute(text("SELECT 1")).scalar()


postgres_database = PostgresDatabase()


def get_session() -> Generator[Session, None, None]:
    """获取默认 PostgreSQL 会话依赖。"""
    yield from postgres_database.session()
