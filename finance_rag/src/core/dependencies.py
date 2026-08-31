"""全局依赖注入入口。

具体 provider 使用延迟导入，避免导入轻量核心模块时初始化外部服务依赖。
"""

from collections.abc import Generator
from typing import Any


def get_knowledge_base(collection_name: str | None = None) -> Any:
    """获取当前向量数据库知识库实例。"""
    from finance_rag.src.infrastructure.vector_store.milvus_kb import get_knowledge_base as factory

    if collection_name is None:
        return factory()
    return factory(collection_name)


def get_llm_client() -> Any:
    """获取全局 LLM 客户端；未配置时返回 None。"""
    from finance_rag.src.core.config import get_model

    return get_model()


def get_db_session() -> Generator[Any, None, None]:
    """获取 PostgreSQL 请求范围会话。"""
    from finance_rag.src.infrastructure.relational_db.postgres import get_session

    yield from get_session()


from finance_rag.src.api.dependencies import (  # noqa: E402,F401
    decode_access_token,
    get_current_user,
)
