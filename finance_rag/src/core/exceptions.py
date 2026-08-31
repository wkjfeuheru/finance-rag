"""全链路统一的异常类型、分类与友好文案工具。

各基础设施层（向量库、嵌入、对象存储、LLM）在失败时抛出这里的具名异常，
上层（查询链路 / API 路由 / 任务）据此：
* 区分「基础设施故障」与「业务空结果」；
* 给出面向用户/运维的中文说明；
* 保留原始异常链（``raise ... from exc``）以便日志定位。

设计原则：只做分类与文案，不吞异常；具体降级策略仍由各调用方决定。
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 异常层级
# ---------------------------------------------------------------------------

class RagError(Exception):
    """RAG 全链路业务异常的基类。"""


class VectorStoreError(RagError):
    """向量数据库查询/写入异常（连接、超时、集合操作等）。"""


class VectorStoreUnavailableError(VectorStoreError):
    """向量数据库不可用（连接失败、集合加载失败）。"""


class EmbeddingError(RagError):
    """嵌入模型加载/推理失败。"""


class StorageError(RagError):
    """对象存储（本地/OSS）操作失败。"""


class LlmError(RagError):
    """LLM 调用失败（连接、超时、限流等）。"""


class RetrievalError(RagError):
    """RAG 检索流程失败。"""


class AgentLoopLimit(RagError):
    """Agent 反思或重检索循环达到上限。"""


# ---------------------------------------------------------------------------
# 分类
# ---------------------------------------------------------------------------

def classify_llm_error(exc: Exception) -> str:
    """把 LLM 调用异常归为 ``timeout|rate_limit|connection|other`` 之一。

    兼容 httpx 异常、OpenAI/DeepSeek SDK 异常以及携带关键字的字符串异常。
    """
    import httpx

    if isinstance(exc, (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.WriteTimeout)):
        return "timeout"
    if isinstance(exc, httpx.ReadError):
        return "connection"

    error_str = str(exc).lower()

    # 429 / 限流关键字
    for keyword in ("429", "rate limit", "rate_limit", "too many requests", "quota"):
        if keyword in error_str:
            return "rate_limit"
    # 超时关键字
    for keyword in ("timeout", "timed out", "read error"):
        if keyword in error_str:
            return "timeout"
    # 连接关键字
    for keyword in ("connection", "connect error", "refused", "reset"):
        if keyword in error_str:
            return "connection"
    return "other"


def is_retryable_llm_error(exc: Exception) -> bool:
    """判断异常是否值得重试（超时 / 限流 / 连接类瞬时错误）。"""
    return classify_llm_error(exc) in ("timeout", "rate_limit", "connection")


# ---------------------------------------------------------------------------
# 友好文案
# ---------------------------------------------------------------------------

def friendly_message(exc: BaseException) -> str:
    """把异常映射为面向用户的中文说明；未知异常回退到 ``str(exc)``。"""
    if isinstance(exc, VectorStoreUnavailableError):
        return (
            "向量数据库连接失败，请检查 MILVUS_URI 与向量数据库服务状态后重试"
        )
    if isinstance(exc, VectorStoreError):
        return f"向量数据库操作失败：{exc}"
    if isinstance(exc, EmbeddingError):
        return f"嵌入模型失败：{exc}"
    if isinstance(exc, StorageError):
        return f"对象存储失败：{exc}"
    if isinstance(exc, LlmError):
        return f"大模型调用失败：{exc}"
    if isinstance(exc, RagError):
        return str(exc)

    # 数据库连接类异常（psycopg / SQLAlchemy OperationalError）单独归类，
    # 避免被下方 classify_llm_error 的 "connection" 关键字误判为「模型服务连接失败」
    if isinstance(exc, Exception):
        exc_module = type(exc).__module__ or ""
        exc_name = type(exc).__name__ or ""
        is_db_error = exc_module.startswith(("psycopg", "psycopg2", "sqlalchemy")) or exc_name in {
            "OperationalError",
            "InterfaceError",
            "DatabaseError",
            "ProgrammingError",
        }
        if is_db_error:
            return "数据库连接失败，请检查 DATABASE_URL 与数据库服务状态后重试"

    kind = classify_llm_error(exc) if isinstance(exc, Exception) else "other"
    if kind == "timeout":
        return "模型调用超时，请稍后重试"
    if kind == "rate_limit":
        return "模型调用频率受限，请稍后重试"
    if kind == "connection":
        return "模型服务连接失败，请检查网络与 API 配置"

    return str(exc)
