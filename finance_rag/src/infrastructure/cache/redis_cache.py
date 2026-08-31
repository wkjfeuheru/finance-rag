"""Redis 缓存适配器。"""

from __future__ import annotations

import json
from typing import Any

from redis import Redis

from finance_rag.src.core.config import REDIS_CACHE_PREFIX, REDIS_TIMEOUT_SECONDS, REDIS_URL


class RedisCache:
    """带命名空间隔离的 Redis JSON 缓存。"""

    def __init__(
        self,
        url: str = REDIS_URL,
        prefix: str = REDIS_CACHE_PREFIX,
        timeout: float = REDIS_TIMEOUT_SECONDS,
    ) -> None:
        self.prefix = prefix.strip(":")
        self._client = Redis.from_url(url, socket_timeout=timeout, decode_responses=True)

    def _key(self, key: str) -> str:
        return f"{self.prefix}:{key.lstrip(':')}"

    def get(self, key: str) -> Any | None:
        value = self._client.get(self._key(key))
        return json.loads(value) if value is not None else None

    def set(self, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        payload = json.dumps(value, ensure_ascii=False)
        self._client.set(self._key(key), payload, ex=ttl_seconds)

    def delete(self, key: str) -> None:
        self._client.delete(self._key(key))

    def healthcheck(self) -> bool:
        """检查 Redis 是否可用。"""
        return bool(self._client.ping())


redis_cache = RedisCache()
