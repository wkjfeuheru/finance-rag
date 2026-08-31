import importlib
import json

import pytest

pytest.importorskip("redis")

from finance_rag.src.infrastructure.cache.redis_cache import RedisCache

redis_module = importlib.import_module(
    "finance_rag.src.infrastructure.cache.redis_cache"
)


class FakeRedis:
    def __init__(self):
        self.values = {}
        self.expirations = {}

    def get(self, key):
        return self.values.get(key)

    def set(self, key, value, ex=None):
        self.values[key] = value
        self.expirations[key] = ex

    def delete(self, key):
        self.values.pop(key, None)

    def ping(self):
        return True


def test_redis_cache_namespaces_json_and_ttl(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(redis_module.Redis, "from_url", lambda *args, **kwargs: fake)
    cache = RedisCache(url="redis://unused", prefix="rag", timeout=1)

    cache.set("answer", {"text": "结果"}, ttl_seconds=30)

    assert fake.values["rag:answer"] == json.dumps({"text": "结果"}, ensure_ascii=False)
    assert fake.expirations["rag:answer"] == 30
    assert cache.get(":answer") == {"text": "结果"}
    assert cache.healthcheck() is True
    cache.delete("answer")
    assert cache.get("answer") is None
