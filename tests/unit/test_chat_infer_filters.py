"""`infer_filters` 开关的契约。

前端「清除过滤」按钮靠它生效：只清本地状态是不够的，必须让后端**不要**
再自动推断回来，否则分析师点了清除、下一次提问过滤条件又出现了。
"""

from finance_rag.src.api.routes import chat as chat_route
from finance_rag.src.schemas.chat import ChatRequest


def test_infer_filters_defaults_to_true():
    assert ChatRequest(query="贵州茅台怎么样").infer_filters is True


def test_infer_filters_can_be_disabled():
    assert ChatRequest(query="贵州茅台怎么样", infer_filters=False).infer_filters is False


async def test_stream_route_forwards_infer_filters(monkeypatch):
    captured: dict = {}

    async def _fake_stream(query, history, **kwargs):
        captured.update({"query": query, **kwargs})
        if False:  # pragma: no cover - 仅用于把函数标记为异步生成器
            yield {}

    monkeypatch.setattr(chat_route, "_chat_fns", lambda: (_fake_stream, None))

    response = await chat_route.chat_stream(
        ChatRequest(query="对比三家券商预测", infer_filters=False),
        current_user="admin",
    )
    # 消费事件流以触发调用
    async for _ in response.body_iterator:
        break

    assert captured["infer_filters"] is False
    assert captured["query"] == "对比三家券商预测"


async def test_non_stream_route_forwards_infer_filters(monkeypatch):
    captured: dict = {}

    async def _fake_chat(query, history, **kwargs):
        captured.update({"query": query, **kwargs})
        return {
            "answer": "ok",
            "sources": [],
            "rewritten_query": query,
        }

    monkeypatch.setattr(chat_route, "_chat_fns", lambda: (None, _fake_chat))

    await chat_route.chat(
        ChatRequest(query="行业景气度", infer_filters=False), current_user="admin"
    )

    assert captured["infer_filters"] is False
