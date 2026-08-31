import json

from finance_rag.src.api.streaming import encode_event


def test_streaming_adapter_emits_json_for_unicode_payload():
    encoded = encode_event({"type": "done", "sources": [{"title": "年报"}]})

    assert encoded["event"] == "done"
    assert "年报" in encoded["data"]
    assert json.loads(encoded["data"])["sources"][0]["title"] == "年报"
