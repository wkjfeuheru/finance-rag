import json

from finance_rag.src.agent.state.models import AgentState
from finance_rag.src.api.streaming import encode_event


def test_agent_state_is_json_serializable():
    state: AgentState = {
        "query": "什么是净资本？",
        "history": [],
        "rewritten_query": "",
        "keywords": [],
        "sub_queries": [],
        "docs": [],
        "reflection_round": 0,
        "answer": "",
        "sources": [],
        "error": "",
    }

    assert json.loads(json.dumps(state, ensure_ascii=False))["query"] == "什么是净资本？"


def test_encode_event_preserves_type_and_payload():
    event = {"type": "token", "content": "答案"}

    encoded = encode_event(event)

    assert encoded["event"] == "token"
    assert json.loads(encoded["data"]) == event
