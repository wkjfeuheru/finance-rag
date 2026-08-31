from finance_rag.src.agent.tools.retrieval import AgentRetrievalTool, RetrievalTool


class FakeRetriever:
    def __init__(self):
        self.calls = []

    def search(self, query, **kwargs):
        self.calls.append((query, kwargs))
        return [{
            "chunk_id": "chunk-1",
            "content": "金融监管规则",
            "score": "0.85",
            "source": "rules.pdf",
            "category": "policy",
        }]


def test_empty_query_does_not_call_retriever():
    retriever = FakeRetriever()

    result = RetrievalTool(retriever).invoke("  ")

    assert result == {"query": "  ", "evidence": [], "metadata": {"count": 0}}
    assert retriever.calls == []


def test_retrieval_tool_normalizes_evidence_and_forwards_options():
    retriever = FakeRetriever()

    result = RetrievalTool(retriever).invoke(
        "  监管规则  ", k=2, filters={"category": "policy"}, use_rerank=True
    )

    assert result["query"] == "  监管规则  "
    assert result["metadata"] == {"count": 1, "filters": {"category": "policy"}}
    assert result["evidence"] == [{
        "id": "chunk-1",
        "content": "金融监管规则",
        "score": 0.85,
        "source": "rules.pdf",
        "title": "",
        "chunk": None,
        "parent_id": "",
        "category": "policy",
        "date": "",
        "heading": "",
        "rerank_score": None,
    }]
    assert retriever.calls[0][0] == "监管规则"
    assert retriever.calls[0][1]["k"] == 2
    assert retriever.calls[0][1]["use_rerank"] is True


def test_agent_tool_exposes_stable_name_and_description():
    assert AgentRetrievalTool.name == "retrieve_finance_evidence"
    assert AgentRetrievalTool.description
