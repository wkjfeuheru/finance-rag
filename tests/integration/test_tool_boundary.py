from finance_rag.src.agent.tools.retrieval import RetrievalTool


class CallableRetriever:
    def __call__(self, query, **kwargs):
        return [{"id": "1", "content": query, "score": 1}]


def test_agent_tool_accepts_callable_retriever_without_provider_import():
    result = RetrievalTool(CallableRetriever()).invoke("资产负债表", k=1)

    assert result["evidence"][0]["id"] == "1"
    assert result["evidence"][0]["content"] == "资产负债表"
