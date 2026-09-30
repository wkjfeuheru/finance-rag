"""断崖检测发生在粗召回之后、重排序之前。"""

from finance_rag.src.eval.retrieval import build_retrieval_strategy
from finance_rag.src.rag.retrieval.hybrid_retriever import BGEReranker, HybridRetriever


def _client(scores: list[float]):
    hits = [
        {
            "id": f"c{i}",
            "distance": score,
            "entity": {"id": f"c{i}", "content": "body", "parent_id": f"p{i}"},
        }
        for i, score in enumerate(scores)
    ]

    class Client:
        def search(self, **_kwargs):
            return [hits]

        def describe_collection(self, _name):
            return {"fields": []}

    return Client()


def test_coarse_cliff_drops_tail_before_rerank():
    seen: dict[str, list[str]] = {}

    class Reranker:
        def rerank(self, _query, candidates, top_n=3):
            seen["ids"] = [item["id"] for item in candidates]
            return candidates[:top_n]

    retriever = HybridRetriever(
        client=_client([1.0, 0.95, 0.2, 0.05]),
        collection_name="finance_kb",
        embed_query_fn=lambda _text: [0.1],
        reranker=Reranker(),
    )
    results = retriever.search(
        "q", k=20, use_dense_only=True, expand_parents=False,
        use_rerank=True, rerank_top_n=3,
    )

    assert seen["ids"] == ["c0", "c1", "c2"]
    assert [item["id"] for item in results] == ["c0", "c1", "c2"]
    assert retriever._last_cliff == {"pre": 4, "post": 3}


def test_rerank_does_not_apply_a_second_cliff():
    reranker = BGEReranker()

    class Model:
        def predict(self, pairs):
            assert len(pairs) == 4
            return [0.95, 0.90, 0.40, 0.38]

    reranker._get_model = lambda: Model()
    results = reranker.rerank(
        "q",
        [{"content": str(i), "score": 1 - i * 0.01} for i in range(4)],
        top_n=4,
    )

    assert len(results) == 4


def test_parent_expansion_keeps_coarse_pool_when_reranking():
    seen: dict[str, int] = {}

    class Reranker:
        def rerank(self, _query, candidates, top_n=3):
            seen["n"] = len(candidates)
            return candidates[:top_n]

    class Parents:
        def get_batch(self, parent_ids):
            return [
                {"id": pid, "content": "x" * 80, "heading": "", "heading_path": ""}
                for pid in parent_ids
            ]

    retriever = HybridRetriever(
        client=_client([1.0, 0.98, 0.96, 0.94]),
        collection_name="finance_kb",
        embed_query_fn=lambda _text: [0.1],
        reranker=Reranker(),
        parent_store=Parents(),
    )
    retriever.search(
        "q", k=1, use_dense_only=True, expand_parents=True,
        use_rerank=True, rerank_top_n=2,
    )

    assert seen["n"] == 4


def test_keyword_boost_keeps_coarse_pool_when_reranking():
    seen: dict[str, int] = {}

    class Reranker:
        def rerank(self, _query, candidates, top_n=3):
            seen["n"] = len(candidates)
            return candidates[:top_n]

    retriever = HybridRetriever(
        client=_client([1.0, 0.98, 0.96, 0.94]),
        collection_name="finance_kb",
        embed_query_fn=lambda _text: [0.1],
        reranker=Reranker(),
    )
    retriever.search(
        "q", k=1, use_dense_only=True, expand_parents=False,
        use_rerank=True, rerank_top_n=2, keywords=["missing-token"],
    )

    assert seen["n"] == 4


def test_l5_records_coarse_cliff_not_final_top_k():
    class Kb:
        last_retrieval_cliff = {"pre": 20, "post": 8}

        def hybrid_search(self, *_args, **_kwargs):
            return [{"id": str(i)} for i in range(5)]

    output = build_retrieval_strategy(Kb(), "L5").retrieve("q")

    assert output.cliff_triggered is True
    assert output.pre_cliff_depth == 20
    assert output.post_cliff_depth == 8
    assert len(output.results) == 5
