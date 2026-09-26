from finance_rag.src.services.citation_validator import apply_refusal_policy


def test_refusal_policy_refuses_when_sources_are_empty():
    should_refuse, should_warn = apply_refusal_policy(None, 0)

    assert should_refuse is True
    assert should_warn is False


def test_refusal_policy_allows_relevant_sources_without_rerank_scores():
    should_refuse, should_warn = apply_refusal_policy(None, 2)

    assert should_refuse is False
    assert should_warn is False
