from finance_rag.src.services.citation_validator import (
    apply_refusal_policy,
    extract_clause_references,
)


def test_extract_clause_references_returns_unique_references():
    assert extract_clause_references("依据第五条和第12条，同时再次引用第五条。") == ["第五条", "第12条"]


def test_refusal_policy_refuses_when_sources_are_empty():
    should_refuse, should_warn = apply_refusal_policy(None, 0)

    assert should_refuse is True
    assert should_warn is False


def test_refusal_policy_allows_relevant_sources_without_rerank_scores():
    should_refuse, should_warn = apply_refusal_policy(None, 2)

    assert should_refuse is False
    assert should_warn is False
