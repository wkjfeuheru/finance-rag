"""合规模块删除后的契约。

删除的目的是「不再维护」，不是「仓库里没有这个词」——因此这里同时锁住
**该没的没了** 与 **该留的还在**，避免清理时误伤通用能力。
"""

import importlib

import pytest


@pytest.mark.parametrize(
    "module",
    [
        "finance_rag.src.api.routes.compliance",
        "finance_rag.src.services.compliance_service",
        "finance_rag.src.services.compliance_rules",
        "finance_rag.src.schemas.compliance",
    ],
)
def test_compliance_modules_are_gone(module):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module)


def test_generic_citation_validation_is_kept():
    """研报同样需要通用引用校验，不能随合规模块一起删掉。"""
    from finance_rag.src.services.citation_validator import (
        CitationValidator,
        apply_refusal_policy,
        run_citation_validation,
    )

    assert hasattr(CitationValidator, "validate")
    assert callable(run_citation_validation)
    assert callable(apply_refusal_policy)


def test_compliance_specific_validators_are_gone():
    from finance_rag.src.services import citation_validator

    assert not hasattr(citation_validator, "validate_clause_citations")
    assert not hasattr(citation_validator, "validate_finding_evidence")
    assert not hasattr(citation_validator, "extract_clause_references")


def test_compliance_category_value_is_kept():
    """分类值保留：删了要动已 seed 的 PG 记录，收益为零。"""
    from finance_rag.src.rag.models.document_category import DOCUMENT_CATEGORIES

    assert "compliance_risk" in DOCUMENT_CATEGORIES


def test_compliance_config_keys_are_gone():
    from finance_rag.src.core import config

    for key in (
        "COMPLIANCE_CATEGORIES",
        "COMPLIANCE_REQUIRE_CLAUSE",
        "COMPLIANCE_CLAUSE_MIN_SCORE",
        "ENABLE_METADATA_FILTER",
    ):
        assert not hasattr(config, key), key


def test_iter_active_kbs_is_explicit_single_collection(monkeypatch):
    """注册表是逻辑分类视图，不是物理集合；这里恒为默认集合一项。"""
    from finance_rag.src.core.config import KB_COLLECTION_NAME
    from finance_rag.src.services import chat_service

    calls: list[tuple] = []

    def _fake_get_kb(*args, **kwargs):
        calls.append((args, kwargs))
        return "kb-default"

    monkeypatch.setattr(chat_service, "get_knowledge_base", _fake_get_kb)

    kbs = chat_service.iter_active_kbs()

    assert kbs == [(KB_COLLECTION_NAME, "kb-default")]
    # 不传参 = 取默认物理集合；传类别名会去建一个不存在的集合
    assert calls == [((), {})]


def test_seed_data_only_seeds_declared_categories():
    from scripts.seed_data import load_seed_data
    from finance_rag.src.rag.models.document_category import DOCUMENT_CATEGORIES

    seeded = {item["name"] for item in load_seed_data()["knowledge_bases"]}

    assert seeded == set(DOCUMENT_CATEGORIES)


def test_app_route_table_has_report_routes_and_no_compliance():
    """路由表级验证：删干净了，且研报新增的两个接口确实挂上了。

    用 ``openapi()["paths"]`` 而不是遍历 ``app.routes``：新版 FastAPI 会把
    ``include_router`` 的结果包成 ``_IncludedRouter``（没有 ``.path``），
    直接遍历会漏掉全部业务路由。
    """
    from finance_rag.src.main import app

    paths = set(app.openapi()["paths"])

    assert "/api/documents/{source}/page/{page}" in paths
    assert "/api/documents/{source}/metadata" in paths
    assert "/api/chat/stream" in paths
    assert not [path for path in paths if "compliance" in path]
