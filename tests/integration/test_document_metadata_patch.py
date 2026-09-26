"""研报元数据人工修正。

抽取链路的最后一道兜底：正则与模型都会抽错。若没有人工修正手段，
元数据维度就形同虚设——写进去一个非法行业名只会让过滤静默失效。
"""

import pytest
from fastapi import HTTPException

from finance_rag.src.api.routes import documents
from finance_rag.src.schemas.document import DocumentMetadataPatch
from finance_rag.src.services.document_service import DocumentManager


class _FakeKB:
    def __init__(self, *, missing: bool = False, has_report_fields: bool = True):
        self.calls: list[tuple[str, dict]] = []
        self._missing = missing
        self._has_report_fields = has_report_fields

    def update_document_metadata(self, source: str, metadata: dict) -> dict:
        if not self._has_report_fields:
            # 与 KnowledgeBase.update_document_metadata 的真实报错文案保持一致，
            # 这样断言的是「端点是否把可操作的提示透传出去」
            raise RuntimeError(
                "集合缺少研报元数据字段，请先执行 rebuild_collection() 重建集合"
            )
        self.calls.append((source, metadata))
        return {"source": source, "updated_count": 0 if self._missing else 3,
                "missing": self._missing}


def _manager(monkeypatch, kb: _FakeKB | None = None) -> DocumentManager:
    """注入假 KB：``DocumentManager.kb`` 是只读属性，这里直接构造时传入。"""
    manager = DocumentManager.__new__(DocumentManager)
    manager._kb = kb or _FakeKB()
    manager._collection_name = "finance_kb"
    return manager


def test_valid_metadata_is_marked_manual(monkeypatch):
    kb = _FakeKB()
    manager = _manager(monkeypatch, kb)

    result = manager.update_document_metadata(
        "研报.pdf", {"security_code": "600519", "industry_l1": "食品饮料",
                     "industry_l2": "白酒Ⅱ"}
    )

    _, sent = kb.calls[0]
    assert sent["security_code"] == "600519"
    assert sent["industry_l2"] == "白酒Ⅱ"
    assert sent["meta_source"] == "manual"
    assert sent["needs_review"] is False
    assert result["updated_count"] == 3


def test_invalid_industry_is_rejected_not_silently_blanked(monkeypatch):
    kb = _FakeKB()
    manager = _manager(monkeypatch, kb)

    with pytest.raises(ValueError) as excinfo:
        manager.update_document_metadata("研报.pdf", {"industry_l1": "白酒"})

    assert "industry_l1" in str(excinfo.value)
    assert kb.calls == []          # 非法值不得写库


def test_empty_string_clears_a_field(monkeypatch):
    kb = _FakeKB()
    manager = _manager(monkeypatch, kb)

    manager.update_document_metadata("研报.pdf", {"broker": ""})

    _, sent = kb.calls[0]
    assert sent["broker"] == ""


def test_untouched_fields_are_not_included(monkeypatch):
    kb = _FakeKB()
    manager = _manager(monkeypatch, kb)

    manager.update_document_metadata("研报.pdf", {"broker": "中信证券"})

    _, sent = kb.calls[0]
    assert "security_code" not in sent
    assert "industry_l1" not in sent


async def test_endpoint_returns_422_for_invalid_industry(monkeypatch):
    manager = _manager(monkeypatch)
    monkeypatch.setattr(documents, "get_document_manager", lambda kb="": manager)

    with pytest.raises(HTTPException) as excinfo:
        await documents.patch_document_metadata(
            "研报.pdf",
            DocumentMetadataPatch(industry_l1="白酒"),
            current_user="admin",
        )

    assert excinfo.value.status_code == 422


async def test_endpoint_returns_404_when_document_missing(monkeypatch):
    manager = _manager(monkeypatch, _FakeKB(missing=True))
    monkeypatch.setattr(documents, "get_document_manager", lambda kb="": manager)

    with pytest.raises(HTTPException) as excinfo:
        await documents.patch_document_metadata(
            "缺失.pdf",
            DocumentMetadataPatch(broker="中信证券"),
            current_user="admin",
        )

    assert excinfo.value.status_code == 404


async def test_endpoint_rejects_empty_payload(monkeypatch):
    manager = _manager(monkeypatch)
    monkeypatch.setattr(documents, "get_document_manager", lambda kb="": manager)

    with pytest.raises(HTTPException) as excinfo:
        await documents.patch_document_metadata(
            "研报.pdf", DocumentMetadataPatch(), current_user="admin"
        )

    assert excinfo.value.status_code == 400


async def test_endpoint_surfaces_stale_schema_as_500(monkeypatch):
    manager = _manager(monkeypatch, _FakeKB(has_report_fields=False))
    monkeypatch.setattr(documents, "get_document_manager", lambda kb="": manager)

    with pytest.raises(HTTPException) as excinfo:
        await documents.patch_document_metadata(
            "研报.pdf",
            DocumentMetadataPatch(broker="中信证券"),
            current_user="admin",
        )

    assert excinfo.value.status_code == 500
    assert "rebuild_collection" in str(excinfo.value.detail)
