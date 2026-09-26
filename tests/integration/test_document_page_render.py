"""原文页渲染接口。

页码级溯源的价值全在「能跳」：给出页码 → 立刻看到那一页。这条链路直接读
对象存储里的原始 PDF 并按需渲染，不落临时文件。
"""

import pymupdf
import pytest
from fastapi import HTTPException

from finance_rag.src.api.routes import documents


def _pdf_bytes(pages: int = 2) -> bytes:
    document = pymupdf.open()
    for index in range(pages):
        page = document.new_page()
        page.insert_text((72, 72), f"page {index + 1}")
    payload = document.tobytes()
    document.close()
    return payload


class _Storage:
    def __init__(self, payload: bytes | None):
        self._payload = payload
        self.requested: list[str] = []

    async def download(self, key: str) -> bytes:
        self.requested.append(key)
        if self._payload is None:
            raise FileNotFoundError(key)
        return self._payload


def _use_storage(monkeypatch, payload: bytes | None) -> _Storage:
    storage = _Storage(payload)
    monkeypatch.setattr(documents, "get_storage", lambda: storage)
    return storage


async def test_render_page_returns_png(monkeypatch):
    storage = _use_storage(monkeypatch, _pdf_bytes(pages=2))

    response = await documents.render_document_page(
        "研报.pdf", 2, current_user="admin"
    )

    assert response.media_type == "image/png"
    assert response.body[:4] == b"\x89PNG"
    assert storage.requested == ["docs/研报.pdf"]


async def test_render_page_out_of_range_returns_404(monkeypatch):
    _use_storage(monkeypatch, _pdf_bytes(pages=2))

    with pytest.raises(HTTPException) as excinfo:
        await documents.render_document_page("研报.pdf", 99, current_user="admin")

    assert excinfo.value.status_code == 404


async def test_render_page_rejects_zero_and_negative(monkeypatch):
    _use_storage(monkeypatch, _pdf_bytes())

    for page in (0, -3):
        with pytest.raises(HTTPException) as excinfo:
            await documents.render_document_page("研报.pdf", page, current_user="admin")
        assert excinfo.value.status_code == 400


async def test_missing_original_returns_404(monkeypatch):
    _use_storage(monkeypatch, None)

    with pytest.raises(HTTPException) as excinfo:
        await documents.render_document_page("缺失.pdf", 1, current_user="admin")

    assert excinfo.value.status_code == 404


async def test_non_pdf_original_returns_400(monkeypatch):
    _use_storage(monkeypatch, "这不是 PDF".encode("utf-8"))

    with pytest.raises(HTTPException) as excinfo:
        await documents.render_document_page("笔记.md", 1, current_user="admin")

    assert excinfo.value.status_code == 400
