"""入库增强（元数据 + 图片块 + 整表）的回归测试。

这一层的核心风险是**顺序**：图片描述会新增子块，而稠密向量是按子块顺序
严格一一对应的（写阶段 ``zip(..., strict=True)``）。一旦把增强挪到嵌入之后，
向量与块就会错位，且不会报错——只会让检索结果悄悄变差。
"""

import pytest
from langchain_core.documents import Document

from finance_rag.src.rag.ingestion import image_captioner, metadata_extractor
from finance_rag.src.rag.ingestion.chunker import DoclingChunks
from finance_rag.src.rag.ingestion.pdf_assets import ContentBlock, ImageAsset
from finance_rag.src.services.ingestion_pipeline import enrich_chunks


def _chunks(**kwargs) -> DoclingChunks:
    defaults = {
        "chunks": [Document(page_content="燃机订单超预期。", metadata={"id": "c1"})],
        "markdown": "## 投资要点\n\n燃机订单超预期。\n",
        "blocks": [ContentBlock(text="## 投资要点 燃机订单超预期。", page=3)],
    }
    defaults.update(kwargs)
    return DoclingChunks(**defaults)


async def test_enrich_merges_metadata_into_document(monkeypatch):
    monkeypatch.setattr(
        metadata_extractor,
        "_llm_extract",
        lambda **_: {"industry_l1": "食品饮料", "industry_l2": "白酒Ⅱ",
                     "report_type": "个股"},
    )

    chunks, metadata = await enrich_chunks(
        source="贵州茅台(600519)2026中报点评-中信证券.pdf",
        title="贵州茅台2026中报点评",
        markdown="## 投资要点\n\n正文",
        chunks=_chunks(),
        metadata={"category": "investment_research"},
        enable_llm=True,
    )

    assert metadata["category"] == "investment_research"   # 原有字段保留
    assert metadata["security_code"] == "600519"
    assert metadata["industry_l2"] == "白酒Ⅱ"
    assert metadata["meta_source"] == "llm"


async def test_enrich_honours_config_switch_when_llm_not_specified(monkeypatch):
    """默认走 config 开关：关掉时不得发起模型调用（离线/批量重跑要靠它）。"""
    from finance_rag.src.core import config

    calls: list[int] = []
    monkeypatch.setattr(config, "ENABLE_METADATA_LLM", False)
    monkeypatch.setattr(
        metadata_extractor, "_llm_extract", lambda **_: calls.append(1) or {}
    )

    _, metadata = await enrich_chunks(
        source="贵州茅台(600519)点评.pdf",
        title="",
        markdown="正文",
        chunks=_chunks(),
        metadata={},
    )

    assert calls == []
    assert metadata["security_code"] == "600519"   # 正则结果仍然可用
    assert metadata["meta_source"] == "regex"


async def test_enrich_appends_image_chunks_after_text_chunks(monkeypatch):
    monkeypatch.setattr(image_captioner, "caption_image", lambda *a, **k: "燃机订单结构图")
    chunks = _chunks(
        images=[ImageAsset(key_hint="4-0", page=4, data=b"\x89PNG", ext="png")]
    )

    enriched, _ = await enrich_chunks(
        source="研报.pdf",
        title="研报",
        markdown="正文",
        chunks=chunks,
        metadata={},
        enable_llm=False,
    )

    assert len(enriched.chunks) == 2
    image_chunk = enriched.chunks[-1]
    assert image_chunk.page_content == "燃机订单结构图"
    assert image_chunk.metadata["block_type"] == "image"
    assert image_chunk.metadata["start_page"] == 4
    assert image_chunk.metadata["image_key"] == "images/研报/4-0.png"
    assert image_chunk.metadata["chunk"] == 1          # 序号接在文本块之后


async def test_enrich_keeps_image_chunk_when_caption_fails(monkeypatch):
    """描述失败也要留块：图还在，分析师仍能靠页码跳转看到原图。"""
    monkeypatch.setattr(image_captioner, "caption_image", lambda *a, **k: "")
    chunks = _chunks(
        images=[ImageAsset(key_hint="7-0", page=7, data=b"\x89PNG", ext="png")]
    )

    enriched, _ = await enrich_chunks(
        source="研报.pdf", title="研报", markdown="正文", chunks=chunks,
        metadata={}, enable_llm=False,
    )

    placeholder = enriched.chunks[-1]
    assert placeholder.metadata["block_type"] == "image"
    assert "第 7 页" in placeholder.page_content


async def test_enrich_without_images_does_not_add_chunks(monkeypatch):
    enriched, _ = await enrich_chunks(
        source="研报.pdf", title="研报", markdown="正文", chunks=_chunks(),
        metadata={}, enable_llm=False,
    )

    assert len(enriched.chunks) == 1


async def test_image_chunk_ids_are_stable_across_reingest(monkeypatch):
    """重复入库必须覆盖同一块，否则每次重传都会新增一批图片块。"""
    monkeypatch.setattr(image_captioner, "caption_image", lambda *a, **k: "图")
    assets = [ImageAsset(key_hint="4-0", page=4, data=b"\x89PNG", ext="png")]

    first, _ = await enrich_chunks(
        source="研报.pdf", title="研报", markdown="正文",
        chunks=_chunks(images=list(assets)), metadata={}, enable_llm=False,
    )
    second, _ = await enrich_chunks(
        source="研报.pdf", title="研报", markdown="正文",
        chunks=_chunks(images=list(assets)), metadata={}, enable_llm=False,
    )

    assert first.chunks[-1].metadata["id"] == second.chunks[-1].metadata["id"]


async def test_metadata_extraction_failure_does_not_break_ingest(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("模型挂了")

    monkeypatch.setattr(metadata_extractor, "extract_metadata", _boom)

    with pytest.raises(RuntimeError):
        # 当前约定：抽取异常向上抛，由流水线按「单文件失败隔离」处理。
        # 降级为「静默吞掉」会让整篇文档带着空元数据入库而不被发现。
        await enrich_chunks(
            source="研报.pdf", title="研报", markdown="正文", chunks=_chunks(),
            metadata={}, enable_llm=True,
        )
