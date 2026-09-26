"""解析层页码与图片资产抽取的回归测试。

背景：`ParseResult` 曾缺少 `images` 字段，而 `chunker.parse_and_chunk` 传入了
`result.images`，导致**每一次上传都在解析阶段抛 AttributeError**。这里的用例
同时锁住「字段存在」与「字段有真实来源」两件事。

工作目录沿用 `test_ingestion_pipeline._workdir` 的约定（`tests/unit/.tmp`）：
受限环境下系统临时目录不可写，且 pytest 的 `tmp_path` 在销毁刚建目录时会被拒。
"""

from pathlib import Path

import pymupdf

from finance_rag.src.rag.ingestion import pdf_assets


def _workdir(name: str) -> Path:
    """仓库内的工作目录（受限环境下系统临时目录可能不可写）。"""
    path = Path(__file__).resolve().parent / ".tmp" / "parse_assets" / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_pdf(path: Path, *, image_pages=(), text_pages=2) -> Path:
    """生成一个最小 PDF：每页一段文字，可选在指定页插入同一张内嵌图片。"""
    doc = pymupdf.open()
    for index in range(text_pages):
        page = doc.new_page()
        page.insert_text((72, 72), f"page {index + 1} text")
    pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 40, 40))
    pixmap.clear_with(255)
    for page_index in image_pages:
        doc[page_index].insert_image(pymupdf.Rect(100, 100, 140, 140), pixmap=pixmap)
    doc.save(path)
    doc.close()
    return path


def test_extract_images_carries_one_based_page_and_hint():
    path = _write_pdf(_workdir("with-image") / "sample.pdf", image_pages=(1,))

    assets = pdf_assets.extract_images(path)

    assert len(assets) == 1
    assert assets[0].page == 2
    assert assets[0].key_hint == "2-0"
    assert assets[0].ext == "png"
    assert assets[0].data[:4] == b"\x89PNG"


def test_extract_images_dedupes_identical_bytes():
    """同一张图重复出现在多页（研报页眉 logo）时只保留一份，避免重复付视觉模型费用。"""
    path = _write_pdf(_workdir("repeated-image") / "repeated.pdf", image_pages=(0, 1))

    assets = pdf_assets.extract_images(path)

    assert len(assets) == 1
    assert assets[0].page == 1


def test_extract_blocks_maps_text_to_one_based_page():
    path = _write_pdf(_workdir("blocks") / "sample.pdf", image_pages=(1,))

    blocks = pdf_assets.extract_blocks(path)

    assert [block.page for block in blocks] == [1, 2]
    assert "page 2 text" in blocks[1].text


def test_extract_assets_tolerate_broken_pdf():
    broken = _workdir("broken") / "broken.pdf"
    broken.write_bytes(b"not a pdf at all")

    assert pdf_assets.extract_images(broken) == ()
    assert pdf_assets.extract_blocks(broken) == ()


def test_parse_result_defaults_to_empty_assets():
    from finance_rag.src.rag.ingestion.mineru_parser import ParseResult

    result = ParseResult(markdown="x")

    assert result.images == ()
    assert result.blocks == ()


def test_text_shortcut_parses_without_assets():
    from finance_rag.src.rag.ingestion.mineru_parser import parse_with_mineru

    note = _workdir("text-shortcut") / "note.md"
    note.write_text("## 投资要点\n\n燃机订单超预期。", encoding="utf-8")

    result = parse_with_mineru(note, source="note.md")

    assert "燃机订单" in result.markdown
    assert result.images == ()
    assert result.blocks == ()


def test_content_list_blocks_map_zero_based_page_idx():
    from finance_rag.src.rag.ingestion.mineru_parser import _blocks_from_content_list

    payload = [
        {"type": "text", "text": "第一页正文", "page_idx": 0},
        {"type": "image", "img_caption": ["产能结构图"], "page_idx": 1},
        {"type": "text", "text": "   ", "page_idx": 2},
        {"type": "table", "table_body": "| 年份 | 营收 |", "page_idx": 2},
    ]

    blocks = _blocks_from_content_list(payload)

    assert [(block.text, block.page) for block in blocks] == [
        ("第一页正文", 1),
        ("产能结构图", 2),
        ("| 年份 | 营收 |", 3),
    ]


def test_content_list_blocks_tolerate_unexpected_payload():
    from finance_rag.src.rag.ingestion.mineru_parser import _blocks_from_content_list

    assert _blocks_from_content_list(None) == ()
    assert _blocks_from_content_list({"not": "a list"}) == ()
    assert _blocks_from_content_list([{"no_page_idx": True}]) == ()


def test_image_object_key_is_source_scoped_and_carries_page():
    asset = pdf_assets.ImageAsset(key_hint="2-0", page=2, data=b"x", ext="png")

    key = pdf_assets.image_object_key("贵州茅台(600519)2026中报点评.pdf", asset)

    assert key == "images/贵州茅台(600519)2026中报点评/2-0.png"


async def test_upload_extracted_images_uses_in_memory_bytes():
    """图片已不再落临时文件，上传必须直接消费 bytes（旧契约会 KeyError）。"""
    from types import SimpleNamespace

    from finance_rag.src.services.document_service import DocumentManager

    uploaded: list[tuple[str, bytes]] = []

    class _Storage:
        async def upload(self, key, data):
            uploaded.append((key, data))

    parsed = SimpleNamespace(
        source="研报.pdf",
        chunks=SimpleNamespace(
            images=[pdf_assets.ImageAsset(key_hint="1-0", page=1, data=b"\x89PNG", ext="png")]
        ),
    )

    await DocumentManager._upload_extracted_images(parsed, _Storage())

    assert uploaded == [("images/研报/1-0.png", b"\x89PNG")]
