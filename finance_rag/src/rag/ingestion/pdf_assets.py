"""PDF 图片与分页文本抽取（pymupdf）。

解析层需要两样 Markdown 拿不到的信息：

- **嵌入图片本体及其所在页**：用于视觉描述并作为独立索引块入库；
- **每页文本块**：用于把切块结果归属到页码（页码级溯源的前提）。

两个函数都对损坏 / 加密 / 非 PDF 输入保持宽容：只告警并返回空结果，
由调用方决定是否阻断（当前约定是不阻断入库）。
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ImageAsset:
    """一张内嵌图片及其所在页（页码 1-based）。"""

    key_hint: str
    page: int
    data: bytes
    ext: str


@dataclass(frozen=True)
class ContentBlock:
    """页面上的一段文本及其所在页（页码 1-based）。"""

    text: str
    page: int


def _open_pdf(path: str | Path):
    """延迟导入 pymupdf：``.md`` / ``.txt`` 短路路径不应付出这份导入开销。"""
    import pymupdf

    return pymupdf.open(str(path))


def extract_images(path: str | Path) -> tuple[ImageAsset, ...]:
    """抽取全部内嵌图片，按图片内容去重。

    同一张图重复出现在多页（研报页眉 logo、共享水印）时只保留首次出现，
    避免为同一张图重复调用视觉模型。``key_hint`` 形如 ``"{页}-{页内序号}"``。
    """
    try:
        doc = _open_pdf(path)
    except Exception as exc:  # noqa: BLE001 - 任何打开失败都只降级
        logger.warning("打开 PDF 失败，跳过图片抽取（%s）：%s", path, exc)
        return ()

    assets: list[ImageAsset] = []
    seen: set[str] = set()
    try:
        for page_index in range(doc.page_count):
            try:
                images = doc[page_index].get_images(full=True)
            except Exception as exc:  # noqa: BLE001 - 单页失败不影响其它页
                logger.warning("读取第 %s 页图片清单失败（%s）：%s", page_index + 1, path, exc)
                continue
            for seq, image in enumerate(images):
                xref = image[0]
                try:
                    info = doc.extract_image(xref)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("提取图片失败（%s xref=%s）：%s", path, xref, exc)
                    continue
                data = info.get("image") or b""
                if not data:
                    continue
                digest = hashlib.md5(data).hexdigest()
                if digest in seen:
                    continue
                seen.add(digest)
                assets.append(
                    ImageAsset(
                        key_hint=f"{page_index + 1}-{seq}",
                        page=page_index + 1,
                        data=data,
                        ext=str(info.get("ext") or "png").lower(),
                    )
                )
    finally:
        doc.close()
    return tuple(assets)


def extract_blocks(path: str | Path) -> tuple[ContentBlock, ...]:
    """按页抽取文本块，作为 MinerU ``content_list`` 不可用时的降级来源。"""
    try:
        doc = _open_pdf(path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("打开 PDF 失败，跳过文本块抽取（%s）：%s", path, exc)
        return ()

    blocks: list[ContentBlock] = []
    try:
        for page_index in range(doc.page_count):
            try:
                raw_blocks = doc[page_index].get_text("blocks")
            except Exception as exc:  # noqa: BLE001
                logger.warning("读取第 %s 页文本块失败（%s）：%s", page_index + 1, path, exc)
                continue
            for block in raw_blocks:
                if len(block) < 5:
                    continue
                text = str(block[4] or "").strip()
                if text:
                    blocks.append(ContentBlock(text=text, page=page_index + 1))
    finally:
        doc.close()
    return tuple(blocks)


def image_object_key(source: str, asset: ImageAsset) -> str:
    """图片在对象存储中的 key。

    与文档 ``source`` 绑定（同文档重复入库覆盖同一张图），key 里带页号，
    便于人工排查时直接对回原文页码。
    """
    return f"images/{Path(source).stem}/{asset.key_hint}.{asset.ext}"


__all__ = [
    "ContentBlock",
    "ImageAsset",
    "extract_blocks",
    "extract_images",
    "image_object_key",
]
