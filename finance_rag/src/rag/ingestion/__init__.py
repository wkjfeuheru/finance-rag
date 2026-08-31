"""文档解析模块。提供统一的 get_parser() 入口。"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from finance_rag.src.core.config import LLAMA_CLOUD_API_KEY, SCAN_PDF_TEXT_THRESHOLD

logger = logging.getLogger(__name__)


@dataclass
class ParseResult:
    """解析结果。

    Attributes:
        markdown: 解析后的 Markdown 文本（扫描版 PDF 由 LlamaParse 产出；
                  普通 PDF 由 SimpleDirectoryReader 产出并可能追加「附图说明」）。
        images: 提取的嵌入图片元数据列表（仅普通 PDF 非空）。
                每条记录形如 ``{"temp_path", "temp_dir", "key", "page",
                "ext", "sha256", "caption"}``，图片字节位于 ``temp_path``，
                主进程负责将其上传到存储后端后清理 ``temp_dir``。
    """

    markdown: str
    images: list[dict] = field(default_factory=list)


class Parser:
    """统一文档解析器，根据文件扩展名选择解析策略。"""

    def parse(self, path: str | Path, *, source: str = "", title: str = "") -> ParseResult:
        """解析文件并返回 Markdown 文本（PDF 同时提取嵌入图片）。

        扫描版 PDF（每页平均字符数低于阈值）且配置了 LlamaParse API key 时，
        走 LlamaParse 云端 OCR；其余文件类型统一走 SimpleDirectoryReader。

        普通 PDF 解析完成后，会额外提取嵌入图片并追加一段 ``## 附图说明``
        Markdown；多模态 API key 缺失时仍保存图片，描述为占位提示。
        """
        path = Path(path)
        ext = path.suffix.lower()
        source = source or path.name
        images: list[dict] = []

        # 扫描版 PDF → LlamaParse 云端 OCR
        if ext == ".pdf" and LLAMA_CLOUD_API_KEY:
            from .scan_detector import is_scanned_pdf

            if is_scanned_pdf(path, SCAN_PDF_TEXT_THRESHOLD):
                from .directory_parser import parse_with_llama_parse

                markdown = parse_with_llama_parse(str(path), LLAMA_CLOUD_API_KEY)
                return ParseResult(markdown=markdown, images=[])

        # 其余文件类型统一使用 LlamaIndex SimpleDirectoryReader
        from .directory_parser import parse_with_simple_reader

        try:
            markdown = parse_with_simple_reader(str(path))
        except Exception:
            # md/txt 回退到直接读文件，保证简单文本稳定解析
            if ext in (".md", ".txt", ".markdown"):
                markdown = path.read_text(encoding="utf-8")
            else:
                raise

        # 普通 PDF 提取嵌入图片 + 多模态描述 → 追加「附图说明」段落
        if ext == ".pdf":
            markdown, images = self._enrich_with_images(path, source, markdown)

        return ParseResult(markdown=markdown, images=images)

    # ------------------------------------------------------------------
    # 内部：普通 PDF 图片提取 + 描述生成
    # ------------------------------------------------------------------

    @staticmethod
    def _enrich_with_images(
        path: Path, source: str, markdown: str
    ) -> tuple[str, list[dict]]:
        """提取 PDF 嵌入图片，生成描述，落盘到临时目录，追加到 Markdown。"""
        try:
            from .image_captioner import (
                persist_images_to_tempdir,
                process_document_images,
            )
        except ImportError:  # pragma: no cover - 防御性
            return markdown, []

        try:
            section, extracted = process_document_images(path, source=source)
            if not extracted:
                return markdown, []

            records, _temp_dir = persist_images_to_tempdir(
                extracted, source=source, parent_dir=path.parent
            )
            if not records:
                return markdown, []
        except Exception as exc:
            # 提取/描述/临时落盘任何失败都不阻断解析：仅保留原始文本内容入库
            logger.warning("PDF 图片处理失败：%s（忽略，原内容入库）", exc)
            return markdown, []

        return markdown + section, records


def get_parser() -> Parser:
    """获取解析器单例。"""
    return Parser()
