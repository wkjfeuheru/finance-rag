"""扫描版 PDF 检测器。

通过计算每页平均字符数来判断 PDF 是否为扫描版（图片型）。
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def is_scanned_pdf(file_path: str | Path, threshold: int = 50) -> bool:
    """检测 PDF 是否为扫描版（图片型）。

    使用 ``pypdf.PdfReader`` 提取每页文本，计算每页平均字符数。
    低于阈值则判定为扫描版；非 PDF 或异常时返回 ``False``。

    Parameters
    ----------
    file_path :
        PDF 文件路径。
    threshold :
        每页平均字符数阈值，默认 50。

    Returns
    -------
    bool
        ``True`` 表示扫描版 PDF，``False`` 表示文本版或非 PDF。
    """
    path = Path(file_path)
    if path.suffix.lower() != ".pdf":
        return False

    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        if not reader.pages:
            return False

        total_chars = 0
        for page in reader.pages:
            text = page.extract_text() or ""
            total_chars += len(text.strip())

        avg_chars = total_chars / len(reader.pages)
        return avg_chars < threshold
    except Exception as exc:
        logger.warning("检测 PDF 类型失败 %s: %s", path, exc)
        return False
