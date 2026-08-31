"""PDF 嵌入图片提取与多模态描述（caption）生成。

* 使用 PyMuPDF 提取 PDF 嵌入图片，按尺寸阈值过滤 + SHA256 去重；
* 通过 OpenAI 兼容端点（默认 DashScope Qwen-VL）生成中文图片描述；
* 未配置 API key 时仅保存图片、描述为占位提示；
* 网络/模型错误时降级为空描述，**不阻断解析**。

编排入口 :func:`process_document_images` 一次完成「提取 → 描述 → Markdown
段落」流程，供 ``Parser.parse`` 在 PDF 解析路径后追加调用。
"""

from __future__ import annotations

import base64
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - 类型提示
    import httpx

logger = logging.getLogger(__name__)


# ===========================================================================
# 数据模型
# ===========================================================================


@dataclass
class ExtractedImage:
    """从 PDF 中成功提取的单张图片。"""

    page: int  # 页码（1-based）
    sha256: str  # 内容 SHA256（十六进制）
    ext: str  # 扩展名（无点），如 "png"
    width: int
    height: int
    size_bytes: int
    data: bytes  # 原始字节（已统一转为 PNG）
    caption: str = ""  # 多模态描述文本；未生成时为空串


# ===========================================================================
# 配置读取（延迟导入，避免在 import 阶段强依赖 settings 完整加载）
# ===========================================================================


def _cfg():
    from finance_rag.src.core import config as settings

    return {
        "api_key": settings.DASHSCOPE_API_KEY,
        "base_url": settings.IMAGE_CAPTION_BASE_URL,
        "model": settings.IMAGE_CAPTION_MODEL,
        "timeout": settings.IMAGE_CAPTION_TIMEOUT_SECONDS,
        "max_images": settings.IMAGE_CAPTION_MAX_IMAGES,
        "min_dim_px": settings.IMAGE_CAPTION_MIN_DIM_PX,
        "min_bytes": settings.IMAGE_CAPTION_MIN_BYTES,
        "prompt": settings.IMAGE_CAPTION_PROMPT,
        "max_retries": settings.IMAGE_CAPTION_MAX_RETRIES,
    }


# ===========================================================================
# 工具函数
# ===========================================================================


_MIME_BY_EXT = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "gif": "image/gif",
    "webp": "image/webp",
    "bmp": "image/bmp",
}


def _source_slug(source: str) -> str:
    """把 source（可能含中文/特殊字符）映射为文件系统安全的 12 位 slug。"""
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:12]


def build_storage_key(source: str, sha256: str, ext: str) -> str:
    """生成存储后端的 key：``images/<source-slug>/<img-sha16>.<ext>``。

    路径仅含 ASCII（source 经 sha256），对 local / S3 后端都安全。
    """
    return f"images/{_source_slug(source)}/{sha256[:16]}.{ext}"


def is_captioner_available() -> bool:
    """是否已配置多模态 API key（决定是否调用远端模型生成描述）。"""
    return bool(_cfg()["api_key"])


# ===========================================================================
# 图片提取（PyMuPDF）
# ===========================================================================


def extract_images_from_pdf(
    path: Path,
    *,
    max_images: int | None = None,
    min_dim_px: int | None = None,
    min_bytes: int | None = None,
) -> list[ExtractedImage]:
    """从 PDF 提取嵌入图片，按尺寸阈值过滤并跨页 SHA256 去重。

    * 统一转 PNG 输出（alpha 通道剥离）；
    * 依赖 ``PyMuPDF``（``fitz``），未安装时返回空列表并记警告；
    * 仅提取尺寸与字节数都达阈值的图片，过滤掉图标/装饰位图。
    """
    cfg = _cfg()
    max_images = cfg["max_images"] if max_images is None else max_images
    min_dim_px = cfg["min_dim_px"] if min_dim_px is None else min_dim_px
    min_bytes = cfg["min_bytes"] if min_bytes is None else min_bytes

    if max_images <= 0:
        return []

    try:
        import pymupdf as fitz  # 1.24+ 推荐的新导入名
    except ImportError:  # pragma: no cover - 旧版本回退
        try:
            import fitz  # type: ignore[import-not-found]
        except ImportError:
            logger.warning(
                "PyMuPDF 未安装，跳过 PDF 嵌入图片提取"
                "（安装命令：pip install pymupdf>=1.24）"
            )
            return []

    images: list[ExtractedImage] = []
    seen_hashes: set[str] = set()

    try:
        doc = fitz.open(str(path))
    except Exception as exc:
        logger.warning("PyMuPDF 打开 PDF 失败：%s（跳过图片提取）", exc)
        return []

    try:
        for page_num in range(len(doc)):
            try:
                page = doc.load_page(page_num)
            except Exception:
                continue
            for img_info in page.get_images(full=True):
                xref = img_info[0]
                try:
                    pix = fitz.Pixmap(doc, xref)
                    if pix.alpha:
                        pix = fitz.Pixmap(fitz.csRGB, pix)  # 剥离 alpha → RGB
                    png_bytes = pix.tobytes("png")
                except Exception as exc:
                    logger.debug("PDF xref=%d 提取失败：%s", xref, exc)
                    continue

                width, height = pix.width, pix.height
                size = len(png_bytes)
                if width < min_dim_px or height < min_dim_px or size < min_bytes:
                    continue

                sha = hashlib.sha256(png_bytes).hexdigest()
                if sha in seen_hashes:
                    continue
                seen_hashes.add(sha)

                images.append(
                    ExtractedImage(
                        page=page_num + 1,
                        sha256=sha,
                        ext="png",
                        width=width,
                        height=height,
                        size_bytes=size,
                        data=png_bytes,
                    )
                )
                if len(images) >= max_images:
                    break
            if len(images) >= max_images:
                break
    finally:
        doc.close()

    return images


# ===========================================================================
# 多模态描述（OpenAI 兼容端点）
# ===========================================================================


def _retryable_http_error(exc: Exception) -> bool:
    """判断 httpx 异常是否值得重试（超时/连接/5xx/429）。"""
    import httpx

    if isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 500, 502, 503, 504)
    return False


def caption_image(
    data: bytes,
    ext: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    timeout: float | None = None,
    prompt: str | None = None,
    max_retries: int | None = None,
    _transport: "httpx.BaseTransport | None" = None,
) -> str:
    """调用多模态端点生成图片中文描述。

    * 未配置 API key → 返回空串（降级，不抛错）；
    * 网络/HTTP/解析异常 → 记录警告并返回空串（不阻断解析）；
    * 端点默认 DashScope Qwen-VL，可换任意 OpenAI 兼容视觉端点。

    内部使用 ``httpx.Client``，测试可通过 ``_transport`` 注入 ``MockTransport``。
    """
    cfg = _cfg()
    api_key = cfg["api_key"] if api_key is None else api_key
    base_url = cfg["base_url"] if base_url is None else base_url
    model = cfg["model"] if model is None else model
    timeout = cfg["timeout"] if timeout is None else timeout
    prompt = cfg["prompt"] if prompt is None else prompt
    max_retries = cfg["max_retries"] if max_retries is None else max_retries
    max_retries = max(0, int(max_retries))

    if not api_key:
        return ""

    mime = _MIME_BY_EXT.get(ext.lower(), "image/png")
    b64 = base64.b64encode(data).decode("ascii")
    data_url = f"data:{mime};base64,{b64}"

    body = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        "max_tokens": 600,
        "temperature": 0.1,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        import httpx  # noqa: WPS433 - 局部导入避免测试/无 key 时强依赖
    except ImportError:
        logger.warning("httpx 未安装，无法生成图片描述")
        return ""

    client_kwargs: dict = {"headers": headers, "timeout": timeout}
    if _transport is not None:
        client_kwargs["transport"] = _transport

    payload = None
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            with httpx.Client(**client_kwargs) as client:
                resp = client.post(
                    f"{base_url.rstrip('/')}/chat/completions",
                    json=body,
                )
                resp.raise_for_status()
                payload = resp.json()
            break
        except Exception as exc:
            last_exc = exc
            if attempt >= max_retries or not _retryable_http_error(exc):
                logger.warning("图片描述请求失败（%s）：%s", model, exc)
                return ""
            logger.warning(
                "图片描述请求瞬时失败，重试 %d/%d（%s）：%s",
                attempt + 1, max_retries, model, exc,
            )

    if payload is None:
        logger.warning("图片描述请求失败（%s）：%s", model, last_exc)
        return ""

    try:
        return (payload["choices"][0]["message"]["content"] or "").strip()
    except (KeyError, IndexError, TypeError):
        logger.warning("图片描述响应解析失败：%s", model)
        return ""


# ===========================================================================
# Markdown 附录与编排
# ===========================================================================


def _placeholder_caption(page: int) -> str:
    """未配置视觉模型时按页码生成的占位描述（每张图唯一，避免被段落去重抹掉）。"""
    return (
        f"【图片内容描述：第 {page} 页图片待视觉模型生成描述"
        f"（请配置 DASHSCOPE_API_KEY 后重新索引）】"
    )


def build_caption_section(source: str, images: list[ExtractedImage]) -> str:
    """将图片列表渲染为 Markdown 附录段落（``## 附图说明`` 起）。

    每张图片生成：

    * ``### 图片 N（第 M 页）`` 标题；
    * ``![图](<storage-key>)`` 图片引用（与 ``build_storage_key`` 一致）；
    * 描述正文（未生成时按页码生成唯一占位提示）。
    """
    if not images:
        return ""
    lines: list[str] = ["", "## 附图说明", ""]
    for idx, img in enumerate(images, start=1):
        key = build_storage_key(source, img.sha256, img.ext)
        lines.append(f"### 图片 {idx}（第 {img.page} 页）")
        lines.append("")
        lines.append(f"![图]({key})")
        lines.append("")
        lines.append(img.caption or _placeholder_caption(img.page))
        lines.append("")
    return "\n".join(lines)


def process_document_images(
    path: Path, source: str
) -> tuple[str, list[ExtractedImage]]:
    """完整流程：提取 → 描述 → 渲染 Markdown。

    返回 ``(caption_markdown_section, extracted_images)``。

    * PyMuPDF 不可用 / 提取无图 → ``("", [])``；
    * 未配置 API key 时仅保存图片，描述用占位文本；
    * 任何阶段异常都会被捕获并返回 ``("", [])``，确保不阻断上层解析。
    """
    try:
        images = extract_images_from_pdf(path)
    except Exception as exc:
        logger.warning("PDF 图片提取异常：%s（跳过）", exc)
        return "", []

    if not images:
        return "", []

    if is_captioner_available():
        for img in images:
            try:
                img.caption = caption_image(img.data, img.ext)
            except Exception as exc:  # pragma: no cover - 防御
                logger.warning("图片描述异常（page=%d）：%s", img.page, exc)
    else:
        logger.info("DASHSCOPE_API_KEY 未配置，跳过图片描述生成：%s", source)

    section = build_caption_section(source, images)
    return section, images


# ===========================================================================
# 临时文件持久化（供 Parser 在子进程/主进程间搬运图片字节）
# ===========================================================================


def persist_images_to_tempdir(
    images: list[ExtractedImage],
    *,
    source: str,
    parent_dir: Path,
) -> tuple[list[dict], Path | None]:
    """把图片字节写到 ``parent_dir/.<uuid>-imgcache/``，返回元数据列表。

    元数据形如：
    ``{"temp_path", "temp_dir", "key", "page", "ext", "sha256", "caption"}``

    图片列表为空时返回 ``([], None)``。
    """
    if not images:
        return [], None

    import uuid

    temp_dir = parent_dir / f".{uuid.uuid4().hex}-imgcache"
    temp_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    for i, img in enumerate(images):
        temp_path = temp_dir / f"{i:03d}-{img.sha256[:12]}.{img.ext}"
        try:
            temp_path.write_bytes(img.data)
        except Exception as exc:
            logger.warning("图片临时落盘失败 %s: %s", temp_path, exc)
            continue
        records.append(
            {
                "temp_path": str(temp_path),
                "temp_dir": str(temp_dir),
                "key": build_storage_key(source, img.sha256, img.ext),
                "page": img.page,
                "ext": img.ext,
                "sha256": img.sha256,
                "caption": img.caption,
            }
        )
    return records, temp_dir
