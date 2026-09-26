"""研报嵌套图片的中文描述（视觉模型）。

**这是本项目里数据外发范围最大的一条链路**：入库时会把每一张图原样发往
外部视觉模型（默认 DashScope ``qwen-vl-max``）。因此：

- 过小/过短的图片（印章、分隔线、页眉 logo）在本地就被拦下，不产生外发；
- 每次真正外发都写一条 ``image_egress`` 审计，便于合规回溯；
- 任何失败都只告警并返回空描述，不阻断入库。

配置值在调用时从 :mod:`finance_rag.src.core.config` 读取（而不是导入期绑定），
这样测试可以直接替换阈值，也不需要为了测一个边界去改环境变量。
"""

from __future__ import annotations

import base64
import json
import logging
import time
import urllib.request
from typing import Any

from finance_rag.src.core import config
from finance_rag.src.utils import audit

logger = logging.getLogger(__name__)

# 未知尺寸时的宽容策略：不因「量不出来」就把图丢掉。
_JPEG_SOF_MARKERS = frozenset(range(0xC0, 0xD0)) - {0xC4, 0xC8, 0xCC}


def _png_dimensions(data: bytes) -> tuple[int, int] | None:
    if data[:8] != b"\x89PNG\r\n\x1a\n" or len(data) < 24:
        return None
    return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")


def _gif_dimensions(data: bytes) -> tuple[int, int] | None:
    if data[:6] not in (b"GIF87a", b"GIF89a") or len(data) < 10:
        return None
    return int.from_bytes(data[6:8], "little"), int.from_bytes(data[8:10], "little")


def _jpeg_dimensions(data: bytes) -> tuple[int, int] | None:
    index, size = 2, len(data)
    while index + 9 < size:
        if data[index] != 0xFF:
            index += 1
            continue
        marker = data[index + 1]
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            index += 2
            continue
        if marker == 0xD9:
            return None
        segment_length = int.from_bytes(data[index + 2 : index + 4], "big")
        if marker in _JPEG_SOF_MARKERS:
            height = int.from_bytes(data[index + 5 : index + 7], "big")
            width = int.from_bytes(data[index + 7 : index + 9], "big")
            return width, height
        index += 2 + segment_length
    return None


def _image_dimensions(data: bytes) -> tuple[int, int] | None:
    """从图片头部解析 ``(宽, 高)``；解析不出返回 None（调用方应宽容处理）。"""
    for parser in (_png_dimensions, _gif_dimensions, _jpeg_dimensions):
        try:
            dimensions = parser(data)
        except Exception:  # noqa: BLE001 - 头部残缺不应抛
            continue
        if dimensions:
            return dimensions
    return None


def _build_vision_payload(data: bytes, *, ext: str, prompt: str) -> dict[str, Any]:
    """构造 OpenAI 兼容的多模态请求体（图片以 data URL 形式内联）。"""
    encoded = base64.b64encode(data).decode("ascii")
    return {
        "model": config.IMAGE_CAPTION_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/{ext};base64,{encoded}"},
                    },
                ],
            }
        ],
    }


def _post_json(url: str, payload: dict[str, Any], headers: dict[str, str], timeout: float) -> Any:
    """最小 JSON POST（标准库实现，避免为一次调用引入新依赖）。"""
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - 配置化端点
        return json.loads(response.read().decode("utf-8"))


def _extract_text(response: Any) -> str:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return ""
    if isinstance(content, list):
        # 部分兼容端点返回分段内容
        return " ".join(
            str(part.get("text", "")) for part in content if isinstance(part, dict)
        ).strip()
    return str(content or "").strip()


def _call_vision(data: bytes, *, ext: str, prompt: str, timeout: float) -> str:
    api_key = config.IMAGE_CAPTION_API_KEY
    payload = _build_vision_payload(data, ext=ext, prompt=prompt)
    url = f"{config.IMAGE_CAPTION_BASE_URL.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    return _extract_text(_post_json(url, payload, headers, timeout))


def caption_image(data: bytes, *, ext: str = "png", source: str = "") -> str:
    """为一张图片生成中文描述；被跳过或失败时返回空串。"""
    if not data or len(data) < config.IMAGE_CAPTION_MIN_BYTES:
        return ""

    dimensions = _image_dimensions(data)
    if dimensions is not None:
        width, height = dimensions
        if min(width, height) < config.IMAGE_CAPTION_MIN_DIM_PX:
            logger.debug("图片过小（%sx%s），跳过描述：%s", width, height, source)
            return ""

    prompt = config.IMAGE_CAPTION_PROMPT
    timeout = config.IMAGE_CAPTION_TIMEOUT_SECONDS
    attempts = max(1, int(config.IMAGE_CAPTION_MAX_RETRIES) + 1)
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            text = _call_vision(data, ext=ext, prompt=prompt, timeout=timeout)
        except Exception as exc:  # noqa: BLE001
            # 描述失败不影响入库，因此这里对超时/限流/5xx/解析错误一视同仁地重试
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(min(2**attempt, 4))
            continue
        if text:
            # 成功外发才记审计：失败没有产生数据出境
            audit.log(
                "image_egress",
                resource=source,
                detail=f"{config.IMAGE_CAPTION_MODEL} 图片描述",
                model=config.IMAGE_CAPTION_MODEL,
                bytes=len(data),
            )
            return text
        last_error = None
        break

    if last_error is not None:
        logger.warning("图片描述失败（%s）：%s", source or "unknown", last_error)
    return ""


__all__ = ["caption_image"]
