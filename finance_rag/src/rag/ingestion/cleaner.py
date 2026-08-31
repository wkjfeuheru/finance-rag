"""内容清洗与去重：规则噪声过滤 + SimHash 近似去重（入库固定流程，不可关闭）。

* ``TextCleaner``：正则匹配常见噪声模式（页码、版权声明、导航链接、OCR 乱码），
  直接过滤；规则保守，避免误删正文。
* ``simhash`` / ``hamming_distance``：纯标准库实现的 64 位 SimHash，
  用于文档内段落级去重与文档间入库拦截（无新增依赖）。

清洗与两类去重均为固定处理逻辑，不提供配置开关；
仅 ``DUPLICATE_HAMMING_THRESHOLD`` 等算法参数以常量形式固化在本模块。
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter

# ---------------------------------------------------------------------------
# 噪声规则（保守匹配）
# ---------------------------------------------------------------------------

# 独立成行的页码：数字 / "第X页" / "- 3 -" / "3 / 42"
_PAGE_LINE_RE = re.compile(
    r"^\s*(?:"
    r"[-—–\s]*\d{1,4}[-—–\s]*"
    r"|第\s*[0-9一二三四五六七八九十百千]+\s*页(?:\s*[/，,共].*)?"
    r"|[-—–\s]*\d{1,4}\s*/\s*\d{1,4}[-—–\s]*"
    r")\s*$"
)

# 版权 / 免责声明行（逐行匹配，避免贪婪误删正文）
_COPYRIGHT_LINE_RE = re.compile(
    r"^\s*(?:"
    r"版权所有"
    r"|未经[^。\n]*许可"
    r"|免责声明\s*[:：]?"
    r"|风险提示\s*[:：]?"
    r"|风险揭示\s*[:：]?"
    r"|本(?:报告|文|材料|文件)版权"
    r"|本(?:报告|文|材料|文件)[^。\n]*仅供参考"
    r"|市场有风险[，,]\s*投资需谨慎"
    r").*$"
)

# 纯 URL 行
_URL_LINE_RE = re.compile(r"^\s*(?:https?://|www\.)\S+\s*$", re.IGNORECASE)

# Markdown 链接：去链接保留文字 [文字](http...) -> 文字
_MD_LINK_RE = re.compile(r"\[([^\]]*)\]\((?:https?://|www\.)[^)]*\)")

# 目录点线行：正文 + 连续点线引导符 + 结尾页码（如 "第一章 总则 ....... 1"）
_TOC_DOT_LINE_RE = re.compile(r"^.{4,}?(?:[.·…]\s*){3,}\s*\d{0,4}\s*$")

# OCR 常见乱码序列（保守替换为单个空格）
_OCR_GARBAGE_RE = re.compile(r"[\uFFFD\uf0b7\uf0d8]+")


class TextCleaner:
    """规则噪声过滤器（固定流程，始终执行）。"""

    def clean(self, markdown: str) -> str:
        """过滤噪声并归一化空白。"""
        if not markdown:
            return markdown
        return self._normalize_whitespace(self._filter_noise(markdown))

    def _filter_noise(self, text: str) -> str:
        """逐行过滤页码/版权行/URL 行；去链接保留文字。"""
        lines = text.split("\n")
        kept: list[str] = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                kept.append("")
                continue
            if _PAGE_LINE_RE.match(stripped):
                continue
            if _COPYRIGHT_LINE_RE.match(stripped):
                continue
            if _URL_LINE_RE.match(stripped):
                continue
            if _TOC_DOT_LINE_RE.match(stripped):
                continue
            kept.append(_MD_LINK_RE.sub(r"\1", line))
        return "\n".join(kept)

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        """压缩连续空行（>2 个空行 -> 1 个）并替换 OCR 乱码字符。"""
        text = _OCR_GARBAGE_RE.sub(" ", text)
        return re.sub(r"\n{3,}", "\n\n", text).strip()


# ---------------------------------------------------------------------------
# SimHash（纯标准库：字符 n-gram + md5 独立位投票 + 标点归一化）
# ---------------------------------------------------------------------------

_BIT_COUNT = 64
_NORMALIZE_RE = re.compile(r"[\W_]+", re.UNICODE)


def _normalize_for_hash(text: str) -> str:
    """哈希前归一化：去除标点/空白，保留中文与字母数字。

    使仅标点/空白差异的近似重复文本哈希一致，显著降低短文本噪声。
    """
    return _NORMALIZE_RE.sub("", text).lower()


def _token_hashes(text: str, n_gram: int = 3) -> list[int]:
    """提取字符 n-gram 并返回其 64 位哈希列表（md5 前 8 字节）。"""
    if len(text) < n_gram:
        grams = [text]
    else:
        grams = [text[i : i + n_gram] for i in range(len(text) - n_gram + 1)]
    return [
        int.from_bytes(hashlib.md5(g.encode("utf-8")).digest()[:8], "big")
        for g in grams
    ]


def simhash(text: str, n_gram: int = 3, normalize: bool = True) -> int:
    """计算 64 位 SimHash 指纹（字符 n-gram + md5 独立位投票）。

    ``normalize=True`` 时先去除标点/空白，使近似重复文本指纹更稳定。
    """
    source = _normalize_for_hash(text) if normalize else text
    if not source:
        return 0

    weights: Counter[int] = Counter()
    for h in _token_hashes(source, n_gram):
        for bit in range(_BIT_COUNT):
            if (h >> bit) & 1:
                weights[bit] += 1
            else:
                weights[bit] -= 1

    value = 0
    for bit in range(_BIT_COUNT):
        if weights[bit] > 0:
            value |= 1 << bit
    return value


def hamming_distance(a: int, b: int) -> int:
    """两个 64 位指纹的海明距离。"""
    return (a ^ b).bit_count()


# 文档间 SimHash 去重：海明距离阈值（固化常量，非配置开关）。
# 海明距离 <= 该值视为疑似重复文档，拦截入库。
DUPLICATE_HAMMING_THRESHOLD = 3


def deduplicate_paragraphs(
    markdown: str,
    threshold: int = 3,
    min_length: int = 40,
) -> str:
    """文档内段落级去重：近似重复段落只保留首次出现（固定流程，不可关闭）。

    * 仅对长度 >= ``min_length`` 的段落做去重（短行噪声差异大，误伤风险高）；
    * Markdown 表格行与标题行（以 ``#`` 开头）不参与去重；
    * 段落顺序保持首次出现位置。
    """
    paragraphs = markdown.split("\n\n")
    seen: list[tuple[int, int]] = []  # (simhash, 原文长度)
    kept: list[str] = []
    removed = 0
    for para in paragraphs:
        text = para.strip()
        if not text:
            kept.append(para)
            continue
        if len(text) < min_length or text.startswith("#") or text.startswith("|"):
            kept.append(para)
            continue

        sig = simhash(text)
        if any(hamming_distance(sig, s) <= threshold for s, _ in seen):
            removed += 1
            continue
        seen.append((sig, len(text)))
        kept.append(para)

    return "\n\n".join(kept)


def find_similar_documents(
    simhash_value: int,
    candidates: dict[str, int],
    threshold: int = DUPLICATE_HAMMING_THRESHOLD,
    exclude_source: str = "",
) -> list[tuple[str, int]]:
    """在候选指纹中查找与 ``simhash_value`` 海明距离 <= threshold 的文档。

    Args:
        simhash_value: 待比对指纹。
        candidates: {source: simhash} 映射。
        threshold: 海明距离阈值（<= 该值视为相似）。
        exclude_source: 排除的 source（替换更新时排除自身）。

    Returns:
        [(source, hamming_distance), ...]，按距离升序。
    """
    result = []
    for source, sig in candidates.items():
        if source == exclude_source or not isinstance(sig, int):
            continue
        dist = hamming_distance(simhash_value, sig)
        if dist <= threshold:
            result.append((source, dist))
    result.sort(key=lambda item: (item[1], item[0]))
    return result
