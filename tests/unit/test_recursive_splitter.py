"""纯标准库递归切分器的等价性与冷启动守卫测试。

分三类：

1. **快速守卫**（默认执行）：边界用例的期望输出固化为字面量，不依赖 langchain。
2. **等价性比对**（``@pytest.mark.slow``）：与真实
   ``langchain_text_splitters.RecursiveCharacterTextSplitter`` 在真实语料上
   逐字符串比对。该测试需要导入重依赖（约 10s+），故默认跳过：
   ``pytest -m slow tests/unit/test_recursive_splitter.py``
3. **冷启动守卫**：断言热路径不导入 torch / transformers / sentence_transformers。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from finance_rag.src.rag.ingestion.recursive_splitter import RecursiveCharacterTextSplitter

# 生产配置：DOCLING_CHUNK_MAX_TOKENS=512、CHILD_OVERLAP_TOKENS=50，按 *4 换算为字符
_PROD_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
_PROD_CHUNK_SIZE = 512 * 4
_PROD_OVERLAP = 50 * 4

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _prod_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=_PROD_CHUNK_SIZE,
        chunk_overlap=_PROD_OVERLAP,
        separators=_PROD_SEPARATORS,
    )


# ---------------------------------------------------------------------------
# 1. 快速守卫：边界行为（不依赖 langchain）
# ---------------------------------------------------------------------------


def test_empty_text_returns_empty_list():
    assert _prod_splitter().split_text("") == []
    assert _prod_splitter().split_text("   ") == []


def test_text_shorter_than_chunk_size_is_single_chunk():
    splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    assert splitter.split_text("短文本") == ["短文本"]


def test_separator_is_kept_then_stripped_at_chunk_boundary():
    """keep_separator=True：分隔符并入前一段，随后由 _join_docs 的 strip() 去掉。

    （期望值取自 langchain 参照实现，见 ``test_matches_langchain_on_edge_cases``。）
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=10, chunk_overlap=0, separators=["\n\n", ""]
    )
    assert splitter.split_text("一二三四五\n\n六七八九十") == ["一二三四五", "六七八九十"]
    # 同一段内不切分时，分隔符原样保留
    assert splitter.split_text("abc\n\ndef") == ["abc\n\ndef"]


def test_oversized_piece_recurses_into_next_separator():
    """片段长度不满足严格 < chunk_size 时继续下钻到更细的分隔符。"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=4, chunk_overlap=0, separators=["\n\n", "\n", ""]
    )
    # \n\n 切出的 "abcdefgh" 仍超长 → 下钻到 "\n"（不命中）→ 逐字符兜底后合并
    assert splitter.split_text("abcdefgh\n\nijkl") == ["abcd", "efgh", "ijk", "l"]


def test_no_matching_separator_falls_back_to_last_separator():
    """没有任何分隔符命中时取 separators[-1]（逐字符兜底）。"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3, chunk_overlap=1, separators=["\n\n", "\n", ""]
    )
    chunks = splitter.split_text("abcdefg")
    assert all(len(chunk) <= 3 for chunk in chunks), chunks
    assert chunks[0].startswith("a")


def test_chunk_size_one_never_loops_forever():
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1, chunk_overlap=0, separators=["\n\n", ""]
    )
    assert splitter.split_text("abc") == ["a", "b", "c"]


def test_overlap_produces_repeated_content():
    """overlap>0 时相邻块应有重复内容（回退语义生效）。"""
    text = "。".join(["句子内容"] * 40) + "。"
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=30, chunk_overlap=10, separators=["。", ""]
    )
    chunks = splitter.split_text(text)
    assert len(chunks) > 1
    assert chunks[0][-5:] in chunks[1] or chunks[1][-5:] in chunks[0]


def test_invalid_arguments_fail_fast():
    with pytest.raises(ValueError):
        RecursiveCharacterTextSplitter(chunk_size=0)
    with pytest.raises(ValueError):
        RecursiveCharacterTextSplitter(chunk_overlap=-1)


def test_chinese_punctuation_separators_used_by_production_config():
    """生产分隔符配置下的基本形状：块长受控且内容不丢失。"""
    text = "这是第一句。这是第二句！这是第三句？这是第四句；这是第五句，结尾。"
    chunks = _prod_splitter().split_text(text)
    assert len("".join(chunks)) >= len(text.rstrip())
    assert all(len(chunk) <= _PROD_CHUNK_SIZE for chunk in chunks)


# ---------------------------------------------------------------------------
# 2. 等价性比对（slow，需重依赖）
# ---------------------------------------------------------------------------

_CORPUS = [
    _REPO_ROOT / "README.md",
    _REPO_ROOT / "finance_rag" / "src" / "eval" / "data" / "evaluation_qa_generated.md",
    _REPO_ROOT / "docs" / "superpowers" / "specs" / "2026-09-26-research-report-qa-design.md",
    _REPO_ROOT / "finance_rag" / "src" / "rag" / "ingestion" / "chunker.py",
]

_EDGE_CASES = [
    "",
    " ",
    "\n\n\n",
    "a",
    "一二三",
    "abcdefgh",
    "。",
    "。。。",
    "\n\n",
    "第一段。\n\n第二段。\n\n第三段。",
    "连续分隔符\n\n\n\n更多内容",
    "无标点长串" * 50,
    "混排。\n\n中文，标点；测试！结束？",
    "| 项目 | 数值 |\n| --- | --- |\n| 收入 | 100 |\n| 支出 | 50 |",
    "a" * 1000,
]

_PARAM_SETS = [
    (chunk_size, overlap)
    for chunk_size in (1, 2, 3, 16, 64, 512, _PROD_CHUNK_SIZE)
    for overlap in (0, 1, 50, _PROD_OVERLAP)
    if overlap < chunk_size
]


def _load_langchain_splitter():
    """导入 langchain 参照实现（重依赖，仅在 slow 测试中调用）。"""
    langchain_splitters = pytest.importorskip(
        "langchain_text_splitters", reason="需要 langchain_text_splitters 作为对照实现"
    )
    return langchain_splitters.RecursiveCharacterTextSplitter


@pytest.mark.slow
@pytest.mark.parametrize("path", _CORPUS, ids=lambda p: p.name)
def test_matches_langchain_on_real_corpus(path: Path):
    """真实语料：与 langchain 实现逐块完全一致。"""
    Reference = _load_langchain_splitter()
    if not path.exists():
        pytest.skip(f"语料不存在：{path}")
    text = path.read_text(encoding="utf-8")

    ours = _prod_splitter().split_text(text)
    theirs = Reference(
        chunk_size=_PROD_CHUNK_SIZE,
        chunk_overlap=_PROD_OVERLAP,
        separators=_PROD_SEPARATORS,
    ).split_text(text)
    assert ours == theirs, f"切块结果与 langchain 不一致：{path.name}"


@pytest.mark.slow
@pytest.mark.parametrize("chunk_size,overlap", _PARAM_SETS, ids=lambda v: str(v))
def test_matches_langchain_on_edge_cases(chunk_size: int, overlap: int):
    """边界用例 + 参数矩阵：与 langchain 实现逐块完全一致。"""
    Reference = _load_langchain_splitter()
    ours = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap, separators=_PROD_SEPARATORS
    )
    theirs = Reference(
        chunk_size=chunk_size, chunk_overlap=overlap, separators=_PROD_SEPARATORS
    )
    for case in _EDGE_CASES:
        assert ours.split_text(case) == theirs.split_text(case), (
            f"输入={case[:40]!r} chunk_size={chunk_size} overlap={overlap}"
        )


@pytest.mark.slow
def test_matches_langchain_for_production_chunker_output():
    """端到端：父块 → 子块 的结果与改用 langchain 时完全一致。"""
    Reference = _load_langchain_splitter()
    from finance_rag.src.rag.ingestion import chunker as chunker_module

    markdown = (
        "# 第一章 总则\n\n"
        + "第一条 为规范公司经营行为，防范金融风险，制定本办法。\n\n"
        + "第二条 本办法适用于公司全部业务条线。\n\n"
        + "## 第一节 适用范围\n\n"
        + "一、投研业务；\n二、合规风控业务；\n三、业务运营。\n\n"
        + "正文段落，包含足够长度以便触发递归切分。" * 30
    )

    hierarchical = chunker_module.HierarchicalChunker(max_tokens=8, parent_max_tokens=1, overlap=2)
    parents = hierarchical._split_into_parents(markdown, "doc.md", "doc")
    ours = [hierarchical._split_parent_into_children(parent.content) for parent in parents]

    class _Patched(chunker_module.HierarchicalChunker):
        def _split_parent_into_children(self, content: str) -> list[str]:
            del content
            raise NotImplementedError

    reference_splitter = Reference(
        chunk_size=8 * 4,
        chunk_overlap=2 * 4,
        separators=_PROD_SEPARATORS,
    )
    theirs = []
    for parent in parents:
        pieces: list[str] = []
        for kind, text in chunker_module._extract_atomic_units(parent.content):
            if kind == "table":
                summary = chunker_module._table_summary(text)
                if summary:
                    pieces.append(summary)
            else:
                pieces.extend(piece for piece in reference_splitter.split_text(text) if piece.strip())
        theirs.append(pieces)

    assert ours == theirs


# ---------------------------------------------------------------------------
# 3. 冷启动守卫
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "module",
    [
        "finance_rag.src.rag.ingestion.chunker",
        "finance_rag.src.rag.ingestion.recursive_splitter",
    ],
)
def test_hot_path_does_not_import_heavy_libraries(module: str):
    """切块热路径不得再间接导入 torch / transformers / sentence_transformers。"""
    code = (
        f"import {module};"
        "import sys;"
        "heavy=[m for m in ('torch','transformers','sentence_transformers','datasets','nltk')"
        " if m in sys.modules];"
        "print(','.join(heavy))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "", f"{module} 仍间接导入重型库：{result.stdout.strip()}"


def test_app_import_does_not_load_torch():
    """应用入口导入后不得加载重型库（冷启动守卫）。"""
    code = (
        "import finance_rag.src.main;"
        "import sys;"
        "heavy=[m for m in ('torch','transformers','sentence_transformers','datasets')"
        " if m in sys.modules];"
        "print(','.join(heavy))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "", f"应用导入仍加载重型库：{result.stdout.strip()}"


# ---------------------------------------------------------------------------
# 4. 启动不阻塞（模型预热后台化）
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lifespan_does_not_block_on_warmup(monkeypatch):
    """预热耗时不得阻塞启动：lifespan 进入 yield 应立即返回，就绪状态异步翻转。"""
    import asyncio
    import time

    from finance_rag.src import main as app_main

    started_at = time.perf_counter()
    release = asyncio.Event()

    async def slow_warmup() -> None:
        await release.wait()

    monkeypatch.setattr(app_main, "_warmup_models", slow_warmup)
    monkeypatch.setattr(app_main, "WARMUP_ENABLED", True)

    async with app_main.lifespan(app_main.app):
        elapsed = time.perf_counter() - started_at
        # 预热仍在阻塞中，但启动已经完成（未等待预热）
        assert elapsed < 1.0, f"启动被预热阻塞了 {elapsed:.2f}s"
        assert app_main._readiness["ready"] is False
        assert app_main._warmup_task is not None
        release.set()
        await asyncio.sleep(0.05)

    # 收尾后存活状态复位
    assert app_main._readiness["live"] is False
    assert app_main._warmup_task is None


@pytest.mark.asyncio
async def test_warmup_disabled_skips_background_task(monkeypatch):
    """WARMUP_ENABLED=false 时不做后台预热，直接置为就绪。"""
    from finance_rag.src import main as app_main

    monkeypatch.setattr(app_main, "WARMUP_ENABLED", False)
    async with app_main.lifespan(app_main.app):
        assert app_main._readiness["ready"] is True
        assert app_main._warmup_task is None
