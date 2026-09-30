"""断崖检测 + 指纹存储 + 上下文组装 独立测试（不依赖 Milvus）。"""

import hashlib
import json
import os
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# 断崖检测：直接测**生产实现**
# ---------------------------------------------------------------------------
# 这里曾放一份手抄的 `_apply_cliff_detection` 副本（理由是"避免导入 Milvus 依赖"），
# 结果是这些用例只验证那份副本：生产逻辑怎么改它们都照样通过，等于没有回归保护。
# 该函数是纯函数，导入 hybrid_retriever 并不需要可用的 Milvus 连接。
from finance_rag.src.rag.retrieval.hybrid_retriever import (  # noqa: E402
    _apply_cliff_detection,
)


class FingerprintStore:
    def __init__(self, store_path):
        self._path = Path(store_path)
        self._data = {}
        self._load()

    def _load(self):
        if self._path.exists():
            try:
                self._data = json.loads(self._path.read_text("utf-8"))
            except (json.JSONDecodeError, OSError):
                self._data = {}

    def _save(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2), "utf-8"
        )

    def is_unchanged(self, source, file_hash):
        r = self._data.get(source)
        return r is not None and r.get("hash") == file_hash

    def update(self, source, file_hash, file_path, chunk_count):
        import time
        self._data[source] = {
            "hash": file_hash,
            "file_path": file_path,
            "chunk_count": chunk_count,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        }
        self._save()

    def remove(self, source):
        self._data.pop(source, None)
        self._save()

    @staticmethod
    def hash_file(file_path):
        sha = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                sha.update(chunk)
        return sha.hexdigest()


def build_context(docs):
    if not docs:
        return "（未检索到相关文档）", []
    chunks = []
    sources = []
    for i, doc in enumerate(docs, start=1):
        content = (doc.get("content") or "")[:2000]
        title = doc.get("title", "文档")
        chunks.append(f"[{i}] {title}\n{content}")
        sources.append({
            "index": i, "title": title,
            "source": doc.get("source", ""), "chunk": doc.get("chunk"),
            "score": round(doc.get("score", 0.0), 4),
            "preview": (doc.get("content") or "")[:200],
        })
    return "\n\n---\n\n".join(chunks), sources


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCliffDetection:
    def test_no_cliff(self):
        c = [{"score": 0.95}, {"score": 0.90}, {"score": 0.88}, {"score": 0.85}]
        assert len(_apply_cliff_detection(c, threshold=0.35, min_results=1)) == 4

    def test_cliff_detected(self):
        c = [{"score": 0.95}, {"score": 0.92}, {"score": 0.90}, {"score": 0.30}, {"score": 0.25}]
        assert len(_apply_cliff_detection(c, threshold=0.35, min_results=1)) == 3

    def test_min_results(self):
        c = [{"score": 0.95}, {"score": 0.10}]
        assert len(_apply_cliff_detection(c, threshold=0.35, min_results=2)) == 2

    def test_high_threshold_truncates_less(self):
        """阈值方向：判据是「相对落差 > 阈值」，因此阈值越高越不容易截断。"""
        # 相邻落差 0.95→0.55 = 42%（>0.35 触发），0.55→0.40 = 27%（两档都不触发）
        c = [{"score": 0.95}, {"score": 0.55}, {"score": 0.40}]
        assert len(_apply_cliff_detection(c, threshold=0.35, min_results=1)) == 1
        assert len(_apply_cliff_detection(c, threshold=0.70, min_results=1)) == 3

    def test_extreme_cliff_keeps_default_floor(self):
        """极端断崖（首条即 99% 落差）下仍至少保留 RERANKER_CLIFF_MIN_RESULTS 条。

        回归点：默认下限曾为 1，实测出现过 20 条候选在首条之后直接截到只剩 1 条，
        一旦该条不相关就没有任何兜底上下文。
        """
        from finance_rag.src.core.config import RERANKER_CLIFF_MIN_RESULTS

        assert RERANKER_CLIFF_MIN_RESULTS >= 3
        # 首条 0.95，其余全部接近 0：断崖落在 i=0，若无下限则只保留 1 条
        c = [{"score": 0.95}] + [{"score": 0.005}] * 19
        kept = _apply_cliff_detection(c)  # 不传参 -> 使用生产默认值
        assert len(kept) == RERANKER_CLIFF_MIN_RESULTS

    def test_disabled(self):
        c = [{"score": 0.95}, {"score": 0.10}]
        assert len(_apply_cliff_detection(c, threshold=0, min_results=1)) == 2

    def test_empty(self):
        assert _apply_cliff_detection([], threshold=0.35, min_results=1) == []

    def test_single(self):
        assert len(_apply_cliff_detection([{"score": 0.95}], threshold=0.35, min_results=1)) == 1


class TestFingerprintStore:
    def test_new_file(self):
        with tempfile.TemporaryDirectory() as d:
            store = FingerprintStore(os.path.join(d, ".fp"))
            assert not store.is_unchanged("test.md", "abc")

    def test_match(self):
        with tempfile.TemporaryDirectory() as d:
            store = FingerprintStore(os.path.join(d, ".fp"))
            store.update("test.md", "abc123", "/tmp/test.md", 5)
            assert store.is_unchanged("test.md", "abc123")

    def test_wrong_hash(self):
        with tempfile.TemporaryDirectory() as d:
            store = FingerprintStore(os.path.join(d, ".fp"))
            store.update("test.md", "abc123", "/tmp/test.md", 5)
            assert not store.is_unchanged("test.md", "wrong")

    def test_remove(self):
        with tempfile.TemporaryDirectory() as d:
            store = FingerprintStore(os.path.join(d, ".fp"))
            store.update("test.md", "abc123", "/tmp/test.md", 5)
            store.remove("test.md")
            assert not store.is_unchanged("test.md", "abc123")

    def test_hash_file(self):
        with tempfile.TemporaryDirectory() as d:
            fp = os.path.join(d, "test.txt")
            Path(fp).write_text("hello world", "utf-8")
            h1 = FingerprintStore.hash_file(fp)
            assert len(h1) == 64
            h2 = FingerprintStore.hash_file(fp)
            assert h1 == h2

    def test_persistence(self):
        with tempfile.TemporaryDirectory() as d:
            fp_path = os.path.join(d, ".fp")
            store1 = FingerprintStore(fp_path)
            store1.update("doc.md", "hash123", "/tmp/doc.md", 10)
            store2 = FingerprintStore(fp_path)
            assert store2.is_unchanged("doc.md", "hash123")


class TestContextBuilding:
    def test_build_context(self):
        docs = [
            {"content": "房地产政策", "source": "r.pdf", "title": "报告", "chunk": 1, "score": 0.9},
            {"content": "金融稳定", "source": "r2.pdf", "title": "报告2", "chunk": 2, "score": 0.8},
        ]
        context, sources = build_context(docs)
        assert "[1]" in context
        assert "[2]" in context
        assert len(sources) == 2

    def test_build_context_empty(self):
        context, sources = build_context([])
        assert "未检索到" in context
        assert sources == []
