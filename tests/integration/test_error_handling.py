"""断崖检测 + 指纹存储 + 上下文组装 独立测试（不依赖 Milvus）。"""

import os
import tempfile
import hashlib
import json
from pathlib import Path

os.environ["RERANKER_CLIFF_THRESHOLD"] = "0.35"
os.environ["RERANKER_CLIFF_MIN_RESULTS"] = "1"


# ---------------------------------------------------------------------------
# 独立实现（与 finance_rag 模块逻辑一致，避免导入 Milvus 依赖）
# ---------------------------------------------------------------------------

def _apply_cliff_detection(candidates, threshold=0.35, min_results=1):
    if not candidates or threshold <= 0 or len(candidates) <= 1:
        return candidates
    scores = [c.get("rerank_score", 0.0) for c in candidates]
    cut = len(candidates)
    for i in range(len(scores) - 1):
        if scores[i] <= 0:
            continue
        drop = (scores[i] - scores[i + 1]) / max(scores[i], 0.001)
        if drop > threshold:
            cut = max(min_results, i + 1)
            break
    return candidates[:cut]


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
        c = [{"rerank_score": 0.95}, {"rerank_score": 0.90}, {"rerank_score": 0.88}, {"rerank_score": 0.85}]
        assert len(_apply_cliff_detection(c, threshold=0.35, min_results=1)) == 4

    def test_cliff_detected(self):
        c = [{"rerank_score": 0.95}, {"rerank_score": 0.92}, {"rerank_score": 0.90}, {"rerank_score": 0.30}, {"rerank_score": 0.25}]
        assert len(_apply_cliff_detection(c, threshold=0.35, min_results=1)) == 3

    def test_min_results(self):
        c = [{"rerank_score": 0.95}, {"rerank_score": 0.10}]
        assert len(_apply_cliff_detection(c, threshold=0.35, min_results=2)) == 2

    def test_disabled(self):
        c = [{"rerank_score": 0.95}, {"rerank_score": 0.10}]
        assert len(_apply_cliff_detection(c, threshold=0, min_results=1)) == 2

    def test_empty(self):
        assert _apply_cliff_detection([], threshold=0.35, min_results=1) == []

    def test_single(self):
        assert len(_apply_cliff_detection([{"rerank_score": 0.95}], threshold=0.35, min_results=1)) == 1


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