"""集合重建后的清理契约。

这是本轮实测撞出来的静默陷阱：``rebuild_collection()`` 抹掉了集合数据，但指纹库
仍记着「这些文件已入库」，于是后续上传**全部被判为内容未变更而跳过**——不报错、
不告警，最后拿到一个永远是空的知识库。baseline 会基于空集合跑出一堆 0。
"""

from sqlalchemy import create_engine, text

from finance_rag.src.infrastructure.vector_store.fingerprint_store import FingerprintStore


def test_fingerprint_clear_reports_and_removes_everything(tmp_path):
    store = FingerprintStore(str(tmp_path / "fp.json"))
    store.update(source="a.pdf", sha256="h1", file_path="a.pdf", chunk_count=3, simhash=1)
    store.update(source="b.pdf", sha256="h2", file_path="b.pdf", chunk_count=4, simhash=2)

    assert store.clear() == 2

    assert store.get("a.pdf") is None
    assert store.is_unchanged("a.pdf", "h1") is False
    # 清空要落盘：重开一个实例也应为空（否则重启后旧指纹又回来了）
    assert FingerprintStore(str(tmp_path / "fp.json")).get("a.pdf") is None


def test_fingerprint_clear_on_empty_store_is_zero(tmp_path):
    store = FingerprintStore(str(tmp_path / "fp.json"))

    assert store.clear() == 0


def test_rebuild_clears_fingerprint_and_derived_stores(monkeypatch):
    """重建后：指纹清空 + PG 派生表按 collection 清空，且集合被重新创建。"""
    from finance_rag.src.infrastructure.vector_store.milvus_kb import KnowledgeBase

    engine = create_engine("sqlite://")
    with engine.begin() as connection:
        connection.execute(text(
            "CREATE TABLE parent_chunks (collection VARCHAR(64), id VARCHAR(64), "
            "content TEXT, heading VARCHAR(512), heading_path VARCHAR(1024), source VARCHAR(512))"
        ))
        connection.execute(text(
            "CREATE TABLE table_chunks (collection VARCHAR(64), id VARCHAR(64), "
            "parent_id VARCHAR(64), payload JSON, markdown TEXT, row_count INTEGER, "
            "source VARCHAR(512), heading_path VARCHAR(1024), start_page INTEGER)"
        ))
        connection.execute(
            text("INSERT INTO parent_chunks (collection, id) VALUES ('finance_kb', 'p1'), ('other_kb', 'p2')")
        )
        connection.execute(
            text("INSERT INTO table_chunks (collection, id) VALUES ('finance_kb', 't1'), ('other_kb', 't2')")
        )

    cleared: list[int] = []
    created: list[str] = []

    class _Client:
        def has_collection(self, name, timeout=None):  # noqa: ANN001
            return False

        def drop_collection(self, name):  # noqa: ANN001
            pass

        def load_collection(self, name, timeout=None):  # noqa: ANN001
            created.append(f"load:{name}")

    class _ParentStore:
        class _repo:  # noqa: N801
            @staticmethod
            def _get_engine():
                return engine

    class _TableStore:
        class _repo:  # noqa: N801
            @staticmethod
            def _get_engine():
                return engine

    class _Fingerprints:
        def clear(self):
            cleared.append(1)
            return 7

    kb = KnowledgeBase(collection_name="finance_kb")
    monkeypatch.setattr(kb, "_get_client", lambda: _Client())
    monkeypatch.setattr(kb, "_create_collection", lambda client: created.append("create"))
    monkeypatch.setattr(kb, "_get_parent_store", lambda: _ParentStore())
    monkeypatch.setattr(kb, "_get_table_store", lambda: _TableStore())
    monkeypatch.setattr(kb, "_get_fingerprint_store", lambda: _Fingerprints())

    kb.rebuild_collection()

    assert cleared == [1]                      # 指纹被清空
    assert created == ["create", "load:finance_kb"]
    with engine.connect() as connection:
        parents = connection.execute(text("SELECT collection, id FROM parent_chunks")).all()
        tables = connection.execute(text("SELECT collection, id FROM table_chunks")).all()
    # 只清当前集合，其它集合不受影响
    assert parents == [("other_kb", "p2")]
    assert tables == [("other_kb", "t2")]
