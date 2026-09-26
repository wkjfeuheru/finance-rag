"""整表仓储（PostgreSQL JSONB）回归测试。

整表不进向量库：向量库只留「表头 + 首行」作索引，表体存在这里，
检索侧按 ``table_id`` 展开。因此这里必须锁住三件事：
二维数组保真、跨集合隔离、覆盖写语义。
"""

from finance_rag.src.infrastructure.relational_db.table_store import TableStoreRepository


def test_upsert_and_fetch_table_rows(tmp_path):
    repository = TableStoreRepository(f"sqlite:///{tmp_path / 'table.db'}")

    repository.upsert_many(
        "finance_kb",
        [
            {
                "id": "t1",
                "parent_id": "p1",
                "markdown": "| 年份 | 营收 |\n| --- | --- |\n| 2026E | 100 |",
                "payload": [["年份", "营收"], ["2026E", "100"]],
                "row_count": 1,
                "source": "s.pdf",
                "heading_path": "盈利预测",
                "start_page": 4,
            }
        ],
    )

    rows = repository.get_many("finance_kb", ["t1"])

    assert len(rows) == 1
    assert rows[0]["payload"] == [["年份", "营收"], ["2026E", "100"]]
    assert rows[0]["row_count"] == 1
    assert rows[0]["start_page"] == 4
    assert rows[0]["heading_path"] == "盈利预测"


def test_upsert_overwrites_and_isolates_collections(tmp_path):
    repository = TableStoreRepository(f"sqlite:///{tmp_path / 'table.db'}")
    item = {
        "id": "t1",
        "parent_id": "p1",
        "markdown": "m",
        "payload": [["a", "b"]],
        "row_count": 0,
        "source": "s.pdf",
    }

    repository.upsert_many("kb_a", [item])
    repository.upsert_many("kb_a", [{**item, "payload": [["updated", "b"]]}])
    assert repository.get_one("kb_a", "t1")["payload"] == [["updated", "b"]]

    repository.upsert_many("kb_b", [item])
    assert repository.get_one("kb_a", "t1")["payload"] == [["updated", "b"]]
    assert repository.get_one("kb_b", "t1")["payload"] == [["a", "b"]]


def test_delete_by_source_only_touches_given_collection(tmp_path):
    repository = TableStoreRepository(f"sqlite:///{tmp_path / 'table.db'}")
    item = {
        "id": "t1",
        "parent_id": "p1",
        "markdown": "m",
        "payload": [["a"]],
        "row_count": 0,
        "source": "s.pdf",
    }
    repository.upsert_many("kb_a", [item])
    repository.upsert_many("kb_b", [item])

    assert repository.delete_by_source("kb_a", "s.pdf") == 1
    assert repository.get_many("kb_a", ["t1"]) == []
    assert len(repository.get_many("kb_b", ["t1"])) == 1


def test_empty_input_is_a_noop(tmp_path):
    repository = TableStoreRepository(f"sqlite:///{tmp_path / 'table.db'}")

    repository.upsert_many("kb_a", [])

    assert repository.get_many("kb_a", ["missing"]) == []
