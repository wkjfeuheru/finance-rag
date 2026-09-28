"""父块 id 的唯一性与重新入库的幂等性。

实测撞到的主键冲突：43 页研报里有两段**完全相同**的块（风险提示/免责声明常逐页重复），
而父块 id 只按 ``source/heading/content`` 哈希、收了 ``index`` 却没用 →
两段算出同一个 id → 写入 PostgreSQL 时撞 ``parent_chunks_pkey``，整篇文档入库失败。

第二个坑：重新入库同一 source 时只是 DELETE 了批次内的 id，旧行残留；
而父块 id 是内容哈希，重复入库会算出同一批 id，于是「旧行还在 → 新 insert 冲突」。
"""

from sqlalchemy import create_engine, text

from finance_rag.src.rag.ingestion.chunker import HierarchicalChunker


def _parent_ids(markdown: str, source: str = "s.pdf"):
    chunker = HierarchicalChunker(max_tokens=512, parent_max_tokens=1)
    return [parent.id for parent in chunker._build_chunks(markdown, source, "t").parents]


def test_identical_sections_get_distinct_parent_ids():
    """同一篇里两段完全相同的块必须拿到不同的 id。"""
    blocked = "## 风险提示\n\n宏观经济风险；市场竞争风险。\n"
    markdown = f"## 投资要点\n\n正文。\n\n{blocked}\n## 盈利预测\n\n数据。\n\n{blocked}"

    ids = _parent_ids(markdown)

    assert len(ids) == len(set(ids)), "出现了重复的父块 id"
    assert len(ids) == 4


def test_identical_paragraphs_are_deduped_by_cleaner_and_ids_stay_unique():
    """无标题文档里重复段落会先被清洗层去重，因此不会走到 id 冲突那一步。

    这条用例记录的是**两层防护的关系**：段落级 SimHash 去重（`_clean_markdown`）
    先消掉重复段落；标题级重复块（表格/标题不参与去重）则由 id 里的序号兜住，
    见上一条用例。
    """
    markdown = "免责声明：本报告仅供内部使用，任何人不得转发。\n\n正文一。\n\n免责声明：本报告仅供内部使用，任何人不得转发。"

    ids = _parent_ids(markdown)

    # 重复段落被清洗层收敛（不足 3 段），且剩下的 id 互不相同
    assert len(ids) < 3
    assert len(set(ids)) == len(ids)


def test_parent_ids_are_stable_for_identical_input():
    """同内容同位置 → 同 id，保证可复现（增量/重跑语义不变）。"""
    markdown = "## 一\n\n甲。\n\n## 二\n\n乙。\n"

    assert _parent_ids(markdown) == _parent_ids(markdown)


def test_parent_ids_differ_when_order_changes():
    first = "## 一\n\n甲。\n\n## 二\n\n乙。\n"
    second = "## 二\n\n乙。\n\n## 一\n\n甲。\n"

    assert _parent_ids(first) != _parent_ids(second)


def test_reingest_purges_stale_derived_rows(monkeypatch):
    """覆盖模式下重新入库前要清掉同 source 的 PG 派生行，否则主键冲突。"""
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
        connection.execute(text(
            "INSERT INTO parent_chunks (collection, source) "
            "VALUES ('finance_kb', 'doc.pdf'), ('finance_kb', 'other.pdf')"
        ))
        connection.execute(text(
            "INSERT INTO table_chunks (collection, source) "
            "VALUES ('finance_kb', 'doc.pdf'), ('finance_kb', 'other.pdf')"
        ))

    class _Store:
        """桩：真实的 ParentStore / TableStore 在 store 层暴露 delete_by_source。"""

        def delete_by_source(self, source: str) -> int:
            with engine.begin() as connection:
                for table in ("parent_chunks", "table_chunks"):
                    connection.execute(
                        text(f"DELETE FROM {table} WHERE collection=:c AND source=:s"),
                        {"c": "finance_kb", "s": source},
                    )
            return 1

    from finance_rag.src.infrastructure.vector_store.milvus_kb import KnowledgeBase

    kb = KnowledgeBase(collection_name="finance_kb")
    monkeypatch.setattr(kb, "_get_parent_store", lambda: _Store())
    monkeypatch.setattr(kb, "_get_table_store", lambda: _Store())

    kb._purge_derived_rows("doc.pdf")

    with engine.connect() as connection:
        parents = connection.execute(text("SELECT source FROM parent_chunks")).scalars().all()
        tables = connection.execute(text("SELECT source FROM table_chunks")).scalars().all()
    assert parents == ["other.pdf"]      # 只清当前 source
    assert tables == ["other.pdf"]
