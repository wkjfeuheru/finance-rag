"""检索侧表格展开的回归测试。

背景：``_expand_to_parents`` 只在父块长度 ∈ [50, 3000] 时替换子块内容，
超长父块会**保留子块内容**（也就是「表头 + 首行」）。盈利预测表整表几乎必然
超过 3000 字符，因此表格必须走独立展开路径，否则整表方案会被静默吃掉。
"""

from finance_rag.src.rag.retrieval.hybrid_retriever import HybridRetriever


class _FakeTableStore:
    def __init__(self, rows: dict[str, dict]):
        self._rows = rows

    def get_batch(self, table_ids: list[str]) -> list[dict]:
        return [self._rows[tid] for tid in table_ids if tid in self._rows]


class _FakeParentStore:
    def __init__(self, rows: dict[str, dict]):
        self._rows = rows

    def get_batch(self, parent_ids: list[str]) -> list[dict]:
        return [self._rows[pid] for pid in parent_ids if pid in self._rows]


def _retriever(*, tables=None, parents=None, schema_fields=None) -> HybridRetriever:
    client = type(
        "_Client",
        (),
        {
            "describe_collection": lambda self, name: {
                "fields": [{"name": f} for f in (schema_fields or [])]
            }
        },
    )()
    return HybridRetriever(
        client=client,
        collection_name="finance_kb",
        embed_query_fn=lambda text: [0.0],
        parent_store=_FakeParentStore(parents or {}),
        table_store=_FakeTableStore(tables or {}),
    )


def _table_match(table_id: str = "t1") -> dict:
    return {
        "id": "c-table",
        "parent_id": table_id,
        "block_type": "table",
        "content": "| 年份 | 营收 |\n| 2026E | 100 |",
        "source": "s.pdf",
        "date": "2026-08-15",
        "score": 1.0,
    }


def _image_match() -> dict:
    return {
        "id": "c-image",
        "parent_id": "t1",
        "block_type": "image",
        "content": "燃机订单结构图",
        "source": "s.pdf",
        "date": "2026-08-15",
        "score": 0.9,
    }


def _table(table_id: str = "t1", rows: int = 400) -> dict:
    body = "\n".join(f"| 2026E | 第{i}行数据 | 明细{i} |" for i in range(rows))
    return {
        "id": table_id,
        "markdown": f"| 年份 | 项目 | 明细 |\n| --- | --- | --- |\n{body}",
        "payload": [["年份", "项目", "明细"]],
        "row_count": rows,
        "source": "s.pdf",
        "start_page": 4,
    }


def test_table_match_expands_full_table_beyond_parent_limit():
    """超过父块 3000 字符上限的整表必须完整展开——这正是独立路径的意义。"""
    retriever = _retriever(tables={"t1": _table(rows=200)})
    full_markdown = _table(rows=200)["markdown"]
    assert len(full_markdown) > 3000          # 前提：确实超过父块上限

    expanded = retriever._expand_tables([_table_match()], k=1)

    assert expanded[0]["content"] == full_markdown   # 一行都没丢
    assert "第199行数据" in expanded[0]["content"]
    assert expanded[0]["truncated"] is False
    assert expanded[0]["table_page"] == 4
    assert expanded[0]["table_row_count"] == 200


def test_oversized_table_is_truncated_and_explicitly_marked():
    """超过 8000 字符的表按行截断，并带上提示语——不静默删数据。"""
    retriever = _retriever(tables={"t1": _table(rows=1000)})

    expanded = retriever._expand_tables([_table_match()], k=1)
    content = expanded[0]["content"]

    assert expanded[0]["truncated"] is True
    assert content.endswith(HybridRetriever.TABLE_TRUNCATION_NOTE)
    assert len(content) <= HybridRetriever.TABLE_MAX_CHARS
    assert "| 年份 | 项目 | 明细 |" in content        # 表头保留
    assert all(line.startswith("|") for line in content.splitlines()[:-3])


def test_small_table_expands_without_truncation():
    retriever = _retriever(
        tables={
            "t1": {
                "id": "t1",
                "markdown": "| 年份 | 营收 |\n| --- | --- |\n| 2026E | 100 |",
                "row_count": 1,
                "source": "s.pdf",
                "start_page": 4,
            }
        }
    )

    expanded = retriever._expand_tables([_table_match()], k=1)

    assert expanded[0]["content"].endswith("| 2026E | 100 |")
    assert expanded[0]["truncated"] is False
    assert expanded[0]["table_page"] == 4


def test_expand_tables_keeps_summary_when_store_has_no_row():
    """整表缺失时保留索引摘要，绝不把内容清空。"""
    retriever = _retriever(tables={})
    match = _table_match()

    expanded = retriever._expand_tables([match], k=1)

    assert expanded[0]["content"] == match["content"]
    assert expanded[0]["truncated"] is False


def test_expand_tables_without_store_is_a_noop():
    retriever = _retriever(tables={})
    retriever._table_store = None
    match = _table_match()

    expanded = retriever._expand_tables([match], k=1)

    assert expanded[0]["content"] == match["content"]


def test_parent_expansion_skips_table_and_image_rows():
    """表格/图片块的 parent_id 指向各自的存储，不能被父块逻辑覆盖。"""
    retriever = _retriever(parents={"t1": {"id": "t1", "content": "父块" * 100}})
    matches = [_table_match(), _image_match()]

    expanded = retriever._expand_to_parents(matches, k=2)

    assert expanded[0]["content"] == matches[0]["content"]
    assert expanded[1]["content"] == matches[1]["content"]


def test_parent_expansion_still_works_for_text_rows():
    parent_text = "父块全文内容" * 20        # 长度需落在父块可替换区间 [50, 3000]
    retriever = _retriever(parents={"p1": {"id": "p1", "content": parent_text}})
    matches = [
        {"id": "c1", "parent_id": "p1", "block_type": "text", "content": "子块", "score": 1.0}
    ]

    expanded = retriever._expand_to_parents(matches, k=1)

    assert expanded[0]["content"] == parent_text


def test_output_fields_include_report_fields_when_schema_has_them():
    retriever = _retriever(
        schema_fields=["id", "content", "block_type", "start_page", "industry_l1"]
    )

    fields = retriever._output_fields()

    assert "block_type" in fields
    assert "industry_l1" in fields
    assert "security_code" not in fields      # 集合没有的字段不能请求
    assert "content" in fields


def test_output_fields_fall_back_to_base_when_schema_probe_fails():
    """旧集合（或探测失败）不能让整个检索挂掉，只能少取几个字段。"""

    class _Boom:
        def describe_collection(self, name):
            raise RuntimeError("集合不存在")

    retriever = HybridRetriever(
        client=_Boom(),
        collection_name="finance_kb",
        embed_query_fn=lambda text: [0.0],
        parent_store=None,
        table_store=None,
    )

    fields = retriever._output_fields()

    assert "id" in fields and "content" in fields
    assert "block_type" not in fields
