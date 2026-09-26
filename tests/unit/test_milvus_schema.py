"""研报元数据在 Milvus 侧的落地契约。

Milvus 集合是 ``enable_dynamic_field=False`` 的固定字段表，且标量**不接受 None**、
VARCHAR 超长会直接让整篇文档入库失败。因此这里锁住三件事：
字段齐备、默认值非空、超长值被截断而不是抛错。
"""

from finance_rag.src.infrastructure.vector_store.milvus_kb import (
    _REPORT_FIELD_SPECS,
    _add_report_fields,
    _report_scalars,
)


class _SchemaStub:
    def __init__(self) -> None:
        self.fields: list[tuple[str, int | None]] = []

    def add_field(self, name, dtype, max_length=None, **kwargs):  # noqa: ANN001
        self.fields.append((name, max_length))


def test_report_fields_cover_metadata_and_evidence_columns():
    schema = _SchemaStub()

    _add_report_fields(schema)

    names = [name for name, _ in schema.fields]
    assert names == [
        "security_code",
        "security_name",
        "industry_l1",
        "industry_l2",
        "report_type",
        "broker",
        "meta_source",
        "block_type",
        "start_page",
        "end_page",
        "image_key",
        "needs_review",
    ]
    limits = dict(schema.fields)
    assert limits["security_code"] == 32
    assert limits["report_type"] == 16
    assert limits["image_key"] == 256
    assert limits["needs_review"] is None  # BOOL 无长度


def test_report_field_specs_declare_varchar_lengths():
    varchar = {
        name: length for name, dtype, length in _REPORT_FIELD_SPECS if length is not None
    }

    assert varchar["security_name"] == 64
    assert varchar["broker"] == 64
    assert varchar["industry_l1"] == 32


def test_scalars_default_to_non_null_values():
    scalars = _report_scalars({}, {})

    assert scalars["security_code"] == ""
    assert scalars["industry_l1"] == ""
    assert scalars["block_type"] == "text"      # 缺省视为普通文本块
    assert scalars["start_page"] == 0           # 页码缺失就是 0，不猜
    assert scalars["end_page"] == 0
    assert scalars["needs_review"] is False
    assert all(value is not None for value in scalars.values())


def test_scalars_read_document_and_chunk_level_sources():
    scalars = _report_scalars(
        {
            "security_code": "600519",
            "security_name": "贵州茅台",
            "industry_l1": "食品饮料",
            "industry_l2": "白酒",
            "report_type": "个股",
            "broker": "中信证券",
            "meta_source": "regex",
        },
        {"block_type": "table", "start_page": 4, "end_page": 5, "image_key": "images/a/4-0.png"},
    )

    assert scalars["security_code"] == "600519"
    assert scalars["industry_l2"] == "白酒"
    assert scalars["block_type"] == "table"
    assert (scalars["start_page"], scalars["end_page"]) == (4, 5)
    assert scalars["image_key"] == "images/a/4-0.png"


def test_scalars_clip_overlong_values_instead_of_raising():
    scalars = _report_scalars(
        {
            "security_code": "6" * 100,
            "security_name": "名" * 100,
            "industry_l1": "行" * 100,
            "industry_l2": "业" * 100,
            "report_type": "类" * 100,
            "broker": "商" * 200,
            "meta_source": "s" * 100,
        },
        {"block_type": "t" * 100, "image_key": "k" * 500},
    )

    assert len(scalars["security_code"]) == 32
    assert len(scalars["security_name"]) == 64
    assert len(scalars["industry_l1"]) == 32
    assert len(scalars["industry_l2"]) == 32
    assert len(scalars["report_type"]) == 16
    assert len(scalars["broker"]) == 64
    assert len(scalars["meta_source"]) == 16
    assert len(scalars["block_type"]) == 16
    assert len(scalars["image_key"]) == 256


def test_scalars_tolerate_none_values():
    scalars = _report_scalars(
        {"security_code": None, "needs_review": None},
        {"start_page": None, "block_type": None},
    )

    assert scalars["security_code"] == ""
    assert scalars["block_type"] == "text"
    assert scalars["start_page"] == 0
    assert scalars["needs_review"] is False


def test_migration_defaults_cover_every_report_field():
    """旧集合搬运时，新集合要求的每一列都必须有默认值，否则整批插入失败。"""
    from scripts.migrate_kb_schema import REPORT_FIELDS, _derive_missing

    derived = _derive_missing({"content": "正文", "parent_id": "p1"})

    for field in REPORT_FIELDS:
        assert field in derived, field
        assert derived[field] is not None, field


def test_migration_only_copies_columns_present_in_backup():
    """备份集合缺少新列时必须跳过，否则 query 会因「字段不存在」直接失败。"""
    from scripts.migrate_kb_schema import _copy_fields_for

    class _Client:
        def describe_collection(self, name):
            return {"fields": [{"name": "id"}, {"name": "content"}, {"name": "category"}]}

    fields = _copy_fields_for(_Client(), "kb_backup")

    assert "category" in fields
    assert "id" in fields
    assert "security_code" not in fields
    assert "block_type" not in fields
