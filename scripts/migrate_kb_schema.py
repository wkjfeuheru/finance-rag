"""把旧 schema 的 Milvus 集合迁移到当前代码期望的 schema（**非破坏性**）。

背景
----
当前代码要求集合包含 ``chunk_key`` / ``content_hash`` / ``heading_path`` / ``is_deleted``
四个字段，而现存集合是旧 schema（14 个字段），检索会直接报错::

    MilvusException: (code=65535, message=field chunk_key not exist)

原始文档源文件已不在仓库内（``files/`` 目录不存在），因此**不能**简单
``rebuild_collection()`` 后重新入库——那会丢失整个语料。但旧集合里的数据
（含 512 维稠密向量与 BM25 稀疏向量）可以**原样搬运**，只需为新字段补值：

============= ==========================================================
新字段         取值方式
============= ==========================================================
chunk_key     ``sha256(f"{parent_id}:{content_hash}")``，与
              ``incremental.build_chunk_fingerprint`` 完全一致
content_hash  ``sha256(content.strip())``，同上
heading_path  ``""``（旧数据未记录标题路径，留空）
is_deleted    ``False``
============= ==========================================================

稠密/稀疏向量直接搬运，**不重新嵌入**，因此迁移是秒级的。

流程
----
1. 体检：对照当前代码 schema 列出缺失字段、统计行数（默认行为，不写入）
2. ``--apply`` 时：
   a. 把原集合**重命名**为 ``<name>_backup_<时间戳>``（保留备份，绝不删除）
   b. 用当前代码的 schema 新建同名集合
   c. 分页读出备份集合全部行，补出新字段后写入新集合并 flush
   d. 校验两侧行数一致，并做一次真实检索冒烟验证

用法
----
    python scripts/migrate_kb_schema.py                 # 只体检，不写入
    python scripts/migrate_kb_schema.py --apply         # 执行迁移
    python scripts/migrate_kb_schema.py --apply --collection other_kb
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 研报维度与证据字段（与 KnowledgeBase._REPORT_FIELD_SPECS 保持一致）。
# 旧集合没有这些列，搬运时按下面的默认值补齐。
REPORT_FIELDS = (
    "security_code", "security_name", "industry_l1", "industry_l2",
    "report_type", "broker", "meta_source", "block_type",
    "start_page", "end_page", "image_key", "needs_review",
)

# 当前代码期望的字段（与 KnowledgeBase.create_collection 保持一致）
REQUIRED_FIELDS = (
    "id", "content", "source", "title", "chunk", "parent_id", "chunk_key",
    "content_hash", "heading_path", "tenant_id", "category", "date", "version",
    "ingested_at", "is_current", "is_deleted", "dense_vector", "sparse_vector",
) + REPORT_FIELDS

# 迁移时需要搬运的字段（新字段在派生阶段补齐）。
#
# 注意：**不能包含 sparse_vector** —— Milvus 禁止通过 query 读取稀疏向量原始数据
# （``not allowed to retrieve raw data of field sparse_vector``），而且新集合的
# ``bm25_fn`` Function 会依据 ``content`` 在插入时自动重建稀疏向量
# （见 milvus_kb.add_parsed_document 的注释「sparse_vector 由 Milvus Function
# 自动生成，不传」）。dense_vector 可以正常读出并原样搬运。
COPY_FIELDS = (
    "id", "content", "source", "title", "chunk", "parent_id", "tenant_id",
    "category", "date", "version", "ingested_at", "is_current", "dense_vector",
) + REPORT_FIELDS


def _describe(client: Any, name: str) -> list[str]:
    desc = client.describe_collection(name)
    return [field["name"] for field in desc.get("fields", [])]


def _derive_missing(row: dict[str, Any]) -> dict[str, Any]:
    """补齐新 schema 中旧集合没有的字段（与入库链路的取值方式一致）。"""
    from finance_rag.src.infrastructure.vector_store.milvus_kb import _report_scalars
    from finance_rag.src.rag.ingestion.incremental import content_hash

    content = str(row.get("content") or "")
    digest = content_hash(content)
    parent_id = str(row.get("parent_id") or "")
    # chunk_key 的算法必须与 incremental.build_chunk_fingerprint 完全一致，
    # 否则增量入库时无法命中已有块、会退化为全部重新嵌入
    chunk_key = hashlib.sha256(f"{parent_id}:{digest}".encode()).hexdigest()
    return {
        "chunk_key": chunk_key,
        "content_hash": digest,
        "heading_path": "",
        "is_deleted": False,
        # 旧数据没有研报元数据：给非空默认值，避免新集合插入报缺字段
        **{
            key: value
            for key, value in _report_scalars(row, row).items()
            if key in REPORT_FIELDS
        },
    }


def _copy_fields_for(client: Any, backup: str) -> tuple[str, ...]:
    """只搬运备份集合里真实存在的列，避免对旧集合查询不存在的字段。"""
    present = set(_describe(client, backup))
    return tuple(field for field in COPY_FIELDS if field in present)


def _count(client: Any, name: str) -> int:
    result = client.query(collection_name=name, filter="", output_fields=["count(*)"])
    row = result[0] if result else {}
    return int(row.get("count(*)", 0))


def _read_all(client: Any, name: str, fields: tuple[str, ...]) -> list[dict[str, Any]]:
    """分页读出全部行（避免单次 query 的 limit 截断）。"""
    rows: list[dict[str, Any]] = []
    iterator = client.query_iterator(
        collection_name=name, filter="", output_fields=list(fields), batch_size=500
    )
    try:
        while True:
            batch = iterator.next()
            if not batch:
                break
            rows.extend(batch)
    finally:
        iterator.close()
    return rows


def inspect(client: Any, collection: str) -> tuple[list[str], int]:
    """体检：返回 (缺失字段, 行数)。"""
    present = set(_describe(client, collection))
    missing = [field for field in REQUIRED_FIELDS if field not in present]
    return missing, _count(client, collection)


def migrate(args: argparse.Namespace) -> int:
    from finance_rag.src.infrastructure.vector_store.milvus_kb import get_knowledge_base

    # 注意：不能传空字符串（会把默认集合名覆盖为空），空值必须走默认参数
    kb = get_knowledge_base(args.collection) if args.collection else get_knowledge_base()
    client = kb._get_client()
    collection = kb.collection_name

    print(f"目标集合：{collection}")
    missing, rows = inspect(client, collection)
    print(f"现有字段数：{len(_describe(client, collection))}")
    print(f"缺失字段：{missing or '（无，schema 已是新版）'}")
    print(f"现有行数：{rows}")

    if not missing and rows:
        print("\n无需迁移：集合 schema 已满足当前代码要求且已有数据。")
        return 0
    if not missing and not rows:
        # schema 已正确但集合为空（例如上次迁移中断在建表之后）：仍需搬运数据
        print("\n集合 schema 已正确但当前为空，将执行数据搬运。")
    if not args.apply:
        print("\n体检模式（未写入）。加 --apply 执行迁移。")
        print("迁移会先把原集合重命名为备份，再新建同名集合并搬运数据，原集合不会被删除。")
        return 0

    backup = args.backup or f"{collection}_backup_{time.strftime('%Y%m%d%H%M%S')}"
    if args.backup:
        print(f"\n[1-2/4] 从既有备份 {backup} 续做：跳过重命名与建表")
        if collection not in set(client.list_collections()):
            raise SystemExit(f"目标集合 {collection} 不存在，无法续做")
    else:
        print(f"\n[1/4] 重命名 {collection} -> {backup}")
        client.rename_collection(old_name=collection, new_name=backup)

        print(f"[2/4] 按当前 schema 新建 {collection}")
        kb._schema_fields = None
        kb._retriever = None
        kb.ensure_collection()
        new_fields = _describe(client, collection)
        print(f"      新集合字段数：{len(new_fields)}")

    print(f"[3/4] 从备份分页搬运 {rows} 行并补齐新字段")
    copy_fields = _copy_fields_for(client, backup)
    skipped = [field for field in COPY_FIELDS if field not in copy_fields]
    if skipped:
        print(f"      备份集合缺少这些列，将由默认值补齐：{', '.join(skipped)}")
    source_rows = _read_all(client, backup, copy_fields)
    print(f"      读出 {len(source_rows)} 行")
    payload: list[dict[str, Any]] = []
    for row in source_rows:
        merged = {field: row.get(field) for field in copy_fields}
        merged.update(_derive_missing(row))
        payload.append(merged)

    inserted = 0
    for start in range(0, len(payload), 200):
        chunk = payload[start : start + 200]
        client.insert(collection_name=collection, data=chunk)
        inserted += len(chunk)
        print(f"      已写入 {inserted}/{len(payload)}")
    client.flush(collection)

    print("[4/4] 校验")
    new_count = _count(client, collection)
    backup_count = _count(client, backup)
    print(f"      新集合行数：{new_count}；备份行数：{backup_count}")

    smoke_ok = False
    try:
        kb._retriever = None
        hits = kb.hybrid_search("合规管理办法", k=5, use_rerank=False)
        smoke_ok = bool(hits)
        print(f"      冒烟检索：{'通过，命中 ' + str(len(hits)) + ' 条' if smoke_ok else '返回空'}")
    except Exception as exc:  # noqa: BLE001
        print(f"      冒烟检索失败：{type(exc).__name__}: {exc}")

    print()
    if new_count == backup_count and smoke_ok:
        print(f"✅ 迁移成功。备份集合保留为 {backup}（确认无误后可自行删除）")
        return 0
    print("⚠ 迁移未完全成功，请检查上面输出；原数据完整保留在备份集合 "
          f"{backup}，可用 rename_collection 回滚。")
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="旧 schema Milvus 集合 -> 当前 schema 的非破坏性迁移",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--collection", default="", help="集合名（空 = 默认知识库）")
    parser.add_argument("--apply", action="store_true", help="真正执行迁移（默认只体检）")
    parser.add_argument("--backup", default="",
                        help="从指定备份集合续做（跳过重命名与建表，用于中断后重试）")
    args = parser.parse_args(argv)
    return migrate(args)


if __name__ == "__main__":
    raise SystemExit(main())
