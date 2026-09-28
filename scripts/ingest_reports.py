"""批量入库研报并输出逐篇验证报告。

为什么要一个脚本而不是界面点上传：

* 端到端验证要看的不是「入库成功没有」，而是**四项能力是否真的生效**——
  元数据抽取命中率、整表是否进了 PostgreSQL、图片描述是否入了索引、
  页码归属率。这些只能从 Milvus / PostgreSQL 直接查；
* baseline 必须可复现：语料换了、集合重建了，同一份脚本能重跑出同一份对照。

用法::

    python scripts/ingest_reports.py                      # 入库 assets/reports 下全部 PDF
    python scripts/ingest_reports.py --limit 3            # 只跑前 3 篇（冒烟）
    python scripts/ingest_reports.py --only 0026 --only 0027   # 按文件名片段筛选
    python scripts/ingest_reports.py --skip-ingest        # 只出验证报告，不入库
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REPORTS_DIR = ROOT / "assets" / "reports"
DEFAULT_CATEGORY = "investment_research"
TERMINAL_STATUSES = {"completed", "failed"}
LINE = "=" * 78
META_KEYS = (
    "security_code", "security_name", "industry_l1", "industry_l2",
    "report_type", "broker", "meta_source", "needs_review",
)


def _discover(args: argparse.Namespace) -> list[Path]:
    files = sorted(p for p in REPORTS_DIR.glob("*.pdf") if p.is_file())
    for fragment in args.only:
        files = [p for p in files if fragment in p.name]
    if args.limit:
        files = files[: args.limit]
    return files


async def _ingest_one(path: Path, category: str, timeout_seconds: int) -> dict:
    """提交一篇并入队等待结算；返回 ``{status, result, error}``。"""
    from starlette.datastructures import UploadFile

    from finance_rag.src.services.document_service import get_document_manager
    from finance_rag.src.services.task_service import get_task_manager

    size_mb = path.stat().st_size / 1024 / 1024
    manager = get_document_manager()
    try:
        with path.open("rb") as handle:
            upload = UploadFile(file=handle, filename=path.name, size=path.stat().st_size)
            submitted = await manager.upload_document_async(upload, category=category)
    except Exception as exc:  # noqa: BLE001 - 单篇失败不中断批次
        return {"status": "rejected", "error": f"{type(exc).__name__}: {exc}", "size_mb": size_mb}

    task_id = submitted["task_id"]
    tm = get_task_manager()
    waited = 0
    while waited < timeout_seconds:
        await asyncio.sleep(2)
        waited += 2
        task = tm.get(task_id)
        if task is None:
            continue
        if task.status.value in TERMINAL_STATUSES:
            return {
                "status": task.status.value,
                "result": task.result,
                "error": task.error,
                "size_mb": size_mb,
            }
    return {"status": "timeout", "error": f"等待 {timeout_seconds}s 未结算", "size_mb": size_mb}


def _verify(kb, source: str, table_store) -> dict:
    """从 Milvus / PostgreSQL 直接核对一篇文档的入库产物。"""
    import sqlalchemy as sa

    client = kb._get_client()
    rows = client.query(
        collection_name=kb.collection_name,
        filter=f'source == "{source}"',
        output_fields=["block_type", "start_page", "end_page", "image_key"],
        limit=16384,
    )
    block_types = Counter(str(r.get("block_type") or "text") for r in rows)
    paged = sum(1 for r in rows if int(r.get("start_page") or 0) > 0)
    with table_store._repo._get_engine().connect() as connection:
        table_rows = connection.execute(
            sa.text(
                "SELECT COUNT(*) AS n, COALESCE(SUM(row_count), 0) AS rows "
                "FROM table_chunks WHERE collection = :c AND source = :s"
            ),
            {"c": kb.collection_name, "s": source},
        ).mappings().one()
    docs = [d for d in kb.list_documents() if d.get("source") == source]
    metadata = {key: docs[0].get(key) for key in META_KEYS} if docs else {}
    return {
        "chunks": len(rows),
        "block_types": dict(block_types),
        "paged": paged,
        "images": block_types.get("image", 0),
        "tables": int(table_rows["n"]),
        "table_data_rows": int(table_rows["rows"]),
        "metadata": metadata,
    }


async def _run(args: argparse.Namespace) -> int:
    from finance_rag.src.core.config import KB_COLLECTION_NAME, MAX_UPLOAD_SIZE_MB
    from finance_rag.src.infrastructure.vector_store.milvus_kb import get_knowledge_base
    from finance_rag.src.rag.ingestion.chunk_worker import shutdown_chunk_pool
    from finance_rag.src.rag.retrieval.table_store import TableStore
    from finance_rag.src.services.ingestion_pipeline import drain_ingestion_pipeline

    files = _discover(args)
    if not files:
        print(f"未找到待入库文件：{REPORTS_DIR}")
        return 1

    kb = get_knowledge_base(KB_COLLECTION_NAME)
    table_store = TableStore(collection=KB_COLLECTION_NAME)

    print(LINE)
    print(f"语料目录：{REPORTS_DIR}")
    print(f"待处理  ：{len(files)} 篇，合计 "
          f"{sum(p.stat().st_size for p in files) / 1024 / 1024:.1f} MB")
    print(f"体积上限：{MAX_UPLOAD_SIZE_MB} MB（超出会被拒收）")
    print(LINE)

    oversized = [p for p in files if p.stat().st_size / 1024 / 1024 > MAX_UPLOAD_SIZE_MB]
    if oversized:
        print("[!] 以下文件超过上限，将被拒收：")
        for path in oversized:
            print(f"    {path.name}  {path.stat().st_size / 1024 / 1024:.1f} MB")

    outcomes: list[tuple[Path, dict]] = []
    if not args.skip_ingest:
        for index, path in enumerate(files, start=1):
            print(f"[{index}/{len(files)}] 入库 {path.name} …", flush=True)
            outcome = await _ingest_one(path, args.category, args.timeout)
            result = outcome.get("result") or {}
            note = ""
            if result.get("skipped"):
                note = "（内容未变更，已跳过）"
            elif outcome["status"] == "failed":
                note = f"  [x] {outcome.get('error')}"
            print(f"      status={outcome['status']} {note}", flush=True)
            outcomes.append((path, outcome))
        print(LINE)
        print("等待流水线排空…")
        await drain_ingestion_pipeline(timeout=120)
        # 必须显式关闭切块进程池：把清理留给解释器退出阶段会卡住/异常退出
        # （见 README「切块进程池」一节），表现为脚本成功跑完却返回非 0。
        shutdown_chunk_pool(wait=False)

    print(LINE)
    print("逐篇验证（Milvus 块 / PostgreSQL 整表 / 存储）")
    print(LINE)
    header = (f"{'文件':<34}{'块':>5}{'图':>4}{'表':>4}{'页码':>7}  "
              f"{'代码':<8}{'行业':<10}{'类型':<5}{'待确认':<7}")
    print(header)
    summary: dict[str, dict] = {}
    for path in files:
        info = _verify(kb, path.name, table_store)
        summary[path.name] = info
        meta = info["metadata"]
        paged_label = f"{info['paged']}/{info['chunks']}"
        print(f"{path.name[:33]:<34}{info['chunks']:>5}{info['images']:>4}"
              f"{info['tables']:>4}{paged_label:>7}  "
              f"{str(meta.get('security_code') or '-'):<8}"
              f"{str(meta.get('industry_l1') or '-'):<10}"
              f"{str(meta.get('report_type') or '-'):<5}"
              f"{'是' if meta.get('needs_review') else '否':<7}")

    print(LINE)
    print("语料级汇总")
    print(LINE)
    total_chunks = sum(i["chunks"] for i in summary.values())
    total_paged = sum(i["paged"] for i in summary.values())
    total_tables = sum(i["tables"] for i in summary.values())
    total_images = sum(i["images"] for i in summary.values())
    print(f"文档           = {len(summary)}")
    print(f"子块合计       = {total_chunks}")
    print(f"页码归属       = {total_paged}/{total_chunks}"
          f"（{total_paged / max(total_chunks, 1) * 100:.1f}%）")
    print(f"整表（PG）     = {total_tables} 张")
    print(f"图片块         = {total_images}")
    codes = Counter(
        i["metadata"].get("security_code") for i in summary.values()
        if i["metadata"].get("security_code")
    )
    print(f"抽出证券代码   = {len(codes)} 个不同标的 / {len(summary)} 篇")
    for code, count in codes.most_common():
        if count > 1:
            names = [n for n, i in summary.items() if i["metadata"].get("security_code") == code]
            print(f"    {code} 有 {count} 篇 -> 可用于跨券商对比：{names}")
    if not [c for c, n in codes.items() if n > 1]:
        print("    [!] 没有任何标的覆盖 >= 2 篇：baseline 的「跨券商对比同一指标」场景")
        print("      在当前语料上构造不出来（多跳题只能跨行业/跨主题）")
    industries = Counter(
        i["metadata"].get("industry_l1") for i in summary.values()
        if i["metadata"].get("industry_l1")
    )
    print(f"抽出行业       = {len(industries)} 个：{dict(industries)}")
    print(f"需要人工确认   = {sum(1 for i in summary.values() if i['metadata'].get('needs_review'))}"
          f"/{len(summary)}")
    sources = Counter(i["metadata"].get("meta_source") for i in summary.values())
    print(f"抽取来源分布   = {dict(sources)}")
    print(LINE)
    return 0


def main(argv: list[str] | None = None) -> int:
    # Windows 控制台常见 GBK 代码页：把不可编码字符替换掉，而不是让 print 抛
    # UnicodeEncodeError 把整批入库打断在半路（曾因此白跑一轮解析）。
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):  # pragma: no cover - 非常规 stdout
            pass

    parser = argparse.ArgumentParser(
        description="批量入库 assets/reports 下的研报并输出逐篇验证报告",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--limit", type=int, default=0, help="只处理前 N 篇（0 = 全部）")
    parser.add_argument("--only", action="append", default=[],
                        help="按文件名片段筛选，可重复；如 --only 0026 --only 0027")
    parser.add_argument("--category", default=DEFAULT_CATEGORY, help="入库分类")
    parser.add_argument("--timeout", type=int, default=600, help="单篇等待上限（秒）")
    parser.add_argument("--skip-ingest", action="store_true", help="跳过入库，只出验证报告")
    args = parser.parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
