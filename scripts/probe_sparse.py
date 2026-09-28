"""探查混合检索里 BM25/稀疏通道是否真的生效。

起因：分层基线里 L0（纯稠密）与 L1（稠密+BM25+RRF）的 recall@5、MRR、P@5
**逐位相同**。两条独立通道融合后排名一字不差，只有两种可能：稀疏通道没参与，
或者它对这批查询恰好零贡献。这个脚本直接把两条通道的原始结果打出来对比，
并检查集合里稀疏向量是否为空——避免把"BM25 没生效"误读成"BM25 没用"。

用法::

    python scripts/probe_sparse.py --query "国机汽车 2026H1 营业收入"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LINE = "=" * 78


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):  # pragma: no cover
            pass

    parser = argparse.ArgumentParser(description="探查 BM25/稀疏通道")
    parser.add_argument("--query", default="国机汽车 2026 年 H1 营业收入是多少")
    parser.add_argument("--k", type=int, default=20)
    args = parser.parse_args(argv)

    from finance_rag.src.core.config import KB_COLLECTION_NAME
    from finance_rag.src.infrastructure.vector_store.milvus_kb import get_knowledge_base

    kb = get_knowledge_base(KB_COLLECTION_NAME)
    client = kb._get_client()
    describe = client.describe_collection(collection_name=kb.collection_name)
    print(LINE)
    print(f"集合 {kb.collection_name}")
    for field in describe.get("fields", []):
        print(f"  field {field.get('name'):24} {field.get('type')} "
              f"{field.get('params', {}) or ''}")
    functions = describe.get("functions") or []
    print(f"  functions: {functions if functions else '(无)'}")
    print(LINE)

    # 稀疏向量是 BM25 Function 的输出，Milvus 不允许直接读原始数据
    # （``not allowed to retrieve raw data of field sparse_vector``），
    # 因此这里不抽样稀疏向量本身，改为直接用两条通道的检索结果对比。
    print(LINE)

    dense = kb.hybrid_search(args.query, k=args.k, use_rerank=False,
                             expand_parents=False, use_dense_only=True)
    hybrid = kb.hybrid_search(args.query, k=args.k, use_rerank=False,
                              expand_parents=False, use_dense_only=False)
    dense_ids = [str(item.get("id") or "") for item in dense]
    hybrid_ids = [str(item.get("id") or "") for item in hybrid]
    print(f"query = {args.query!r}   k={args.k}")
    print(f"  dense-only 返回 {len(dense_ids)} 条")
    print(f"  hybrid     返回 {len(hybrid_ids)} 条")
    print(f"  两条通道结果完全相同：{dense_ids == hybrid_ids}")
    print(f"  交集 {len(set(dense_ids) & set(hybrid_ids))} / 并集 {len(set(dense_ids) | set(hybrid_ids))}")
    print("  dense-only 前10：" + ", ".join(c[:8] for c in dense_ids[:10]))
    print("  hybrid     前10：" + ", ".join(c[:8] for c in hybrid_ids[:10]))
    for index, (left, right) in enumerate(zip(dense_ids, hybrid_ids), 1):
        if left != right:
            print(f"  首个差异位置 rank={index}: dense={left[:8]} hybrid={right[:8]}")
            break
    else:
        print("  逐位一致（在前述长度内）")
    # 纯词典命中探针：数字串/专名是 BM25 的强项，稠密向量反而容易漏；
    # 若连这种查询两条通道仍逐位一致，说明稀疏通道没有参与融合。
    probe = "153.62亿"
    dense_probe = [str(i.get("id") or "") for i in kb.hybrid_search(
        probe, k=args.k, use_rerank=False, expand_parents=False, use_dense_only=True)]
    hybrid_probe = [str(i.get("id") or "") for i in kb.hybrid_search(
        probe, k=args.k, use_rerank=False, expand_parents=False, use_dense_only=False)]
    print(f"  探针 query={probe!r}：dense==hybrid ? {dense_probe == hybrid_probe}")
    print("    dense  前5：" + ", ".join(c[:8] for c in dense_probe[:5]))
    print("    hybrid 前5：" + ", ".join(c[:8] for c in hybrid_probe[:5]))
    print(LINE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
