"""Run the four paired RAG ablation experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from finance_rag.src.eval.ablation import (
    DEFAULT_ROUNDS, DEFAULT_SAMPLES_PER_ROUND, DEFAULT_SEED,
    AblationBackend, AblationRunner, build_round_samples, default_output_dir,
    make_knowledge_bases, metadata_for_run, summarize_ablation,
    select_dataset_entries, select_experiments, write_ablation_reports,
)
from finance_rag.src.eval.ablation_compat import prepare_collections_windows_safe
from finance_rag.src.eval.ragas_eval import get_test_set_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 RAG 四组配对消融实验")
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--samples-per-round", type=int, default=DEFAULT_SAMPLES_PER_ROUND)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "files")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument(
        "--dataset-profile", choices=("hard", "all", "local_precision"), default="hard",
        help="hard 使用跨段落问题；all 使用完整测试集；local_precision 使用指定的12题",
    )
    parser.add_argument(
        "--experiments", nargs="+",
        choices=("query_rewrite", "rerank", "chunking", "retrieval"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    entries = get_test_set_loader().load_test_set()
    entries = select_dataset_entries(entries, args.dataset_profile)
    if not entries:
        raise RuntimeError(f"数据集配置 {args.dataset_profile} 没有测试问题")
    experiment_names = args.experiments
    if experiment_names is None and args.dataset_profile == "local_precision":
        experiment_names = ["chunking"]
    experiments = select_experiments(experiment_names)
    rounds = build_round_samples(
        entries, rounds=args.rounds,
        samples_per_round=args.samples_per_round, seed=args.seed,
    )
    kbs = make_knowledge_bases(args.data_dir)
    if not args.skip_prepare:
        print("正在准备 Docling 与递归切块 benchmark 集合……", flush=True)
        collection_info = prepare_collections_windows_safe(kbs, args.data_dir)
    else:
        collection_info = {}
        for name, kb in kbs.items():
            stats = kb.get_stats()
            if stats.get("chunk_count", 0) <= 0:
                raise RuntimeError(f"{name} benchmark 集合为空，不能使用 --skip-prepare")
            collection_info[name] = {"collection": kb.collection_name, "stats": stats}
    metadata = metadata_for_run(
        args.rounds, args.samples_per_round, args.seed, collection_info, experiments
    )
    output_dir = args.output_dir or default_output_dir(ROOT)
    print(f"将执行 {metadata['total_evaluations']} 次评估，输出：{output_dir}", flush=True)
    rows = AblationRunner(AblationBackend(kbs), experiments=experiments).run(rounds)
    summary = summarize_ablation(rows)
    write_ablation_reports(output_dir, metadata, rows, summary)
    print(json.dumps({name: item["comparison"] for name, item in summary.items()},
                     ensure_ascii=False, indent=2), flush=True)
    print(f"报告已生成：{output_dir / 'report.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

