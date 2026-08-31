"""RAG 优化点 A/B 实验命令行入口。

用法示例：

    # 列出全部内置实验
    python scripts/ab_rag.py --list

    # 基于当前知识库自动生成测试集（人工抽查后作为评测基准）
    python scripts/ab_rag.py --generate-testset --count 40

    # 跑单个实验（--fast = 3 项核心指标；--full = 6 项）
    python scripts/ab_rag.py --experiment hyde_on_off --fast --sample 5

    # 跑多个 / 全部实验
    python scripts/ab_rag.py --experiment dynamic_k_on_off hyde_on_off --fast
    python scripts/ab_rag.py --all --fast --sample 5

    # 自定义测试集 / 抽样种子
    python scripts/ab_rag.py --experiment hyde_on_off \\
        --testset path/to/qa.md --sample 10 --seed 7

结果落盘 ``scripts/results/ab/<实验名>/<时间戳>/``。
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# 项目根目录（scripts/ 的上一级），保证以任意 CWD 运行时都能导入项目模块
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)

DEFAULT_TESTSET = (
    ROOT / "finance_rag" / "src" / "eval" / "data" / "evaluation_qa_generated.md"
)


def _resolve_testset(path: str | None) -> Path:
    """优先显式路径，其次自动生成测试集；缺失时抛错并提示生成。"""
    if path:
        return Path(path)
    if DEFAULT_TESTSET.exists():
        return DEFAULT_TESTSET
    raise FileNotFoundError(
        f"测试集不存在：{DEFAULT_TESTSET}\n"
        "请先运行 python scripts/ab_rag.py --generate-testset"
    )


def _cmd_list() -> int:
    from finance_rag.src.eval.ab_catalog import AB_EXPERIMENTS

    for exp in AB_EXPERIMENTS:
        print(f"{exp.name:24s} {exp.label}  [{exp.kind}]")
    return 0


def _cmd_generate_testset(args: argparse.Namespace) -> int:
    from finance_rag.src.eval.testset_generator import generate_testset

    entries, out = generate_testset(
        count=args.count,
        seed=args.seed,
        per_doc_limit=args.per_doc,
        out_path=args.out if args.out else None,
    )
    if not entries:
        print("未生成任何测试集条目（请检查 LLM 配置与知识库是否为空）")
        return 1
    dist: dict[str, int] = {}
    for entry in entries:
        dist[entry["question_type"]] = dist.get(entry["question_type"], 0) + 1
    print(f"已生成 {len(entries)} 条 → {out}")
    print(f"类型分布：{dist}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from finance_rag.src.eval.ab_catalog import AB_EXPERIMENTS, select_experiments
    from finance_rag.src.eval.ab_runner import run_experiment

    experiments = AB_EXPERIMENTS if args.all else select_experiments(args.experiment)
    testset_path = _resolve_testset(args.testset)
    if not testset_path.exists():
        print(f"测试集不存在：{testset_path}")
        return 1

    for exp in experiments:
        print(f"\n=== 实验：{exp.name}（{exp.label}）===")
        run_experiment(
            exp,
            testset_path=testset_path,
            fast=not args.full,
            sample=args.sample,
            seed=args.seed,
            allow_missing=args.allow_missing,
            outdir=Path(args.outdir) if args.outdir else None,
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RAG 优化点 A/B 实验")
    parser.add_argument("--list", action="store_true", help="列出全部内置实验")
    parser.add_argument(
        "--generate-testset", action="store_true", help="基于知识库生成测试集"
    )
    parser.add_argument("--count", type=int, default=40, help="目标总题数（默认 40，约 7:2:1）")
    parser.add_argument("--per-doc", type=int, default=8, help="每文档纳入候选的最大块数（默认 8）")
    parser.add_argument("--out", default=None, help="测试集输出路径（默认自动）")
    parser.add_argument("--experiment", nargs="*", default=None, help="实验名（可多个）")
    parser.add_argument("--all", action="store_true", help="运行全部实验")
    parser.add_argument("--fast", action="store_true", help="快速模式（3 项指标，默认）")
    parser.add_argument("--full", action="store_true", help="完整模式（6 项指标）")
    parser.add_argument("--sample", type=int, default=5, help="抽样条目数（默认 5）")
    parser.add_argument("--seed", type=int, default=7, help="随机种子（默认 7）")
    parser.add_argument("--testset", default=None, help="自定义测试集路径")
    parser.add_argument(
        "--allow-missing", action="store_true", help="保留相关文档不在知识库的条目"
    )
    parser.add_argument("--outdir", default=None, help="结果输出根目录")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    if args.list:
        return _cmd_list()
    if args.generate_testset:
        return _cmd_generate_testset(args)
    if args.all or args.experiment:
        return _cmd_run(args)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
