#!/usr/bin/env python3
"""性能测试脚本。

用法：
    python scripts/run_perf_benchmark.py [--repeat 5] [--output-dir DIR]

输出：
    artifacts/perf/<timestamp>/perf_results.json
    artifacts/perf/<timestamp>/perf_report.md
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from finance_rag.src.eval.perf_benchmark import PerfBenchmarkRunner, write_perf_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG 性能测试")
    parser.add_argument(
        "--repeat",
        type=int,
        default=5,
        help="每条查询重复检索次数（默认 5）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="报告输出目录（默认 artifacts/perf/<timestamp>）",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    if output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = ROOT / "artifacts" / "perf" / stamp
    else:
        output_dir = Path(output_dir)

    print("=" * 60)
    print("RAG 性能测试启动")
    print(f"重复次数：{args.repeat}")
    print(f"输出目录：{output_dir}")
    print("=" * 60)

    runner = PerfBenchmarkRunner(repeat=args.repeat)
    results = runner.run_all()

    write_perf_report(output_dir, results)

    print(f"\n性能测试完成。报告已输出到：{output_dir}")
    print(f"  - JSON：{output_dir / 'perf_results.json'}")
    print(f"  - Markdown：{output_dir / 'perf_report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
