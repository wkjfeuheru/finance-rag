"""Run only query-rewrite, rerank and chunking paired ablations."""

from __future__ import annotations

import sys

import finance_rag.ablation as ablation


def main() -> int:
    # The shared implementation also supports retrieval-mode experiments.
    # This entry point intentionally selects the three experiments in this spec.
    ablation.EXPERIMENTS = tuple(
        item for item in ablation.EXPERIMENTS
        if item.name in {"query_rewrite", "rerank", "chunking"}
    )
    from scripts.run_ablation_benchmark import main as run
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
