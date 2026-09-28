"""金融 RAG 统一评测命令行。"""

from __future__ import annotations

import argparse
import csv
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any

from .dataset import DatasetValidationError, EvaluationDataset
from .statistics import EXPERIMENT_SPECS, compare_runs

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_RESULTS = ROOT / "scripts" / "results" / "eval"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _records_path(path: str | Path) -> Path:
    value = Path(path)
    return value / "records.jsonl" if value.is_dir() else value


def _write_comparison(report: dict[str, Any], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        f"# A/B 对比：{report['experiment']}", "",
        f"- 判定：{report['decision']}",
        f"- 主指标：{report['primary_metric']}",
        f"- 护栏：{', '.join(report['guardrails'])}",
        f"- Bootstrap：{report['bootstrap_samples']} 次（seed={report['seed']}）", "",
    ]
    for tier in ("gold", "silver"):
        section = report["tiers"][tier]
        lines += [f"## {tier.title()}（n={section['n']}）", "", "| 指标 | 基线 | 候选 | 差值 | 95% CI | 判定 | 覆盖率 |", "|---|---:|---:|---:|---|---|---:|"]
        for name, metric in section["metrics"].items():
            ci = "N/A" if metric["ci95"] is None else f"[{metric['ci95'][0]:.4f}, {metric['ci95'][1]:.4f}]"
            fmt = lambda value: "N/A" if value is None else f"{value:.4f}"  # noqa: E731
            lines.append(
                f"| {name} | {fmt(metric['baseline'])} | {fmt(metric['candidate'])} | "
                f"{fmt(metric['delta'])} | {ci} | {metric['verdict']} | {metric['coverage']:.1%} |"
            )
        lines.append("")
    (output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    with (output / "largest_changes.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["tier", "metric", "case_id", "delta"])
        writer.writeheader()
        for tier, section in report["tiers"].items():
            for metric, values in section["metrics"].items():
                for change in values["largest_changes"]:
                    writer.writerow({"tier": tier, "metric": metric, **change})


def _dataset_version(path: Path, explicit: str | None) -> str:
    if explicit:
        return explicit
    manifest_path = path.with_suffix(".manifest.json")
    if manifest_path.exists():
        return str(json.loads(manifest_path.read_text(encoding="utf-8"))["dataset_version"])
    return path.stem


def _cmd_dataset(args: argparse.Namespace) -> int:
    if args.dataset_action == "generate":
        from .testset_generator import (
            build_evidence_resolver,
            generate_evaluation_dataset,
            knowledge_base_fingerprint,
        )

        output = Path(args.out)
        if args.from_markdown:
            from .migration import migrate_markdown_dataset
            from finance_rag.src.infrastructure.vector_store.milvus_kb import get_knowledge_base

            kb = get_knowledge_base()
            dataset = migrate_markdown_dataset(
                args.from_markdown,
                build_evidence_resolver(kb),
                kb_fingerprint=knowledge_base_fingerprint(kb),
            )
            dataset.save(output)
        else:
            dataset, output = generate_evaluation_dataset(
                count=args.count, seed=args.seed, per_doc_limit=args.per_doc, out_path=output
            )
        review = Path(args.review_out) if args.review_out else output.with_suffix(".review.csv")
        dataset.export_review(review)
        print(f"已生成 {len(dataset.cases)} 条 JSONL 候选：{output}")
        print(f"人工审阅表：{review}")
        return 0

    dataset = EvaluationDataset.load(args.dataset)
    if args.dataset_action == "validate":
        resolver = None
        if args.online:
            from .testset_generator import build_evidence_resolver
            from finance_rag.src.infrastructure.vector_store.milvus_kb import get_knowledge_base
            resolver = build_evidence_resolver(get_knowledge_base())
        audit = dataset.validate(evidence_resolver=resolver)
        print(f"校验通过：{audit.case_count} 条，在线核对证据 {audit.checked_evidence} 条")
        return 0
    if args.dataset_action == "export-review":
        dataset.export_review(args.out)
        print(f"已导出审阅表：{args.out}")
        return 0
    if args.dataset_action == "import-review":
        updated = dataset.import_review(args.review)
        updated.save(args.out)
        print(f"已导入审核结果：{args.out}")
        return 0
    if args.dataset_action == "freeze":
        fingerprint = args.kb_fingerprint
        if not fingerprint:
            from .testset_generator import knowledge_base_fingerprint
            from finance_rag.src.infrastructure.vector_store.milvus_kb import get_knowledge_base
            fingerprint = knowledge_base_fingerprint(get_knowledge_base())
        manifest = dataset.freeze(
            args.out,
            kb_fingerprint=fingerprint,
            dataset_version=args.version,
            max_source_share=args.max_source_share,
        )
        print(f"已冻结 {manifest['case_count']} 条：{args.out}")
        print(f"数据集版本：{manifest['dataset_version']}；知识库指纹：{manifest['kb_fingerprint']}")
        return 0
    raise ValueError(args.dataset_action)


def _cmd_retrieval(args: argparse.Namespace) -> int:
    from .retrieval import LAYER_SPECS, build_retrieval_strategy
    from .runner import EvaluationRunner
    from .testset_generator import knowledge_base_fingerprint
    from finance_rag.src.infrastructure.vector_store.milvus_kb import get_knowledge_base

    dataset_path = Path(args.dataset)
    dataset = EvaluationDataset.load(dataset_path)
    invalid_status = [
        case.id for case in dataset.cases
        if (case.tier == "gold" and case.review_status != "approved")
        or (case.tier == "silver" and case.review_status != "auto_pass")
    ]
    if invalid_status and not args.diagnostic:
        raise DatasetValidationError(
            f"正式运行仅接受 approved Gold / auto_pass Silver；不合格条目：{invalid_status[:10]}"
        )
    if args.sample and args.sample < len(dataset.cases):
        cases = random.Random(args.seed).sample(list(dataset.cases), args.sample)
        dataset = EvaluationDataset(cases, source_path=dataset_path)

    kb = get_knowledge_base()
    current_fingerprint = knowledge_base_fingerprint(kb)
    manifest_path = dataset_path.with_suffix(".manifest.json")
    expected_fingerprint = None
    if manifest_path.exists():
        expected_fingerprint = json.loads(manifest_path.read_text(encoding="utf-8")).get("kb_fingerprint")
    if expected_fingerprint and expected_fingerprint != current_fingerprint and not args.allow_drift:
        raise DatasetValidationError(
            f"知识库指纹漂移：数据集={expected_fingerprint} 当前={current_fingerprint}"
        )
    comparable = not bool(expected_fingerprint and expected_fingerprint != current_fingerprint)

    version = _dataset_version(dataset_path, args.dataset_version)
    run_id = args.run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    layers = [layer.upper() for layer in args.strategy]
    unknown = sorted(set(layers) - set(LAYER_SPECS))
    if unknown:
        raise ValueError(f"未知检索层：{unknown}")
    runner = EvaluationRunner()
    run_records: dict[str, list[dict[str, Any]]] = {}
    for order, layer in enumerate(layers, 1):
        output = Path(args.outdir) / version / f"{run_id}-{layer}"
        strategy = build_retrieval_strategy(
            kb, layer, candidate_k=args.candidate_k, top_k=5
        )
        result = runner.run(
            dataset,
            strategy,
            "retrieval",
            output,
            metadata={
                "dataset_version": version,
                "kb_fingerprint": current_fingerprint,
                "expected_kb_fingerprint": expected_fingerprint,
                "comparable": comparable,
                "seed": args.seed,
                "execution_order": order,
                "execution_plan": layers,
            },
        )
        dataset.save(output / "dataset_snapshot.jsonl")
        run_records[layer] = result.records
        print(f"[{layer}] n={result.summary['n']} → {output}")

    pairs = (
        ("L0", "L1", "hybrid_vs_dense"),
        ("L2", "L3", "rerank_on_off"),
        ("L3", "L5", "cliff_on_off"),
    )
    for baseline, candidate, experiment in pairs:
        if baseline not in run_records or candidate not in run_records:
            continue
        report = compare_runs(
            run_records[baseline], run_records[candidate], EXPERIMENT_SPECS[experiment],
            bootstrap_samples=args.bootstrap, seed=args.seed,
        )
        comparison_dir = Path(args.outdir) / version / f"{run_id}-{experiment}"
        _write_comparison(report, comparison_dir)
        print(f"[{experiment}] {report['decision']} → {comparison_dir}")
    if not comparable:
        print("警告：诊断运行检测到知识库指纹漂移，结果标记为不可比较")
    return 0


def _cmd_ab(args: argparse.Namespace) -> int:
    baseline = _read_jsonl(_records_path(args.baseline))
    candidate = _read_jsonl(_records_path(args.candidate))
    report = compare_runs(
        baseline,
        candidate,
        EXPERIMENT_SPECS[args.experiment],
        bootstrap_samples=args.bootstrap,
        seed=args.seed,
    )
    _write_comparison(report, Path(args.out))
    print(f"A/B 判定：{report['decision']} → {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="金融 RAG 统一评测")
    groups = parser.add_subparsers(dest="command", required=True)

    dataset = groups.add_parser("dataset", help="数据集生成、校验、审核和冻结")
    dataset_actions = dataset.add_subparsers(dest="dataset_action", required=True)
    generate = dataset_actions.add_parser("generate")
    generate.add_argument("--count", type=int, default=240)
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--per-doc", type=int, default=40)
    generate.add_argument("--out", default=str(DEFAULT_DATA_DIR / "evaluation_candidates.jsonl"))
    generate.add_argument("--review-out")
    generate.add_argument("--from-markdown", help="只读迁移旧 Markdown；证据来源按 chunk 在线解析")
    validate = dataset_actions.add_parser("validate")
    validate.add_argument("--dataset", required=True)
    validate.add_argument("--online", action="store_true")
    export_review = dataset_actions.add_parser("export-review")
    export_review.add_argument("--dataset", required=True)
    export_review.add_argument("--out", required=True)
    import_review = dataset_actions.add_parser("import-review")
    import_review.add_argument("--dataset", required=True)
    import_review.add_argument("--review", required=True)
    import_review.add_argument("--out", required=True)
    freeze = dataset_actions.add_parser("freeze")
    freeze.add_argument("--dataset", required=True)
    freeze.add_argument("--out", required=True)
    freeze.add_argument("--version", required=True)
    freeze.add_argument("--kb-fingerprint")
    freeze.add_argument("--max-source-share", type=float, default=0.20)

    retrieval = groups.add_parser("retrieval")
    retrieval_actions = retrieval.add_subparsers(dest="retrieval_action", required=True)
    run = retrieval_actions.add_parser("run")
    run.add_argument("--dataset", required=True)
    run.add_argument("--strategy", nargs="+", default=["L0", "L1", "L2", "L3", "L5"])
    run.add_argument("--candidate-k", type=int, default=20)
    run.add_argument("--sample", type=int, default=0)
    run.add_argument("--seed", type=int, default=7)
    run.add_argument("--bootstrap", type=int, default=10_000)
    run.add_argument("--dataset-version")
    run.add_argument("--run-id")
    run.add_argument("--outdir", default=str(DEFAULT_RESULTS))
    run.add_argument("--allow-drift", action="store_true")
    run.add_argument("--diagnostic", action="store_true")

    ab = groups.add_parser("ab")
    ab_actions = ab.add_subparsers(dest="ab_action", required=True)
    compare = ab_actions.add_parser("run")
    compare.add_argument("--baseline", required=True)
    compare.add_argument("--candidate", required=True)
    compare.add_argument("--experiment", choices=sorted(EXPERIMENT_SPECS), required=True)
    compare.add_argument("--out", required=True)
    compare.add_argument("--bootstrap", type=int, default=10_000)
    compare.add_argument("--seed", type=int, default=7)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "dataset":
        return _cmd_dataset(args)
    if args.command == "retrieval" and args.retrieval_action == "run":
        return _cmd_retrieval(args)
    if args.command == "ab" and args.ab_action == "run":
        return _cmd_ab(args)
    raise ValueError("未知命令")
