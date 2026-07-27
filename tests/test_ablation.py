from __future__ import annotations

import unittest
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from finance_rag.ablation import (
    EXPERIMENTS,
    AblationBackend,
    AblationRunner,
    AblationStrategy,
    build_round_samples,
    evidence_recall,
    metadata_for_run,
    render_report,
    select_dataset_entries,
    select_experiments,
    summarize_ablation,
)
from finance_rag.evaluation import TestSetLoader, get_test_set_loader


@dataclass
class Entry:
    query: str
    related_docs: str = "doc.md"


class LoaderTests(unittest.TestCase):
    def test_loader_preserves_markdown_question_id(self):
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "evaluation_qa.md"
            path.write_text(
                "## 21. GDP是多少\n\n- 相关文档：report.pdf\n\n"
                "### 标准答案\n\n超过126万亿元。\n",
                encoding="utf-8",
            )
            entries = TestSetLoader(path).load_test_set()
        self.assertEqual(entries[0].question_id, 21)


class DatasetProfileTests(unittest.TestCase):
    def test_local_precision_profile_selects_exact_existing_ids(self):
        entries = get_test_set_loader().load_test_set()
        selected = select_dataset_entries(entries, "local_precision")
        self.assertEqual(
            [entry.question_id for entry in selected],
            [2, 4, 6, 8, 9, 10, 11, 14, 15, 16, 18, 21],
        )
        self.assertTrue(all(entry.required_facts for entry in selected))

    def test_local_precision_profile_rejects_missing_whitelist_entry(self):
        entries = get_test_set_loader().load_test_set()
        with self.assertRaisesRegex(RuntimeError, "缺少问题编号"):
            select_dataset_entries(
                [entry for entry in entries if entry.question_id != 21],
                "local_precision",
            )

    def test_hard_profile_still_selects_multi_evidence_entries(self):
        entries = get_test_set_loader().load_test_set()
        selected = select_dataset_entries(entries, "hard")
        self.assertTrue(selected)
        self.assertTrue(all(entry.test_type == "multi_evidence" for entry in selected))

    def test_all_profile_returns_every_entry(self):
        entries = get_test_set_loader().load_test_set()
        self.assertEqual(select_dataset_entries(entries, "all"), entries)


class SamplingTests(unittest.TestCase):
    def test_sampling_is_reproducible_and_unique_within_round(self):
        entries = [Entry(str(i)) for i in range(26)]
        first = build_round_samples(entries, rounds=5, samples_per_round=5, seed=7)
        second = build_round_samples(entries, rounds=5, samples_per_round=5, seed=7)
        self.assertEqual(
            [[item.query for item in group] for group in first],
            [[item.query for item in group] for group in second],
        )
        self.assertTrue(all(len({item.query for item in group}) == 5 for group in first))

    def test_invalid_sample_size_is_rejected(self):
        with self.assertRaises(ValueError):
            build_round_samples([Entry("a")], samples_per_round=2)

    def test_evidence_recall_counts_supported_facts(self):
        contexts = ["IOPV每15秒更新一次，溢价率用于判断价格偏离。"]
        facts = ("IOPV每15秒更新", "最小申赎单位50万份")
        self.assertEqual(evidence_recall(contexts, facts), 0.5)
        self.assertIsNone(evidence_recall(contexts, ()))

class ExperimentTests(unittest.TestCase):
    def test_each_experiment_changes_only_target_factor(self):
        expected = {
            "query_rewrite": {"name", "use_rewrite"},
            "rerank": {"name", "use_rerank"},
            "chunking": {"name", "chunking"},
            "retrieval": {"name", "dense_weight", "sparse_weight"},
        }
        for experiment in EXPERIMENTS:
            before = vars(experiment.before)
            after = vars(experiment.after)
            changed = {key for key in before if before[key] != after[key]}
            self.assertEqual(changed, expected[experiment.name])

    def test_dense_only_uses_search_and_never_hybrid_search(self):
        client = Mock()
        client.search.return_value = [[{
            "entity": {"content": "x", "source": "doc.md", "title": "d", "chunk": 0},
            "distance": 0.9,
        }]]
        kb = Mock()
        kb.collection_name = "dense"
        kb._get_client.return_value = client
        kb._embed_query.return_value = [0.1, 0.2]
        strategy = AblationStrategy(
            name="dense", sparse_weight=0.0, use_rerank=False
        )
        result = AblationBackend._dense_only_search(kb, "query", strategy)
        self.assertEqual(result[0]["source"], "doc.md")
        client.search.assert_called_once()
        client.hybrid_search.assert_not_called()


class ExperimentSelectionTests(unittest.TestCase):
    def test_chunking_selection_contains_only_chunking(self):
        selected = select_experiments(["chunking"])
        self.assertEqual([item.name for item in selected], ["chunking"])
        changed = {
            key for key, value in vars(selected[0].before).items()
            if value != vars(selected[0].after)[key]
        }
        self.assertEqual(changed, {"name", "chunking"})

    def test_selected_experiment_metadata_counts_fifty_evaluations(self):
        selected = select_experiments(["chunking"])
        metadata = metadata_for_run(5, 5, 7, {}, selected)
        self.assertEqual(metadata["total_evaluations"], 50)
        self.assertEqual([item["name"] for item in metadata["experiments"]], ["chunking"])

    def test_runner_emits_only_selected_experiment_with_paired_query(self):
        backend = Mock()
        backend.rewritten_query.side_effect = lambda query, _enabled: query
        backend.retrieve.return_value = []
        backend.generate.return_value = ""
        selected = select_experiments(["chunking"])
        rows = AblationRunner(backend, experiments=selected).run([[Entry("same query")]])
        self.assertEqual([row["experiment"] for row in rows], ["chunking", "chunking"])
        self.assertEqual([row["query"] for row in rows], ["same query", "same query"])
        self.assertEqual({row["arm"] for row in rows}, {"before", "after"})


class SummaryTests(unittest.TestCase):
    @staticmethod
    def row(experiment, arm, round_number, value):
        return {
            "experiment": experiment,
            "arm": arm,
            "round": round_number,
            "retrieval": {
                "hit_rate_at_5": value, "precision_at_5": value,
                "recall_at_5": value, "mrr_at_5": value,
                "evidence_recall_at_5": value,
            },
            "quality": {
                "faithfulness": value, "answer_relevancy": value,
                "context_precision": value, "context_recall": value,
            },
            "success": True, "errors": [],
        }

    def test_two_level_average_and_change(self):
        rows = []
        for experiment in EXPERIMENTS:
            rows.extend([
                self.row(experiment.name, "before", 1, 0.2),
                self.row(experiment.name, "before", 1, 0.4),
                self.row(experiment.name, "before", 2, 0.6),
                self.row(experiment.name, "after", 1, 0.4),
                self.row(experiment.name, "after", 1, 0.6),
                self.row(experiment.name, "after", 2, 0.8),
            ])
        summary = summarize_ablation(rows)
        comparison = summary["query_rewrite"]["comparison"]["quality"]["faithfulness"]
        self.assertEqual(comparison["before"], 0.45)
        self.assertEqual(comparison["after"], 0.65)
        self.assertEqual(comparison["absolute"], 0.2)

    def test_missing_metric_is_not_treated_as_zero(self):
        rows = []
        for experiment in EXPERIMENTS:
            before = self.row(experiment.name, "before", 1, 0.5)
            after = self.row(experiment.name, "after", 1, 0.7)
            before["quality"]["faithfulness"] = None
            rows.extend([before, after])
        summary = summarize_ablation(rows)
        result = summary["query_rewrite"]["comparison"]["quality"]["faithfulness"]
        self.assertIsNone(result["before"])
        self.assertIsNone(result["absolute"])

    def test_selected_experiment_report_includes_pair_and_missing_counts(self):
        rows = [
            self.row("chunking", "before", 1, 0.5),
            self.row("chunking", "after", 1, 0.7),
        ]
        summary = summarize_ablation(rows)
        metadata = metadata_for_run(1, 1, 7, {}, select_experiments(["chunking"]))
        report = render_report(metadata, summary)
        self.assertIn("切块策略", report)
        self.assertIn("有效配对数", report)
        self.assertIn("优化前缺失率", report)
        self.assertNotIn("查询改写", report)


if __name__ == "__main__":
    unittest.main()

