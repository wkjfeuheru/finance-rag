# Local Precision Chunking Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add and execute a reproducible 50-evaluation comparison of recursive chunking and Docling HybridChunker using 12 existing local-precision questions.

**Architecture:** Preserve the existing ablation runner and reporting pipeline. Add stable question IDs to parsed entries, centralize dataset-profile selection and local fact annotations in `finance_rag.ablation`, allow the runner to receive an explicit experiment subset, and expose both choices through the existing CLI without changing the default hard/four-experiment behavior.

**Tech Stack:** Python 3.11+, `unittest`, Docling `HybridChunker`, LangChain `RecursiveCharacterTextSplitter`, Milvus, BM25/dense hybrid retrieval, Ragas.

## Global Constraints

- Use only existing questions 2, 4, 6, 8, 9, 10, 11, 14, 15, 16, 18, and 21; do not edit their text or standard answers.
- Compare only recursive chunking against Docling HybridChunker for the formal run.
- Fix query rewrite on, reranking on, hybrid weights to dense 0.7 and sparse 0.3, `top_k=5`, and `rerank_top_n=3`.
- Execute 5 rounds with 5 distinct questions per round and a fixed seed; paired arms use identical questions.
- Write results under `artifacts/benchmarks/<timestamp>-local-precision-chunking/` without modifying earlier artifacts.
- Missing metrics are excluded rather than converted to zero; report valid pairs, errors, and missing rates.
- Start only `milvus-standalone`, `milvus-etcd`, and `milvus-minio` for the formal run; stop those three in every terminal outcome and verify port 19530 is closed.

---

## File Structure

- Modify `finance_rag/evaluation.py`: retain the numeric Markdown question ID on every `TestSetEntry`.
- Modify `finance_rag/ablation.py`: define the local-precision profile, attach local fact annotations, filter experiments, and make metadata reflect the selected experiment count.
- Modify `scripts/run_ablation_benchmark.py`: expose `local_precision` and `--experiments`, then pass selected experiments through the run.
- Modify `tests/test_ablation.py`: cover IDs, profile validation, annotations, experiment isolation, pairing, and evaluation count.
- Create runtime artifacts only under `artifacts/benchmarks/<timestamp>-local-precision-chunking/`.

### Task 1: Preserve Question IDs in the Evaluation Loader

**Files:**
- Modify: `finance_rag/evaluation.py:218-320`
- Test: `tests/test_ablation.py`

**Interfaces:**
- Produces: `TestSetEntry.question_id: int` populated from the `## N.` heading.
- Consumes: existing `TestSetLoader._QUESTION_PATTERN`, whose first capture group is the numeric ID.

- [ ] **Step 1: Write the failing loader test**

Add imports for `Path`, `tempfile.TemporaryDirectory`, and `TestSetLoader`, then add:

```python
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
```

- [ ] **Step 2: Run the test and confirm the missing-field failure**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_ablation.LoaderTests.test_loader_preserves_markdown_question_id -v`

Expected: FAIL because `TestSetEntry` has no `question_id`.

- [ ] **Step 3: Add the field and populate it**

Add `question_id: int = 0` to `TestSetEntry`, include it in `to_dict()`, and in `_parse()` set:

```python
question_id = int(match.group(1))
query = match.group(2).strip()
...
entries.append(TestSetEntry(
    question_id=question_id,
    query=query,
    ground_truth=ground_truth,
    related_docs=related_docs,
    related_sections=related_sections,
    required_facts=required_facts,
    test_type=test_type,
))
```

- [ ] **Step 4: Run the focused and existing tests**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_ablation -v`

Expected: all tests PASS.

- [ ] **Step 5: Commit only Task 1 files**

```powershell
git add -- "python project/finance-rag-retrieval/finance_rag/evaluation.py" "python project/finance-rag-retrieval/tests/test_ablation.py"
git commit -m "test: preserve evaluation question ids" -- "python project/finance-rag-retrieval/finance_rag/evaluation.py" "python project/finance-rag-retrieval/tests/test_ablation.py"
```

### Task 2: Add the Local-Precision Dataset Profile

**Files:**
- Modify: `finance_rag/ablation.py:25-125`
- Test: `tests/test_ablation.py`

**Interfaces:**
- Produces: `LOCAL_PRECISION_QUESTION_IDS: tuple[int, ...]`.
- Produces: `LOCAL_PRECISION_REQUIRED_FACTS: dict[int, tuple[str, ...]]`.
- Produces: `select_dataset_entries(entries: Sequence[TestSetEntry], profile: str) -> list[TestSetEntry]`.
- Consumes: `TestSetEntry.question_id`, `required_facts`, and `test_type`.

- [ ] **Step 1: Write failing profile-selection tests**

Add tests that load the real evaluation file and assert:

```python
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
```

Also assert `hard` still selects `test_type == "multi_evidence"` and `all` returns every entry.

- [ ] **Step 2: Run the focused tests and verify undefined-symbol failures**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_ablation.DatasetProfileTests -v`

Expected: FAIL because `select_dataset_entries` and constants are not defined.

- [ ] **Step 3: Define exact local fact annotations**

Define the ID tuple and a mapping containing short, independently matchable facts derived from the unchanged standard answers. Use at least two facts per question, including these required anchors:

```python
LOCAL_PRECISION_REQUIRED_FACTS = {
    2: ("每个自然年度最高缴费12000元", "未使用额度不能结转"),
    4: ("达到法定退休年龄可以领取", "可以一次性或分期领取"),
    6: ("IOPV每15秒更新一次", "溢价率等于市价除以IOPV减1"),
    8: ("申购使用一篮子证券", "最小申购赎回单位通常为50万份或100万份"),
    9: ("账户开通满2年以上", "前20个交易日日均资产不低于10万元"),
    10: ("T加2日公布中签结果", "T加2日16点前足额缴款"),
    11: ("赎回登记日前卖出或转股", "不处理会按约101元强制赎回"),
    14: ("估值低位加倍定投", "估值高位减少或暂停定投"),
    15: ("定投金额不超过月收入10%到20%", "不能影响正常生活"),
    16: ("达到预设收益率后全部赎回", "达到目标后分2至3批赎回"),
    18: ("普通股票涨跌幅限制为10%", "科创板和创业板涨跌幅限制为20%"),
    21: ("2023年GDP超过126万亿元", "同比增长5.2%"),
}
```

Implement selection with `dataclasses.replace(entry, required_facts=...)`. Validate that every configured ID occurs exactly once; raise `RuntimeError` listing missing or duplicate IDs. Preserve source order.

- [ ] **Step 4: Run all ablation tests**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_ablation -v`

Expected: all tests PASS, including unchanged hard/all behavior.

- [ ] **Step 5: Commit only Task 2 files**

```powershell
git add -- "python project/finance-rag-retrieval/finance_rag/ablation.py" "python project/finance-rag-retrieval/tests/test_ablation.py"
git commit -m "feat: add local precision evaluation profile" -- "python project/finance-rag-retrieval/finance_rag/ablation.py" "python project/finance-rag-retrieval/tests/test_ablation.py"
```

### Task 3: Make Experiment Selection Explicit

**Files:**
- Modify: `finance_rag/ablation.py:300-450`
- Modify: `scripts/run_ablation_benchmark.py:15-75`
- Test: `tests/test_ablation.py`

**Interfaces:**
- Produces: `select_experiments(names: Sequence[str] | None) -> tuple[Experiment, ...]`.
- Changes: `AblationRunner.__init__(backend, experiments=EXPERIMENTS)` and `run()` iterates `self.experiments`.
- Changes: `metadata_for_run(..., experiments: Sequence[Experiment] = EXPERIMENTS)` computes `total_evaluations = rounds * samples_per_round * 2 * len(experiments)` and serializes only selected experiments.
- CLI: `--experiments` accepts one or more of `query_rewrite`, `rerank`, `chunking`, and `retrieval`.

- [ ] **Step 1: Write failing experiment-isolation tests**

Add tests:

```python
def test_chunking_selection_contains_only_chunking(self):
    selected = select_experiments(["chunking"])
    self.assertEqual([item.name for item in selected], ["chunking"])
    changed = {
        key for key, value in vars(selected[0].before).items()
        if value != vars(selected[0].after)[key]
    }
    self.assertEqual(changed, {"name", "chunking"})

def test_selected_experiment_metadata_counts_fifty_evaluations(self):
    metadata = metadata_for_run(5, 5, 7, {}, select_experiments(["chunking"]))
    self.assertEqual(metadata["total_evaluations"], 50)
    self.assertEqual([x["name"] for x in metadata["experiments"]], ["chunking"])
```

Use a mock backend and one round to verify `AblationRunner(..., experiments=selected)` emits only `chunking` rows and paired arms receive the same query.

- [ ] **Step 2: Run tests and verify the new interfaces are missing**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_ablation.ExperimentSelectionTests -v`

Expected: FAIL because experiment selection and injection are not implemented.

- [ ] **Step 3: Implement experiment selection and runner injection**

Build a name-to-experiment dictionary from `EXPERIMENTS`. Reject unknown names and duplicates with `ValueError`. Default `None` returns all four experiments. Store the tuple on `AblationRunner` and use it for row generation and metadata.

- [ ] **Step 4: Wire the CLI**

Change `--dataset-profile` choices to `("hard", "all", "local_precision")`. Add:

```python
parser.add_argument(
    "--experiments",
    nargs="+",
    choices=("query_rewrite", "rerank", "chunking", "retrieval"),
)
```

In `main()`, call `select_dataset_entries`, then default `local_precision` to `("chunking",)` when `--experiments` is omitted; otherwise preserve the existing four-experiment default. Pass the selected tuple to `metadata_for_run` and `AblationRunner`.

- [ ] **Step 5: Run unit tests and CLI help**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_ablation -v
.\.venv\Scripts\python.exe scripts\run_ablation_benchmark.py --help
```

Expected: tests PASS; help lists `local_precision` and `--experiments`; existing `hard` invocation remains valid.

- [ ] **Step 6: Commit only Task 3 files**

```powershell
git add -- "python project/finance-rag-retrieval/finance_rag/ablation.py" "python project/finance-rag-retrieval/scripts/run_ablation_benchmark.py" "python project/finance-rag-retrieval/tests/test_ablation.py"
git commit -m "feat: isolate chunking ablation runs" -- "python project/finance-rag-retrieval/finance_rag/ablation.py" "python project/finance-rag-retrieval/scripts/run_ablation_benchmark.py" "python project/finance-rag-retrieval/tests/test_ablation.py"
```

### Task 4: Execute and Verify the Formal Benchmark

**Files:**
- Create: `artifacts/benchmarks/<timestamp>-local-precision-chunking/raw_results.json`
- Create: `artifacts/benchmarks/<timestamp>-local-precision-chunking/per_query.csv`
- Create: `artifacts/benchmarks/<timestamp>-local-precision-chunking/summary.json`
- Create: `artifacts/benchmarks/<timestamp>-local-precision-chunking/report.md`

**Interfaces:**
- Consumes: `--dataset-profile local_precision --experiments chunking --rounds 5 --samples-per-round 5 --seed 20260725`.
- Produces: one `chunking` comparison with 25 before and 25 after evaluations.

- [ ] **Step 1: Start only the Milvus dependency containers**

Run:

```powershell
docker start milvus-etcd milvus-minio milvus-standalone
```

Poll until `Test-NetConnection -ComputerName localhost -Port 19530 -InformationLevel Quiet` returns `True`. Inspect the three container states and stop immediately if any is unhealthy.

- [ ] **Step 2: Validate benchmark collections before spending model calls**

Use the project virtual environment to instantiate `make_knowledge_bases(Path("files"))`, call `get_stats()` for `docling` and `recursive`, and assert each collection exists, has `document_count == 7`, and has `chunk_count > 0`.

- [ ] **Step 3: Create the timestamped directory and run the benchmark**

Run the equivalent of:

```powershell
.\.venv\Scripts\python.exe scripts\run_ablation_benchmark.py `
  --skip-prepare `
  --dataset-profile local_precision `
  --experiments chunking `
  --rounds 5 `
  --samples-per-round 5 `
  --seed 20260725 `
  --output-dir artifacts\benchmarks\<timestamp>-local-precision-chunking
```

Capture stdout, stderr, and PID in that directory. If run asynchronously, create a heartbeat monitor that checks both parent and child Python processes and waits for `report.md` plus `summary.json`.

- [ ] **Step 4: Verify artifacts and result integrity**

Parse `summary.json` and assert:

```python
assert metadata["total_evaluations"] == 50
assert list(summary["experiments"]) == ["chunking"]
assert summary["experiments"]["chunking"]["before"]["sample_count"] == 25
assert summary["experiments"]["chunking"]["after"]["sample_count"] == 25
```

Confirm the report contains before, after, absolute change, relative change, valid sample counts, error counts, and missing rates for `context_precision`, `evidence_recall_at_5`, `context_recall`, `faithfulness`, `answer_relevancy`, and `quality_composite`.

- [ ] **Step 5: Stop Milvus in a `finally`-equivalent terminal action**

Run:

```powershell
docker stop milvus-standalone milvus-etcd milvus-minio
```

Verify all three states are `exited` and no TCP listener remains on local port 19530. Do not stop Docker Desktop or any other container.

- [ ] **Step 6: Report the real comparison**

Provide clickable local links to `report.md` and `summary.json`. State each requested metric as `before → after`, absolute change, relative change, valid paired sample count, errors, and missing rate. Do not claim Docling improved unless the measured results satisfy the design's conclusion rule.

---

## Plan Self-Review

- Spec coverage: dataset selection, unchanged QA content, isolated chunking experiment, fixed retrieval configuration, 50 evaluations, reporting, validation, and Milvus cleanup all map to explicit tasks.
- Placeholder scan: runtime timestamp is intentionally generated at execution; no implementation placeholder remains.
- Type consistency: `question_id`, `select_dataset_entries`, `select_experiments`, injected `experiments`, and metadata signatures are defined before consumption.
