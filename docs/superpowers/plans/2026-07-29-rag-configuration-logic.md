# RAG Configuration Logic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make project-root environment loading, optimization booleans, Chinese tokenizer selection, and global/request-level BGE reranker control deterministic and tested without changing the DashScope embedding path.

**Architecture:** Keep `finance_rag.config` as the public constant-based configuration interface, but centralize environment loading and strict boolean parsing in small testable functions. Derive the tokenizer once in that module, then compute one effective reranker flag at the chat boundary and pass it unchanged to either retrieval path.

**Tech Stack:** Python 3, python-dotenv, unittest, unittest.mock, LangChain async chat pipeline.

## Global Constraints

- Process environment variables take precedence over repository-root `.env`, which takes precedence over code defaults.
- `USE_ZH_TOKENIZER=true` always forces `BAAI/bge-large-zh-v1.5`.
- `ENABLE_RERANKER` defaults to `true`; a request may disable but may not override a global disable.
- Dense retrieval remains DashScope `text-embedding-v2` with `EMBEDDING_DIM=1536`.
- Do not rebuild, delete, migrate, or rename a Milvus collection.
- Do not add runtime dependencies.
- Preserve unrelated existing working-tree changes; stage only task-specific hunks when committing.

---

## File Map

- `finance_rag/config.py`: project-root dotenv loading, strict boolean parser, optimization constants, tokenizer derivation, reranker global constant.
- `finance_rag/chat.py`: calculation and propagation of the effective reranker decision.
- `.env.example`: documented runtime switches and precedence.
- `tests/test_config.py`: isolated unit coverage for dotenv loading, boolean parsing, tokenizer precedence, and all boolean constants.
- `tests/test_chat_reranker.py`: observable chat-boundary coverage for effective reranking in normal and multi-stage retrieval.

### Task 1: Centralize environment and optimization configuration

**Files:**
- Create: `tests/test_config.py`
- Modify: `finance_rag/config.py:1-101`
- Modify: `.env.example:1-76`

**Interfaces:**
- Produces: `load_project_env(project_root: Path) -> None`
- Produces: `env_bool(name: str, default: bool) -> bool`
- Produces: `resolve_chunk_tokenizer(use_zh: bool, configured: str | None) -> str`
- Produces: `ENABLE_RERANKER: bool`
- Preserves: all existing exported configuration constants and `model`.

- [ ] **Step 1: Write failing tests for dotenv precedence and strict booleans**

Create `tests/test_config.py` with real temporary `.env` files and isolated process environment:

```python
from __future__ import annotations

import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from finance_rag.config import env_bool, load_project_env


class ProjectEnvironmentTests(unittest.TestCase):
    def test_project_env_loads_values_without_overwriting_process_environment(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / ".env").write_text(
                "ENABLE_SMART_CHUNKER=true\nRERANKER_DEVICE=cuda\n",
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {"RERANKER_DEVICE": "cpu"},
                clear=True,
            ):
                load_project_env(root)
                self.assertEqual(os.environ["ENABLE_SMART_CHUNKER"], "true")
                self.assertEqual(os.environ["RERANKER_DEVICE"], "cpu")


class BooleanConfigTests(unittest.TestCase):
    def test_env_bool_accepts_documented_true_and_false_values(self):
        cases = {
            "true": True,
            " 1 ": True,
            "YES": True,
            "On": True,
            "false": False,
            " 0 ": False,
            "NO": False,
            "Off": False,
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw), patch.dict(
                os.environ, {"FEATURE_FLAG": raw}, clear=True
            ):
                self.assertIs(env_bool("FEATURE_FLAG", not expected), expected)

    def test_env_bool_uses_default_only_when_variable_is_absent(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertTrue(env_bool("FEATURE_FLAG", True))
            self.assertFalse(env_bool("FEATURE_FLAG", False))

    def test_env_bool_rejects_invalid_nonempty_value_with_name_and_value(self):
        with patch.dict(os.environ, {"FEATURE_FLAG": "enabled"}, clear=True):
            with self.assertRaisesRegex(
                ValueError, "FEATURE_FLAG.*enabled"
            ):
                env_bool("FEATURE_FLAG", False)
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_config -v
```

Expected: import failure because `env_bool` and `load_project_env` do not exist.

- [ ] **Step 3: Implement project-root dotenv loading and strict boolean parsing**

In `finance_rag/config.py`, replace the bare `load_dotenv()` call with these interfaces and invoke the loader before reading constants:

```python
PROJECT_ROOT = Path(__file__).resolve().parents[1]

_TRUE_VALUES = frozenset({"true", "1", "yes", "on"})
_FALSE_VALUES = frozenset({"false", "0", "no", "off"})


def load_project_env(project_root: Path = PROJECT_ROOT) -> None:
    load_dotenv(dotenv_path=project_root / ".env", override=False)


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise ValueError(
        f"Environment variable {name} must be a boolean value; got {raw!r}"
    )


load_project_env()
```

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_config -v
```

Expected: all three environment/boolean tests pass.

- [ ] **Step 5: Write failing tokenizer and optimization-switch tests**

Extend `tests/test_config.py`:

```python
from finance_rag.config import resolve_chunk_tokenizer


class ChunkTokenizerTests(unittest.TestCase):
    def test_chinese_switch_forces_bge_over_explicit_tokenizer(self):
        self.assertEqual(
            resolve_chunk_tokenizer(True, "custom/tokenizer"),
            "BAAI/bge-large-zh-v1.5",
        )

    def test_disabled_chinese_switch_uses_explicit_tokenizer(self):
        self.assertEqual(
            resolve_chunk_tokenizer(False, "custom/tokenizer"),
            "custom/tokenizer",
        )

    def test_disabled_chinese_switch_defaults_to_minilm(self):
        self.assertEqual(
            resolve_chunk_tokenizer(False, None),
            "sentence-transformers/all-MiniLM-L6-v2",
        )


class OptimizationSwitchTests(unittest.TestCase):
    def test_every_boolean_switch_uses_strict_parser(self):
        switch_names = (
            "USE_ZH_TOKENIZER",
            "ENABLE_RERANKER",
            "CHAT_ENABLE_QUERY_REWRITE",
            "ENABLE_SMART_CHUNKER",
            "ENABLE_MULTI_STAGE_RETRIEVAL",
            "ENABLE_METADATA_FILTER",
            "ENABLE_FINANCIAL_EXPERT_PROMPT",
            "ENABLE_CITATION_VALIDATION",
        )
        for name in switch_names:
            with self.subTest(name=name), patch.dict(
                os.environ, {name: "invalid"}, clear=True
            ):
                with self.assertRaisesRegex(ValueError, name):
                    env_bool(name, False)
```

- [ ] **Step 6: Run the tokenizer tests and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_config.ChunkTokenizerTests -v
```

Expected: import failure because `resolve_chunk_tokenizer` does not exist.

- [ ] **Step 7: Implement tokenizer derivation and migrate all boolean constants**

Add the pure resolver and replace every `.lower() == "true"` configuration expression:

```python
DEFAULT_CHUNK_TOKENIZER = "sentence-transformers/all-MiniLM-L6-v2"
ZH_CHUNK_TOKENIZER = "BAAI/bge-large-zh-v1.5"


def resolve_chunk_tokenizer(use_zh: bool, configured: str | None) -> str:
    if use_zh:
        return ZH_CHUNK_TOKENIZER
    return configured or DEFAULT_CHUNK_TOKENIZER


USE_ZH_TOKENIZER = env_bool("USE_ZH_TOKENIZER", False)
ENABLE_RERANKER = env_bool("ENABLE_RERANKER", True)
DOCLING_CHUNK_TOKENIZER = resolve_chunk_tokenizer(
    USE_ZH_TOKENIZER,
    os.getenv("DOCLING_CHUNK_TOKENIZER"),
)
CHAT_ENABLE_QUERY_REWRITE = env_bool("CHAT_ENABLE_QUERY_REWRITE", True)
ENABLE_SMART_CHUNKER = env_bool("ENABLE_SMART_CHUNKER", False)
ENABLE_MULTI_STAGE_RETRIEVAL = env_bool("ENABLE_MULTI_STAGE_RETRIEVAL", False)
ENABLE_METADATA_FILTER = env_bool("ENABLE_METADATA_FILTER", False)
ENABLE_FINANCIAL_EXPERT_PROMPT = env_bool(
    "ENABLE_FINANCIAL_EXPERT_PROMPT", False
)
ENABLE_CITATION_VALIDATION = env_bool("ENABLE_CITATION_VALIDATION", False)
```

Define each constant once; remove the earlier duplicate read of `USE_ZH_TOKENIZER`.

- [ ] **Step 8: Update `.env.example` to match runtime behavior**

Document that the file must be copied to `.env`, add `ENABLE_RERANKER=true`, make `USE_ZH_TOKENIZER=false` an active explicit default, and keep optional optimization defaults aligned with `config.py`:

```dotenv
# Copy this file to .env. This template is not loaded directly.
# Priority: process environment > .env > code defaults.

# Global BGE reranker gate. Requests may disable reranking but cannot bypass false.
ENABLE_RERANKER=true

# true forces BAAI/bge-large-zh-v1.5 and ignores DOCLING_CHUNK_TOKENIZER.
USE_ZH_TOKENIZER=false
DOCLING_CHUNK_TOKENIZER=sentence-transformers/all-MiniLM-L6-v2

ENABLE_SMART_CHUNKER=false
ENABLE_MULTI_STAGE_RETRIEVAL=false
ENABLE_METADATA_FILTER=false
ENABLE_FINANCIAL_EXPERT_PROMPT=false
ENABLE_CITATION_VALIDATION=false
```

Keep existing non-boolean settings and redacted/empty key placeholders intact.

- [ ] **Step 9: Run Task 1 tests and verify GREEN**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_config -v
```

Expected: all tests pass without network or model downloads.

- [ ] **Step 10: Commit only Task 1 hunks**

Inspect and interactively stage only the changes from this task so pre-existing edits remain untouched:

```powershell
git diff -- finance_rag/config.py .env.example tests/test_config.py
git add -p -- finance_rag/config.py .env.example
git add -- tests/test_config.py
git diff --cached --check
git commit -m "fix: centralize rag environment configuration"
```

### Task 2: Enforce the global BGE reranker gate at the chat boundary

**Files:**
- Create: `tests/test_chat_reranker.py`
- Modify: `finance_rag/chat.py:17-228`

**Interfaces:**
- Consumes: `finance_rag.config.ENABLE_RERANKER: bool`
- Produces: `reranking_enabled(requested: bool) -> bool`
- Preserves: `chat_stream(...)` and `retrieve_and_rerank(...)` public signatures.

- [ ] **Step 1: Write failing unit tests for the effective decision**

Create `tests/test_chat_reranker.py`:

```python
from __future__ import annotations

import unittest
from unittest.mock import patch

import finance_rag.chat as chat_module


class EffectiveRerankerTests(unittest.TestCase):
    def test_global_disable_cannot_be_overridden_by_request(self):
        with patch.object(chat_module, "ENABLE_RERANKER", False):
            self.assertFalse(chat_module.reranking_enabled(True))

    def test_global_enable_still_allows_request_to_disable(self):
        with patch.object(chat_module, "ENABLE_RERANKER", True):
            self.assertFalse(chat_module.reranking_enabled(False))

    def test_both_global_and_request_must_enable_reranking(self):
        with patch.object(chat_module, "ENABLE_RERANKER", True):
            self.assertTrue(chat_module.reranking_enabled(True))
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_chat_reranker.EffectiveRerankerTests -v
```

Expected: errors because `reranking_enabled` and the imported `ENABLE_RERANKER` symbol are absent.

- [ ] **Step 3: Implement the effective decision and use it in synchronous retrieval**

Import `ENABLE_RERANKER` in `finance_rag/chat.py`, add:

```python
def reranking_enabled(requested: bool) -> bool:
    return ENABLE_RERANKER and requested
```

Change `retrieve_and_rerank` to pass `reranking_enabled(use_rerank)` into `kb.hybrid_search`.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_chat_reranker.EffectiveRerankerTests -v
```

Expected: all three tests pass.

- [ ] **Step 5: Write failing async tests for SSE and both retrieval branches**

Extend `tests/test_chat_reranker.py` with a real async-generator traversal and a minimal knowledge-base fake:

```python
class FakeKnowledgeBase:
    def __init__(self):
        self.search_kwargs = None

    def hybrid_search(self, query, **kwargs):
        self.search_kwargs = kwargs
        return []


class ChatStreamRerankerTests(unittest.IsolatedAsyncioTestCase):
    async def collect_events(self, *, globally_enabled, requested):
        kb = FakeKnowledgeBase()
        with (
            patch.object(chat_module, "ENABLE_RERANKER", globally_enabled),
            patch.object(chat_module, "model", object()),
            patch.object(chat_module, "get_knowledge_base", return_value=kb),
        ):
            events = [
                event
                async for event in chat_module.chat_stream(
                    "query",
                    use_rewrite=False,
                    use_rerank=requested,
                )
            ]
        return kb, events

    async def test_global_disable_suppresses_status_and_normal_retrieval_rerank(self):
        kb, events = await self.collect_events(
            globally_enabled=False, requested=True
        )
        self.assertNotIn("reranking", [event.get("stage") for event in events])
        self.assertFalse(kb.search_kwargs["use_rerank"])

    async def test_both_enabled_emit_status_and_enable_normal_retrieval_rerank(self):
        kb, events = await self.collect_events(
            globally_enabled=True, requested=True
        )
        self.assertIn("reranking", [event.get("stage") for event in events])
        self.assertTrue(kb.search_kwargs["use_rerank"])

    async def test_request_disable_suppresses_status_and_normal_retrieval_rerank(self):
        kb, events = await self.collect_events(
            globally_enabled=True, requested=False
        )
        self.assertNotIn("reranking", [event.get("stage") for event in events])
        self.assertFalse(kb.search_kwargs["use_rerank"])
```

Also add the multi-stage fake and test before changing production code:

```python
class FakePipeline:
    last_kwargs = None

    def __init__(self, **kwargs):
        pass

    def retrieve(self, query, **kwargs):
        type(self).last_kwargs = kwargs
        return []


class MultiStageRerankerTests(unittest.IsolatedAsyncioTestCase):
    async def test_global_disable_reaches_multi_stage_pipeline(self):
        kb = FakeKnowledgeBase()
        kb._get_retriever = lambda: object()
        with (
            patch.object(chat_module, "ENABLE_RERANKER", False),
            patch.object(chat_module, "model", object()),
            patch.object(chat_module, "get_knowledge_base", return_value=kb),
            patch("finance_rag.retrieve.AdvancedRetrievalPipeline", FakePipeline),
        ):
            events = [
                event
                async for event in chat_module.chat_stream(
                    "query",
                    use_rewrite=False,
                    use_rerank=True,
                    strategy="optimized",
                )
            ]
        self.assertFalse(FakePipeline.last_kwargs["use_rerank"])
        self.assertNotIn("reranking", [event.get("stage") for event in events])
```

The empty search results intentionally end the generators before LLM generation, so the tests exercise application flow without invoking an external model.

- [ ] **Step 6: Run async tests and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_chat_reranker.ChatStreamRerankerTests -v
```

Expected: the global-disable cases still emit `reranking` and pass `use_rerank=True` to both retrieval branches.

- [ ] **Step 7: Compute the value once in `chat_stream` and propagate it**

At the start of retrieval in `chat_stream`, derive:

```python
effective_use_rerank = reranking_enabled(use_rerank)
```

Use `effective_use_rerank` for all three consumers:

```python
if effective_use_rerank:
    yield {"type": "status", "stage": "reranking"}

# AdvancedRetrievalPipeline.retrieve(...)
use_rerank=effective_use_rerank

# kb.hybrid_search(...)
use_rerank=effective_use_rerank
```

- [ ] **Step 8: Run async tests and verify GREEN**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_chat_reranker -v
```

Expected: all effective-decision and normal retrieval tests pass.

- [ ] **Step 9: Verify normal and multi-stage propagation together**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_chat_reranker.MultiStageRerankerTests -v
```

Expected: PASS, proving both retrieval branches consume the same effective value after both tests were observed failing in Step 6.

- [ ] **Step 10: Commit only Task 2 hunks**

```powershell
git diff -- finance_rag/chat.py tests/test_chat_reranker.py
git add -p -- finance_rag/chat.py
git add -- tests/test_chat_reranker.py
git diff --cached --check
git commit -m "fix: enforce global bge reranker switch"
```

### Task 3: Regression and runtime configuration verification

**Files:**
- Verify only; no production files are added by this task.

**Interfaces:**
- Consumes: all configuration and chat behavior delivered by Tasks 1 and 2.
- Produces: verification evidence for the completed change.

- [ ] **Step 1: Run all targeted tests together**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_config tests.test_chat_reranker -v
```

Expected: all tests pass with no network requests or model downloads.

- [ ] **Step 2: Run the existing regression suite**

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Expected: the existing ablation suite and new configuration suites all pass.

- [ ] **Step 3: Verify actual defaults in a clean subprocess**

Run with the relevant process variables removed and from a directory other than the repository root:

```powershell
$names = @(
  'USE_ZH_TOKENIZER', 'DOCLING_CHUNK_TOKENIZER', 'ENABLE_RERANKER',
  'ENABLE_SMART_CHUNKER', 'ENABLE_MULTI_STAGE_RETRIEVAL',
  'ENABLE_METADATA_FILTER', 'ENABLE_FINANCIAL_EXPERT_PROMPT',
  'ENABLE_CITATION_VALIDATION'
)
foreach ($name in $names) { Remove-Item "Env:$name" -ErrorAction SilentlyContinue }
Push-Location $env:TEMP
& 'F:\python project\finance-rag-retrieval\.venv\Scripts\python.exe' -c "import sys; sys.path.insert(0, r'F:\python project\finance-rag-retrieval'); from finance_rag.config import ENABLE_RERANKER, USE_ZH_TOKENIZER, DOCLING_CHUNK_TOKENIZER; print(ENABLE_RERANKER, USE_ZH_TOKENIZER, DOCLING_CHUNK_TOKENIZER)"
Pop-Location
```

Expected output when the repository has no `.env`:

```text
True False sentence-transformers/all-MiniLM-L6-v2
```

If a real repository `.env` exists during execution, compare the output with that file and the process environment using the documented precedence instead of expecting defaults.

- [ ] **Step 4: Check diff hygiene and working-tree separation**

```powershell
git diff --check
git status --short
git diff -- finance_rag/config.py finance_rag/chat.py .env.example tests/test_config.py tests/test_chat_reranker.py
```

Expected: no whitespace errors; only intended task hunks are staged/committed, and unrelated pre-existing modifications remain preserved.
