# Finance RAG Project Structure Engineering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate the Finance RAG repository to an installable `src/` layout with explicit API, application, domain, infrastructure, evaluation, data, and runtime boundaries while preserving every existing external contract.

**Architecture:** Introduce packaging and path abstractions first, then migrate behavior behind compatibility modules in small vertical slices. Keep `finance_rag.*`, HTTP routes, environment variables, Milvus collections, report schemas, and legacy commands stable while implementation moves into focused packages.

**Tech Stack:** Python 3.11+, FastAPI, Pydantic 2, LangChain, Docling, Milvus, Ragas, pytest, ruff, mypy, Vue 3, Vite.

## Global Constraints

- Preserve all existing HTTP routes and response formats.
- Preserve all environment variable names and Milvus collection names.
- Preserve benchmark JSON, CSV, and Markdown report schemas.
- Preserve the public `finance_rag.*` import paths.
- Preserve `python main.py`, both existing benchmark scripts, and `cd frontend; npm run dev`.
- Add `python -m finance_rag`, `finance-rag-server`, and `finance-rag-ablation`.
- Move historical `artifacts/` into `var/artifacts/` without deleting results.
- Split `files/` into `data/source/` and `data/evaluation/`.
- Remove `libs/` only after proving it contains no project-owned code or runtime-only dependency.
- Do not change RAG algorithms, model selection, retrieval weights, Milvus schema, or frontend behavior.
- Do not add Docker Compose, CI configuration, cloud deployment, or a frontend redesign.
- Keep runtime directories and secrets out of Git.
- Run a failing test before each behavior change and keep legacy compatibility tests green after every migration.

---

## File Structure

### Repository and packaging

- Create `pyproject.toml`: build metadata, dependency groups, console scripts, pytest/ruff/mypy configuration.
- Create `.gitignore`: Python, Node, runtime, model-cache, IDE, and secret exclusions.
- Create `.env.example`: every supported environment variable with safe example values.
- Create `README.md`: setup, architecture, development, benchmark, and compatibility commands.
- Keep `main.py`: thin compatibility launcher only.

### Backend package

- Move package root to `src/finance_rag/`.
- Create `src/finance_rag/__main__.py`: standard server entry.
- Create `src/finance_rag/api/`: app factory, dependencies, schemas, and route modules.
- Create `src/finance_rag/application/`: chat, document, and evaluation orchestration.
- Create `src/finance_rag/domain/`: dependency-free models, retrieval rules, evaluation models, and exceptions.
- Create `src/finance_rag/infrastructure/`: settings, logging, LLM, Milvus, and parsing adapters.
- Create `src/finance_rag/evaluation/`: datasets, metrics, ablation runner, and reporting.
- Retain old module filenames under `src/finance_rag/` as re-exporting compatibility modules.

### Data, runtime, tests, and docs

- Create `data/source/` and move knowledge documents there.
- Create `data/evaluation/` and move `evaluation_qa.md` there.
- Create `var/artifacts/` and move the complete historical artifact tree there.
- Create `var/logs/` and move existing logs there.
- Reorganize tests into `tests/unit/`, `tests/integration/`, and `tests/fixtures/`.
- Create `docs/architecture/overview.md` and `docs/operations/benchmark.md`.

---

### Task 1: Establish Packaging, Ignore Rules, and Standard Entry Points

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `.env.example`
- Create: `README.md`
- Move: `finance_rag/` to `src/finance_rag/` without changing module contents
- Create: `src/finance_rag/__main__.py`
- Modify: `main.py`
- Test: `tests/unit/test_packaging.py`

**Interfaces:**
- Produces: `finance_rag.__main__.main() -> None`.
- Produces console scripts `finance-rag-server` and `finance-rag-ablation`.
- Preserves `python main.py`.

- [ ] **Step 1: Write failing packaging tests**

```python
from pathlib import Path
import tomllib


def test_pyproject_declares_src_layout_and_scripts():
    config = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert config["tool"]["setuptools"]["package-dir"] == {"": "src"}
    assert config["project"]["scripts"]["finance-rag-server"] == "finance_rag.__main__:main"
    assert config["project"]["scripts"]["finance-rag-ablation"] == "finance_rag.evaluation.cli:main"


def test_runtime_and_secret_paths_are_ignored():
    rules = Path(".gitignore").read_text(encoding="utf-8").splitlines()
    for required in (".env", ".venv/", "frontend/node_modules/", "frontend/dist/", "var/"):
        assert required in rules
```

- [ ] **Step 2: Run tests and confirm missing-file failures**

Run: `\.venv\Scripts\python.exe -m pytest tests/unit/test_packaging.py -v`

Expected: FAIL because `pyproject.toml` and `.gitignore` do not exist.

- [ ] **Step 3: Move the existing package intact**

Create `src/`, then move the complete current `finance_rag/` directory to
`src/finance_rag/`. Do not split or edit modules in this step. Confirm the move
preserved every tracked file:

```powershell
git diff --summary
git status --short
```

Expected: every former `finance_rag/*` file is represented under
`src/finance_rag/*`; there are no deletions without a corresponding destination.

- [ ] **Step 4: Add packaging metadata and scripts**

Create `pyproject.toml` with this minimum structure, copying the current pinned and ranged dependencies from `requirements.txt` into `project.dependencies`:

```toml
[build-system]
requires = ["setuptools>=75", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "finance-rag"
version = "2.0.0"
requires-python = ">=3.11"
dependencies = [
  "python-dotenv~=1.2.2",
  "numpy~=2.3.4",
  "langchain~=1.2.10",
  "langchain-core~=1.2.17",
  "langchain-community~=0.4.1",
  "dashscope",
  "pymilvus>=2.4.0",
  "docling>=2,<3",
  "langchain-deepseek",
  "fastapi>=0.115.0",
  "uvicorn[standard]>=0.30.0",
  "pydantic>=2.0.0",
  "sentence-transformers>=2.7.0",
  "python-multipart>=0.0.9",
  "sse-starlette>=2.1.0",
  "ragas==0.4.3",
  "datasets>=4.0.0",
]

[project.optional-dependencies]
dev = ["pytest>=8", "ruff>=0.12", "mypy>=1.16"]

[project.scripts]
finance-rag-server = "finance_rag.__main__:main"
finance-rag-ablation = "finance_rag.evaluation.cli:main"

[tool.setuptools]
package-dir = {"" = "src"}

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
markers = ["integration: requires local services or external model APIs"]

[tool.ruff]
target-version = "py311"
line-length = 100

[tool.mypy]
python_version = "3.11"
packages = ["finance_rag"]
```

Create `src/finance_rag/__main__.py`:

```python
from finance_rag.api import app
import uvicorn


def main() -> None:
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
```

Change `main.py` to import and call `finance_rag.__main__.main` while retaining frontend mounting compatibility until Task 8 moves it into the app factory.

- [ ] **Step 5: Add safe repository documentation files**

Create `.gitignore` containing at least:

```gitignore
.env
.venv/
__pycache__/
*.py[cod]
.pytest_cache/
.mypy_cache/
.ruff_cache/
.vscode/
frontend/node_modules/
frontend/dist/
var/
*.log
```

Create `.env.example` with the exact names currently consumed by `finance_rag/config.py`, using blank API keys and existing safe defaults. Create a README quickstart for editable install, server start, frontend start, tests, and benchmark execution.

- [ ] **Step 6: Run packaging tests**

Run: `\.venv\Scripts\python.exe -m pytest tests/unit/test_packaging.py -v`

Expected: PASS.

- [ ] **Step 7: Commit Task 1**

```powershell
git add pyproject.toml .gitignore .env.example README.md main.py finance_rag src/finance_rag tests/unit/test_packaging.py
git commit -m "build: establish src package layout"
```

### Task 2: Centralize Project Paths and Preserve Legacy Fallbacks

**Files:**
- Create: `src/finance_rag/infrastructure/__init__.py`
- Create: `src/finance_rag/infrastructure/paths.py`
- Test: `tests/unit/infrastructure/test_paths.py`

**Interfaces:**
- Produces immutable `ProjectPaths` with `root`, `source_data`, `evaluation_data`, `artifacts`, `logs`, and `frontend_dist`.
- Produces `get_project_paths(root: Path | None = None) -> ProjectPaths`.
- Produces `resolve_existing(preferred: Path, legacy: Path) -> Path`.

- [ ] **Step 1: Write failing path-resolution tests**

```python
from pathlib import Path
from finance_rag.infrastructure.paths import get_project_paths, resolve_existing


def test_project_paths_use_engineered_layout(tmp_path: Path):
    paths = get_project_paths(tmp_path)
    assert paths.source_data == tmp_path / "data" / "source"
    assert paths.evaluation_data == tmp_path / "data" / "evaluation"
    assert paths.artifacts == tmp_path / "var" / "artifacts"
    assert paths.logs == tmp_path / "var" / "logs"
    assert paths.frontend_dist == tmp_path / "frontend" / "dist"


def test_legacy_path_is_used_only_when_preferred_is_missing(tmp_path: Path):
    preferred = tmp_path / "data" / "source"
    legacy = tmp_path / "files"
    legacy.mkdir()
    assert resolve_existing(preferred, legacy) == legacy
    preferred.mkdir(parents=True)
    assert resolve_existing(preferred, legacy) == preferred
```

- [ ] **Step 2: Run the tests and confirm the import failure**

Run: `\.venv\Scripts\python.exe -m pytest tests/unit/infrastructure/test_paths.py -v`

Expected: FAIL because `finance_rag.infrastructure.paths` does not exist.

- [ ] **Step 3: Implement the immutable path object**

```python
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    source_data: Path
    evaluation_data: Path
    artifacts: Path
    logs: Path
    frontend_dist: Path


def get_project_paths(root: Path | None = None) -> ProjectPaths:
    project_root = (root or Path(__file__).resolve().parents[3]).resolve()
    return ProjectPaths(
        root=project_root,
        source_data=project_root / "data" / "source",
        evaluation_data=project_root / "data" / "evaluation",
        artifacts=project_root / "var" / "artifacts",
        logs=project_root / "var" / "logs",
        frontend_dist=project_root / "frontend" / "dist",
    )


def resolve_existing(preferred: Path, legacy: Path) -> Path:
    return preferred if preferred.exists() or not legacy.exists() else legacy
```

- [ ] **Step 4: Run path tests**

Run: `\.venv\Scripts\python.exe -m pytest tests/unit/infrastructure/test_paths.py -v`

Expected: PASS.

- [ ] **Step 5: Commit Task 2**

```powershell
git add src/finance_rag/infrastructure tests/unit/infrastructure/test_paths.py
git commit -m "feat: centralize project paths"
```

### Task 3: Move Data and Runtime Trees Without Losing History

**Files:**
- Move: `files/docs/evaluation_qa.md` to `data/evaluation/evaluation_qa.md`
- Move: remaining `files/*` to `data/source/`
- Move: `artifacts/*` to `var/artifacts/`
- Move: `logs/*` to `var/logs/`
- Modify: `src/finance_rag/infrastructure/paths.py`
- Test: `tests/unit/infrastructure/test_paths.py`
- Test: `tests/integration/test_data_layout.py`

**Interfaces:**
- Adds `ProjectPaths.test_set_file: Path`.
- Adds `ProjectPaths.ensure_runtime_dirs() -> None`, creating only `var/artifacts` and `var/logs`.
- Preserves legacy read fallback for one migration window.

- [ ] **Step 1: Add failing data-layout tests**

```python
from finance_rag.infrastructure.paths import get_project_paths


def test_repository_data_uses_new_layout():
    paths = get_project_paths()
    assert paths.test_set_file == paths.evaluation_data / "evaluation_qa.md"
    assert paths.test_set_file.exists()
    assert any(paths.source_data.iterdir())


def test_runtime_directory_creation_does_not_create_source_data(tmp_path):
    paths = get_project_paths(tmp_path)
    paths.ensure_runtime_dirs()
    assert paths.artifacts.is_dir()
    assert paths.logs.is_dir()
    assert not paths.source_data.exists()
```

- [ ] **Step 2: Run tests and confirm missing new-layout failures**

Run: `\.venv\Scripts\python.exe -m pytest tests/unit/infrastructure/test_paths.py tests/integration/test_data_layout.py -v`

Expected: FAIL because `test_set_file`, `ensure_runtime_dirs`, and new data paths do not exist.

- [ ] **Step 3: Extend `ProjectPaths` and move the trees**

Implement:

```python
@property
def test_set_file(self) -> Path:
    preferred = self.evaluation_data / "evaluation_qa.md"
    legacy = self.root / "files" / "docs" / "evaluation_qa.md"
    return resolve_existing(preferred, legacy)

def ensure_runtime_dirs(self) -> None:
    self.artifacts.mkdir(parents=True, exist_ok=True)
    self.logs.mkdir(parents=True, exist_ok=True)
```

Move files with native PowerShell `Move-Item -LiteralPath` commands after resolving and verifying every source and destination remains under the project root. Preserve all historical artifact subdirectories unchanged.

- [ ] **Step 4: Verify file counts and benchmark hashes**

Before moving, record relative file lists, sizes, and SHA-256 hashes under `artifacts/`. After moving, compare them to `var/artifacts/` and require exact equality.

Run the data-layout tests again and confirm PASS.

- [ ] **Step 5: Commit Task 3 without runtime artifacts**

```powershell
git add src/finance_rag/infrastructure/paths.py tests data
git commit -m "refactor: separate data and runtime paths"
```

`var/` remains ignored and is not staged.

### Task 4: Remove Vendored Dependencies Safely

**Files:**
- Remove: `libs/`
- Modify: `pyproject.toml`
- Test: `tests/unit/test_dependency_hygiene.py`

**Interfaces:**
- Guarantees imports resolve from the active environment, never repository `libs/`.
- Produces no runtime API.

- [ ] **Step 1: Write failing dependency-hygiene tests**

```python
from pathlib import Path


def test_repository_does_not_vendor_site_packages():
    assert not Path("libs").exists()


def test_source_does_not_add_libs_to_sys_path():
    offenders = []
    for path in Path("src").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "sys.path" in text and "libs" in text:
            offenders.append(str(path))
    assert offenders == []
```

- [ ] **Step 2: Run the test and confirm `libs/` causes failure**

Run: `\.venv\Scripts\python.exe -m pytest tests/unit/test_dependency_hygiene.py -v`

Expected: FAIL because `libs/` exists.

- [ ] **Step 3: Audit `libs/` before deletion**

Compare top-level packages under `libs/` with installed distributions and search all project source for `sys.path`, `PYTHONPATH`, and direct `libs` references. Require zero project-only `.py` modules and zero active references. If any project-owned module is found, stop and move it into `src/finance_rag` in a separate reviewed task rather than deleting it.

- [ ] **Step 4: Remove `libs/` and verify environment imports**

Remove only the resolved absolute `libs/` directory under this project. Run:

```powershell
.\.venv\Scripts\python.exe -c "import docling, pymilvus, ragas, sentence_transformers, finance_rag"
```

Expected: exit 0.

- [ ] **Step 5: Run hygiene and existing tests**

Run: `\.venv\Scripts\python.exe -m pytest tests/unit/test_dependency_hygiene.py tests/test_ablation.py -v`

Expected: PASS.

- [ ] **Step 6: Commit Task 4**

```powershell
git add -A libs pyproject.toml tests/unit/test_dependency_hygiene.py
git commit -m "build: remove vendored dependencies"
```

### Task 5: Migrate Evaluation Models, Datasets, Metrics, and Reporting

**Files:**
- Create: `src/finance_rag/domain/evaluation.py`
- Create: `src/finance_rag/evaluation/__init__.py`
- Create: `src/finance_rag/evaluation/datasets.py`
- Create: `src/finance_rag/evaluation/metrics.py`
- Create: `src/finance_rag/evaluation/reporting.py`
- Create: `src/finance_rag/evaluation/ablation.py`
- Create compatibility modules: `src/finance_rag/evaluation.py`, `src/finance_rag/benchmark.py`, `src/finance_rag/ablation.py`
- Move and split tests into: `tests/unit/evaluation/`

**Interfaces:**
- `domain.evaluation.TestSetEntry`, `AblationStrategy`, and `Experiment` are dependency-free dataclasses.
- `evaluation.datasets.TestSetLoader` and `select_dataset_entries` preserve existing behavior.
- `evaluation.metrics` preserves metric names and missing-value rules.
- `evaluation.reporting.write_ablation_reports` preserves report schemas and writes atomically.
- Compatibility modules re-export every currently imported public symbol.

- [ ] **Step 1: Write failing public-compatibility tests**

```python
def test_legacy_evaluation_imports_resolve_to_new_modules():
    from finance_rag.ablation import select_experiments as legacy
    from finance_rag.evaluation.ablation import select_experiments as current
    assert legacy is current


def test_legacy_loader_import_resolves_to_dataset_module():
    from finance_rag.evaluation import TestSetLoader as legacy
    from finance_rag.evaluation.datasets import TestSetLoader as current
    assert legacy is current
```

- [ ] **Step 2: Run compatibility tests and confirm missing-module failure**

Run: `\.venv\Scripts\python.exe -m pytest tests/unit/evaluation/test_compatibility.py -v`

Expected: FAIL because the new packages do not exist.

- [ ] **Step 3: Move dependency-free dataclasses first**

Move `TestSetEntry`, `AblationStrategy`, and `Experiment` to `domain/evaluation.py`. Update their original modules to import and re-export the classes. Run focused loader and experiment tests after this move.

- [ ] **Step 4: Split datasets and metrics**

Move loader/profile logic into `evaluation/datasets.py`. Move pure retrieval, paired-summary, change, and missing-rate logic into `evaluation/metrics.py`. Keep model/LLM-backed quality evaluation behind the existing infrastructure boundary until Task 6.

- [ ] **Step 5: Make report writes atomic**

Implement a private helper in `reporting.py`:

```python
def _atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding=encoding)
    temporary.replace(path)
```

Build every report payload in memory first. Write four temporary files, then replace their final paths only after all serialization succeeds. Add a test that forces report rendering to raise and asserts no final output file is replaced.

- [ ] **Step 6: Move runner and add thin compatibility modules**

Move experiment selection, runner orchestration, recursive chunking comparison, and metadata into `evaluation/ablation.py`. Legacy modules contain imports and explicit `__all__` only; they must not duplicate implementation.

- [ ] **Step 7: Run evaluation tests**

Run: `\.venv\Scripts\python.exe -m pytest tests/unit/evaluation tests/test_ablation.py -v`

Expected: PASS with all previous 16 ablation tests plus new compatibility and atomic-write tests.

- [ ] **Step 8: Commit Task 5**

```powershell
git add src/finance_rag/domain src/finance_rag/evaluation src/finance_rag/ablation.py src/finance_rag/benchmark.py tests
git commit -m "refactor: modularize evaluation pipeline"
```

### Task 6: Isolate Configuration, Logging, Parsing, LLM, and Milvus Adapters

**Files:**
- Create: `src/finance_rag/domain/exceptions.py`
- Create: `src/finance_rag/infrastructure/config.py`
- Create: `src/finance_rag/infrastructure/logging.py`
- Create: `src/finance_rag/infrastructure/llm/client.py`
- Create: `src/finance_rag/infrastructure/milvus/knowledge_base.py`
- Create: `src/finance_rag/infrastructure/parsing/docling.py`
- Create compatibility modules: `src/finance_rag/config.py`, `logging_config.py`, `knowledge_base.py`, `chunking.py`
- Test: `tests/unit/infrastructure/`
- Test: `tests/integration/infrastructure/`

**Interfaces:**
- `Settings.from_env() -> Settings` preserves current environment names and defaults.
- `ExternalServiceError(stage: str, service: str, recoverable: bool, message: str)` hides secrets.
- `get_knowledge_base()` preserves its current singleton behavior and collection defaults.
- Compatibility modules preserve public symbols.

- [ ] **Step 1: Write failing settings and exception tests**

```python
def test_settings_preserve_existing_defaults(monkeypatch):
    monkeypatch.delenv("KB_COLLECTION_NAME", raising=False)
    monkeypatch.delenv("MILVUS_URI", raising=False)
    from finance_rag.infrastructure.config import Settings
    settings = Settings.from_env()
    assert settings.kb_collection_name == "finance_kb"
    assert settings.milvus_uri == "http://localhost:19530"
    assert settings.hybrid_dense_weight == 0.7
    assert settings.hybrid_sparse_weight == 0.3


def test_external_error_never_exposes_secret():
    from finance_rag.domain.exceptions import ExternalServiceError
    error = ExternalServiceError("generation", "deepseek", True, "request failed")
    assert str(error) == "deepseek generation failed: request failed"
```

- [ ] **Step 2: Run tests and confirm missing-module failures**

Run: `\.venv\Scripts\python.exe -m pytest tests/unit/infrastructure -v`

- [ ] **Step 3: Implement `Settings` without import-time client creation**

Move environment parsing from the current `config.py` into a frozen dataclass. Keep `safe_parse_json` as a pure helper. Replace the global eager model with `get_chat_model(settings: Settings)` cached by explicit factory call.

- [ ] **Step 4: Move adapters one at a time**

Move logging, chunking, and Milvus implementations while retaining existing tests after each move. Translate SDK exceptions at adapter boundaries into `ExternalServiceError`, preserving original exception chaining with `raise ... from exc`.

- [ ] **Step 5: Run unit and integration import tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/infrastructure -v
.\.venv\Scripts\python.exe -m pytest tests/integration/infrastructure -v -m integration
```

Integration tests may require Milvus but must not call external model APIs unless an explicit marker and credentials are present.

- [ ] **Step 6: Commit Task 6**

```powershell
git add src/finance_rag/domain src/finance_rag/infrastructure src/finance_rag/config.py src/finance_rag/logging_config.py src/finance_rag/knowledge_base.py src/finance_rag/chunking.py tests
git commit -m "refactor: isolate infrastructure adapters"
```

### Task 7: Introduce Application Services and Preserve Online Behavior

**Files:**
- Create: `src/finance_rag/application/chat_service.py`
- Create: `src/finance_rag/application/document_service.py`
- Create: `src/finance_rag/application/evaluation_service.py`
- Create: `src/finance_rag/domain/models.py`
- Create: `src/finance_rag/domain/retrieval.py`
- Modify compatibility modules: `chat.py`, `document_manager.py`, `retrieve.py`
- Test: `tests/unit/application/`

**Interfaces:**
- `ChatService.chat(query, history, options) -> ChatResult`.
- `ChatService.stream(...) -> AsyncIterator[ChatEvent]`.
- `DocumentService.add/list/delete` returns domain models, not FastAPI schemas.
- `EvaluationService.evaluate_strategy(...)` returns the existing metric payload shape.

- [ ] **Step 1: Write failing service tests with fake ports**

```python
async def test_chat_service_orchestrates_rewrite_retrieve_generate():
    rewriter = FakeRewriter("rewritten")
    retriever = FakeRetriever([Context(content="ctx", source="doc.md")])
    generator = FakeGenerator("answer")
    service = ChatService(rewriter, retriever, generator)
    result = await service.chat("question", [], ChatOptions())
    assert result.answer == "answer"
    assert result.rewritten_query == "rewritten"
    assert result.sources[0].source == "doc.md"
```

Implement test fakes inside `tests/unit/application/fakes.py`; do not add test-only methods to production classes.

- [ ] **Step 2: Run tests and confirm missing service failures**

Run: `\.venv\Scripts\python.exe -m pytest tests/unit/application -v`

- [ ] **Step 3: Define domain input/output models and protocols**

Use frozen dataclasses for `ChatOptions`, `Context`, `ChatResult`, `ChatEvent`, `DocumentInfo`, and evaluation results. Define `Protocol` interfaces for rewrite, retrieval, generation, document storage, and evaluation dependencies.

- [ ] **Step 4: Implement services by moving orchestration only**

Move orchestration from `chat.py`, `document_manager.py`, and API evaluation handlers. Keep algorithms in domain/evaluation and external calls in infrastructure. Compatibility functions obtain default services from factories and preserve current signatures.

- [ ] **Step 5: Run application and legacy tests**

Run: `\.venv\Scripts\python.exe -m pytest tests/unit/application tests/test_ablation.py -v`

Expected: PASS.

- [ ] **Step 6: Commit Task 7**

```powershell
git add src/finance_rag/application src/finance_rag/domain src/finance_rag/chat.py src/finance_rag/document_manager.py src/finance_rag/retrieve.py tests
git commit -m "refactor: introduce application services"
```

### Task 8: Split FastAPI App, Schemas, Dependencies, and Routes

**Files:**
- Create: `src/finance_rag/api/app.py`
- Create: `src/finance_rag/api/dependencies.py`
- Create: `src/finance_rag/api/errors.py`
- Create: `src/finance_rag/api/schemas/chat.py`
- Create: `src/finance_rag/api/schemas/documents.py`
- Create: `src/finance_rag/api/schemas/evaluation.py`
- Create: `src/finance_rag/api/routes/chat.py`
- Create: `src/finance_rag/api/routes/documents.py`
- Create: `src/finance_rag/api/routes/evaluation.py`
- Create: `src/finance_rag/api/routes/health.py`
- Replace: `src/finance_rag/api.py` with compatibility re-export
- Test: `tests/integration/api/`

**Interfaces:**
- `create_app() -> FastAPI`.
- `app = create_app()` remains available from `finance_rag.api`.
- Every existing route, method, status, response field, and SSE event name remains unchanged.

- [ ] **Step 1: Capture the current route contract in failing tests**

```python
def test_route_contract_is_preserved():
    from finance_rag.api import app
    actual = {(route.path, tuple(sorted(route.methods or []))) for route in app.routes}
    required = {
        ("/", ("GET",)),
        ("/api/health", ("GET",)),
        ("/api/chat", ("POST",)),
        ("/api/chat/stream", ("POST",)),
        ("/api/documents/upload", ("POST",)),
        ("/api/documents", ("GET",)),
        ("/api/documents/{source:path}", ("DELETE",)),
        ("/api/kb/stats", ("GET",)),
        ("/api/test-queries", ("GET",)),
        ("/api/evaluate-strategy", ("POST",)),
    }
    assert required <= actual


def test_app_factory_does_not_open_milvus_on_import(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("Milvus must be lazy")

    monkeypatch.setattr("finance_rag.infrastructure.milvus.knowledge_base.get_knowledge_base", fail)
    from finance_rag.api.app import create_app
    create_app()
```

- [ ] **Step 2: Run tests and confirm missing app-factory failure**

Run: `\.venv\Scripts\python.exe -m pytest tests/integration/api -v`

- [ ] **Step 3: Move schemas without changing field definitions**

Copy each current Pydantic model into the matching schema module. Preserve defaults, constraints, descriptions, and response models exactly. Add schema serialization snapshot tests using literal dictionaries.

- [ ] **Step 4: Move routes by capability**

Each route calls an injected application service and maps domain results to schemas. Move exception-to-HTTP mapping to `api/errors.py`. Keep SSE event names and JSON payloads byte-compatible except for nondeterministic timestamps.

- [ ] **Step 5: Add app factory and static frontend mounting**

`create_app()` configures metadata, CORS, exception handlers, routers, and static frontend mounting using `ProjectPaths.frontend_dist`. It must not create Milvus or model clients until a dependency is requested.

- [ ] **Step 6: Run API and frontend build verification**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/integration/api -v
Set-Location frontend
npm run build
```

Expected: tests PASS and Vite build exits 0.

- [ ] **Step 7: Commit Task 8**

```powershell
git add src/finance_rag/api src/finance_rag/api.py main.py tests/integration/api
git commit -m "refactor: modularize FastAPI application"
```

### Task 9: Standardize Evaluation CLI and Legacy Script Wrappers

**Files:**
- Create: `src/finance_rag/evaluation/cli.py`
- Modify: `scripts/run_ablation_benchmark.py`
- Modify: `scripts/run_three_ablation_benchmark.py`
- Test: `tests/unit/evaluation/test_cli.py`

**Interfaces:**
- `finance_rag.evaluation.cli.parse_args(argv: Sequence[str] | None = None) -> Namespace`.
- `finance_rag.evaluation.cli.main(argv: Sequence[str] | None = None) -> int`.
- Legacy scripts call `main()` and contain no benchmark logic.

- [ ] **Step 1: Write failing CLI compatibility tests**

```python
def test_local_precision_defaults_to_chunking_only():
    args = parse_args(["--dataset-profile", "local_precision"])
    assert args.dataset_profile == "local_precision"
    assert args.experiments is None


def test_legacy_script_delegates_to_cli():
    result = subprocess.run(
        [sys.executable, "scripts/run_ablation_benchmark.py", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "local_precision" in result.stdout
    assert "--experiments" in result.stdout
```

- [ ] **Step 2: Run tests and confirm missing CLI module**

Run: `\.venv\Scripts\python.exe -m pytest tests/unit/evaluation/test_cli.py -v`

- [ ] **Step 3: Move parser and main into package CLI**

Accept optional `argv` for testability. Use `ProjectPaths` for defaults. Preserve all existing flags and the local-precision chunking default. Keep the current Milvus validation and report behavior.

- [ ] **Step 4: Replace legacy scripts with wrappers**

```python
from finance_rag.evaluation.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
```

Give the three-experiment wrapper its explicit default experiment list through CLI arguments rather than duplicating runner code.

- [ ] **Step 5: Run CLI tests and help commands**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/evaluation/test_cli.py -v
.\.venv\Scripts\python.exe scripts\run_ablation_benchmark.py --help
.\.venv\Scripts\python.exe -m finance_rag.evaluation.cli --help
```

- [ ] **Step 6: Commit Task 9**

```powershell
git add src/finance_rag/evaluation/cli.py scripts tests/unit/evaluation/test_cli.py
git commit -m "refactor: standardize evaluation CLI"
```

### Task 10: Reorganize the Test Suite and Complete Engineering Documentation

**Files:**
- Move: `tests/test_ablation.py` into focused files under `tests/unit/evaluation/`
- Create: `tests/conftest.py`
- Create: `docs/architecture/overview.md`
- Create: `docs/operations/benchmark.md`
- Modify: `README.md`
- Modify: `requirements.txt`

**Interfaces:**
- Default `pytest` runs unit tests without external services.
- Integration tests require `-m integration` and explicit service prerequisites.
- `requirements.txt` remains as a compatibility export of runtime dependencies, with `pyproject.toml` as source of truth.

- [ ] **Step 1: Add a test collection gate**

Run and save the current test count:

```powershell
.\.venv\Scripts\python.exe -m pytest --collect-only -q
```

Add a repository test that asserts no test module remains directly under `tests/` except `conftest.py`, and that unit tests do not carry the integration marker.

- [ ] **Step 2: Split tests by responsibility**

Move loader/profile tests to `test_datasets.py`, selection/runner tests to `test_ablation.py`, and summary/report tests to `test_reporting.py`. Preserve every assertion and test name. Do not reduce the collected test count.

- [ ] **Step 3: Write architecture and operations docs**

`docs/architecture/overview.md` must include the dependency rule `api -> application -> domain`, with infrastructure implementing domain/application ports. `docs/operations/benchmark.md` must include container startup, collection validation, authorized external data flow, benchmark command, artifact verification, and finally-style Milvus shutdown.

- [ ] **Step 4: Update compatibility dependency file**

Keep `requirements.txt` as a generated-compatible flat runtime list matching `project.dependencies`. Add a header comment that `pyproject.toml` is authoritative. Verify each package spec is present in both representations.

- [ ] **Step 5: Run the complete quality gate**

```powershell
.\.venv\Scripts\python.exe -m pytest -v
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m mypy src\finance_rag
Set-Location frontend
npm run build
```

Expected: every command exits 0. If integration tests require Milvus, run them separately with only `milvus-etcd`, `milvus-minio`, and `milvus-standalone`, then stop those containers and verify port 19530 is closed.

- [ ] **Step 6: Verify compatibility commands**

```powershell
.\.venv\Scripts\python.exe main.py
.\.venv\Scripts\python.exe -m finance_rag
.\.venv\Scripts\python.exe scripts\run_ablation_benchmark.py --help
.\.venv\Scripts\finance-rag-ablation.exe --help
```

For server commands, start each in a bounded subprocess, assert `/api/health` responds, then stop the exact process. Do not kill arbitrary processes holding port 8000.

- [ ] **Step 7: Commit Task 10**

```powershell
git add tests docs README.md requirements.txt
git commit -m "docs: complete engineering migration guide"
```

### Task 11: Final Repository and Migration Audit

**Files:**
- Modify only files required by findings from the checks below.

**Interfaces:**
- Confirms the repository meets the approved design and contains no duplicate implementation or runtime data.

- [ ] **Step 1: Verify repository boundaries and ignored runtime data**

```powershell
git rev-parse --show-toplevel
git status --short --ignored
git check-ignore .env .venv frontend/node_modules var/artifacts var/logs
```

Expected: Git top-level is this project; all listed runtime paths are ignored.

- [ ] **Step 2: Verify no old implementation tree remains**

Assert the root `finance_rag/` and `libs/` directories do not exist. Assert all compatibility modules under `src/finance_rag/` contain imports, `__all__`, docstrings, and no duplicated classes or functions.

- [ ] **Step 3: Verify data migration integrity**

Compare the Task 3 manifests. Require exact file counts, sizes, and hashes for historical artifacts. Assert `data/evaluation/evaluation_qa.md` still loads all original questions and preserves IDs and answers.

- [ ] **Step 4: Run the complete quality gate again**

Repeat every Task 10 Step 5 command from a fresh shell. Record exact test count and build output in the handoff.

- [ ] **Step 5: Commit audit-only fixes if any**

If Step 1–4 required changes, commit only those changes:

```powershell
git diff --name-only
git ls-files --others --exclude-standard
```

Review those concrete paths, stage each audit-owned file by its literal name,
then run `git commit -m "chore: finalize engineering migration"`. Do not use
`git add -A` and do not create an empty commit.

If no files changed, do not create an empty commit.

---

## Plan Self-Review

- Spec coverage: packaging, directory boundaries, compatibility, data migration, runtime isolation, vendored dependency removal, API split, application services, infrastructure adapters, evaluation split, CLI entry points, testing, documentation, Git boundaries, and final verification each map to an explicit task.
- Placeholder scan: no `TBD`, `TODO`, deferred implementation instruction, or undefined “similar to” step remains.
- Type consistency: `ProjectPaths`, `Settings`, `ExternalServiceError`, application service interfaces, `create_app`, and CLI signatures are defined before later consumption.
- Scope: frontend behavior, algorithms, models, API contracts, deployment, and CI remain explicitly outside this migration.
