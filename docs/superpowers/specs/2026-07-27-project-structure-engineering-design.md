# Finance RAG 项目目录工程化重构设计

## 目标

在不改变业务行为和外部契约的前提下，将项目迁移为标准 `src/` 布局，明确在线 RAG、离线评估、基础设施和运行时数据的边界，并建立可安装、可测试、可维护的工程结构。

本次重构保持以下契约不变：

- 现有 HTTP 路由和响应格式；
- 环境变量名称；
- Milvus collection 名称；
- benchmark JSON、CSV 和 Markdown 报告格式；
- `finance_rag.*` 公共导入路径；
- `python main.py`、现有 benchmark 脚本和前端开发命令。

## 当前问题

- 后端源码直接位于仓库根部，缺少标准打包元数据和安装入口。
- `api.py`、`evaluation.py`、`ablation.py` 等文件承担过多职责。
- API、业务编排、领域规则和 Milvus/LLM/Docling 实现相互耦合。
- `files/`、`artifacts/`、`logs/` 与源码混放，路径解析分散。
- `libs/` 保存约 118 MB 的第三方包副本，与虚拟环境和依赖声明重复。
- 缺少 README、`.gitignore`、`.env.example`、测试分层和统一质量命令。
- 项目原 `.git/` 为空，Git 会错误回退到 `F:\` 父级仓库。

## 目标目录

```text
finance-rag-retrieval/
├─ pyproject.toml
├─ README.md
├─ .env.example
├─ .gitignore
├─ main.py
├─ src/
│  └─ finance_rag/
│     ├─ __init__.py
│     ├─ __main__.py
│     ├─ api/
│     │  ├─ app.py
│     │  ├─ dependencies.py
│     │  ├─ schemas/
│     │  └─ routes/
│     │     ├─ chat.py
│     │     ├─ documents.py
│     │     ├─ evaluation.py
│     │     └─ health.py
│     ├─ application/
│     │  ├─ chat_service.py
│     │  ├─ document_service.py
│     │  └─ evaluation_service.py
│     ├─ domain/
│     │  ├─ models.py
│     │  ├─ retrieval.py
│     │  └─ evaluation.py
│     ├─ infrastructure/
│     │  ├─ config.py
│     │  ├─ logging.py
│     │  ├─ llm/
│     │  ├─ milvus/
│     │  └─ parsing/
│     ├─ evaluation/
│     │  ├─ datasets.py
│     │  ├─ metrics.py
│     │  ├─ ablation.py
│     │  └─ reporting.py
│     └─ compatibility/
├─ scripts/
├─ tests/
│  ├─ unit/
│  ├─ integration/
│  └─ fixtures/
├─ frontend/
├─ data/
│  ├─ source/
│  └─ evaluation/
├─ var/
│  ├─ artifacts/
│  └─ logs/
└─ docs/
   ├─ architecture/
   ├─ operations/
   └─ superpowers/
```

只在存在明确职责时创建子模块，避免为分层而制造空壳文件。

## 模块边界

### Domain

包含检索策略、评估条目、实验配置和领域异常等纯业务概念。不得依赖 FastAPI、Milvus、Docling、LangChain 的具体客户端或文件系统布局。

### Application

编排聊天、文档管理和评估用例。它依赖领域接口，通过依赖注入接收基础设施实现，不直接创建外部客户端。

### Infrastructure

封装配置、日志、LLM、Milvus、文档解析、嵌入和重排序。外部 SDK 的异常在此转换为带阶段、服务和可恢复性信息的应用异常。

### API

负责请求校验、依赖装配、HTTP/SSE 响应和异常映射。路由不包含检索或评估算法。FastAPI 应用由工厂创建，导入模块本身不应启动连接或创建运行时文件。

### Evaluation

承载数据集选择、指标、消融运行和报告生成。它复用领域和基础设施接口，但不依赖 API 层。在线策略评估通过 application 服务调用该能力。

### Compatibility

为旧模块导入提供薄代理。例如旧的 `finance_rag.ablation` 重新导出新 `finance_rag.evaluation` 中的公开符号。代理不保存业务实现，并在文档中标记弃用计划。

## 数据流

### 在线问答

```text
HTTP/SSE 请求
  -> API schema 与 route
  -> ChatService
  -> 查询改写 / 检索策略
  -> Milvus adapter / reranker / LLM adapter
  -> 领域结果
  -> API response schema
```

### 文档入库

```text
上传文件
  -> API 校验
  -> DocumentService
  -> parser/chunker adapter
  -> embedding 与 Milvus adapter
  -> 文档统计和领域结果
```

### 离线评估

```text
CLI 参数
  -> dataset profile
  -> 固定采样与实验配置
  -> application/infrastructure 检索和生成
  -> metrics
  -> 临时产物
  -> 原子替换正式 JSON/CSV/Markdown
```

## 路径与运行时数据

新增集中式 `ProjectPaths` 配置对象，统一解析：

- `data/source/`：知识库原始文档；
- `data/evaluation/`：评估题集和固定夹具；
- `var/artifacts/`：历史和新 benchmark 产物；
- `var/logs/`：运行日志；
- `frontend/dist/`：生产前端静态资源。

默认优先使用新路径。环境变量继续覆盖默认值。若迁移阶段检测到旧路径而新路径不存在，可以兼容读取并记录弃用警告；程序不得自动复制出双份数据。

历史 `artifacts/` 整体迁移到 `var/artifacts/` 并保留。`files/` 按用途拆分到 `data/source/` 和 `data/evaluation/`。

## 包管理与入口

使用 `pyproject.toml` 定义：

- Python 版本和项目元数据；
- 运行依赖；
- `dev`、`evaluation` 等可选依赖组；
- pytest、ruff 和 mypy 配置；
- console scripts：`finance-rag-server` 与 `finance-rag-ablation`。

新增标准入口：

```powershell
python -m finance_rag
finance-rag-server
finance-rag-ablation
```

以下入口保留为兼容代理：

```powershell
python main.py
python scripts\run_ablation_benchmark.py ...
python scripts\run_three_ablation_benchmark.py ...
cd frontend; npm run dev
```

`libs/` 在确认无自定义修改后移除，依赖统一由 `pyproject.toml` 和虚拟环境安装。

## 错误处理

- 领域层定义稳定异常类型，不依赖 HTTP。
- application 层补充用例上下文，但不暴露密钥或完整外部响应。
- infrastructure 层标记错误阶段、外部服务和是否可重试。
- API 层集中映射 HTTP 状态与安全错误消息。
- CLI 返回非零退出码，并将详细错误写入 stderr。
- benchmark 先写临时文件，四份产物全部成功后再原子替换，避免半成品报告。
- 正式 benchmark 的 Milvus 启停仍由显式运行流程管理，不在模块导入时执行。

## 测试与质量门禁

### 测试分层

- `tests/unit/`：领域规则、指标、路径解析、报告和纯服务逻辑；不访问外部服务。
- `tests/integration/`：FastAPI 路由、Milvus adapter、文档解析和配置装配；通过 pytest marker 显式运行。
- `tests/fixtures/`：最小、脱敏、可重复的测试数据。
- 兼容测试确保旧入口和旧导入路径保持可用。

### 验收命令

```powershell
pytest
pytest -m integration
ruff check .
ruff format --check .
mypy src/finance_rag
cd frontend; npm run build
```

mypy 采用渐进策略：新模块严格检查，兼容层可在明确注释下暂时放宽。

## 迁移顺序

1. 初始化项目独立 Git 仓库并建立忽略规则。
2. 添加 `pyproject.toml`、README、`.env.example`、`src/` 骨架和测试配置。
3. 引入 `ProjectPaths`，增加新旧路径行为测试。
4. 迁移 `files/`、历史 `artifacts/` 和 `logs/`。
5. 确认并移除 `libs/`，在干净环境安装声明依赖。
6. 迁移领域模型、指标和报告逻辑。
7. 迁移 Milvus、LLM、Docling、嵌入和重排序适配。
8. 迁移 application 服务并保留旧模块代理。
9. 拆分 FastAPI schema、route、dependencies 和 app factory。
10. 更新 CLI 与兼容脚本。
11. 重组测试并执行完整质量门禁。
12. 更新架构、开发、部署和 benchmark 操作文档。

每一步都必须保持快速测试通过。涉及路径移动时先增加兼容测试，再执行移动。

## Git 与版本控制

项目根使用独立 `.git/`，不读取或修改 `F:\` 父级仓库。`.gitignore` 至少排除：

- `.venv/`、`__pycache__/`、pytest/mypy/ruff 缓存；
- `frontend/node_modules/`、`frontend/dist/`；
- `var/`、模型缓存和临时 benchmark 文件；
- `.env`、密钥和本地 IDE 配置。

测试夹具和必要的数据说明通过白名单纳入版本控制。历史 benchmark 迁移保留在本地，但默认不提交。

## 非目标

- 不重新设计 Vue 界面或改变前端功能。
- 不改变 RAG 算法、模型选择、权重或 Milvus schema。
- 不修改 API、环境变量、collection 名称或报告 schema。
- 不在本轮引入 Docker Compose、CI 平台或云部署体系；可在结构稳定后单独设计。
- 不删除历史 benchmark 产物。

## 完成标准

- 源码从 `src/finance_rag` 安装和运行。
- 新旧入口均通过自动化测试。
- 在线 API 和正式 benchmark 的外部契约保持不变。
- 所有数据与运行时目录符合新布局，历史产物完整保留。
- `libs/` 不再存在，干净虚拟环境可从声明依赖安装。
- 默认测试、静态检查、类型检查和前端构建通过。
- 项目 Git 根准确指向当前目录，不再回退到 `F:\`。
