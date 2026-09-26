# 金融 Agentic RAG 知识库平台

层级切块 + 稠密/稀疏混合检索 + BGE 重排序的金融领域 RAG 问答平台。

[![Python](https://img.shields.io/badge/python-3.11+-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688)](https://fastapi.tiangolo.com/)
[![Milvus](https://img.shields.io/badge/Milvus-2.4+-00A86B)](https://milvus.io/)
[![Vue](https://img.shields.io/badge/Vue-3.4-42b883)](https://vuejs.org/)

---

## 核心特性

- **混合检索**：稠密向量（ONNX INT8 本地嵌入）+ 稀疏向量（BM25），RRF 融合排序
- **本地重排序**：BGE Reranker 本地推理，断崖检测自动截断低质结果；截断下限 `RERANKER_CLIFF_MIN_RESULTS=3`，保证极端断崖下仍留有兜底上下文
- **层级切块**：父子块结构，子块存 Milvus 精确检索，父块存 PostgreSQL 全文还原
- **语义分块**：可选，本地句子嵌入检测语义断点按主题切分（开关控制）
- **内容清洗与去重**：正则过滤页码/版权/导航噪声；SimHash 段落级去重 + 文档间疑似重复拦截
- **文档分类**：四分类（投研类 / 合规风控类 / 业务运营类 / 管理类），分类作为强制字段写入
- **知识库类别管理**：内置四类与自定义类别统一管理（支持改名/删除）；所有类别共用同一底层集合，为逻辑分类视图
- **元数据过滤**：检索时叠加结构化条件（分类多选 `in`、日期范围 `gte/lte`、部门精确匹配等）
- **Agentic RAG**：查询改写 -> 多查询检索 -> 关键词增强 -> 可选 LangGraph 反思循环
- **HyDE 检索增强**：可选，先让 LLM 生成假设性回答，用其向量扩大召回
- **动态 K**：可选，按问题复杂度（规则分类）自动调整召回深度
- **增量更新与版本**：元数据（mtime/size/version）+ SHA256 混合检测；可选版本保留模式（历史版本可追溯，检索仅命中最新版本）
- **引用验证**：自动校验 LLM 回答中的 `[N]` 引用标记与来源文档的匹配度
- **拒答机制**：检索为空或引用校验低分时拒绝回答（流式路径尾部追加警示）
- **云存储支持**：本地文件系统 / S3（MinIO 兼容）双后端，环境变量一键切换
- **异步索引**：在线上传异步入库（task_id 轮询）
- **流式入库流水线**：进程单例**常驻队列 + 分阶段 worker**（解析 → 切块 → 嵌入 → 写库四段重叠），上传完成即入队、**入队即返回**，不等同批其它文档，阶段重叠覆盖整个服务会话
- **背压与自愈**：各阶段队列有界，满时上传接口返回 **503**（不阻塞调用方）；阶段 worker 异常不终止并自动重启，由 supervisor 兜底；关闭时先排空在途文档再释放进程池/线程池
- **两阶段写库**：写锁只包住向量库「删旧行 + 插新行」原子区（含文档间 SimHash 去重的"先查再插"），**存储上传等网络 IO 移出锁外并支持多消费者并行**
- **集合级写锁落在资源侧**：`KnowledgeBase` 按 collection 自持写锁，入库流水线、对象存储事件回调、删除/管理路径共用同一串行点，不再依赖各调用方自行持锁
- **本地 MinerU 文档解析**：统一解析支持格式，配置 exclude 去除页眉、页脚和页码，随后执行规则清洗与 SimHash 去重
- **策略评估**：Ragas 指标（faithfulness / answer_relevancy / context_precision 等 6 项）+ 检索指标（Hit Rate / MRR / NDCG），A/B 实验统一入口 `scripts/ab_rag.py`（配置开关驱动，无 HTTP API）
- **JWT 鉴权**：单用户登录，开发模式可免密

---

## 架构

```
┌─────────────┐     HTTP/SSE      ┌──────────────────────────────┐
│  Vue3 前端   │ ◄──────────────► │  FastAPI (finance_rag.src)    │
│  Element Plus│                   │                              │
└─────────────┘                   │  /api/auth  /api/chat        │
                                  │  /api/documents              │
                                  │  /api/knowledge-bases        │
                                  └──────────┬───────────────────┘
                                             │
                    ┌────────────────────────┼─────────────────────────┐
                    │                        │                         │
                    ▼                        ▼                         ▼
           ┌──────────────┐      ┌───────────────────┐      ┌──────────────────┐
           │   Milvus      │      │  DeepSeek LLM     │      │  S3 / MinIO      │
           │ 稠密+BM25稀疏  │      │  langchain-deepseek│      │  boto3           │
           │ RRF 融合      │      │                   │      │                  │
           └──────────────┘      └───────────────────┘      └──────────────────┘
                    │
                    ▼
           ┌──────────────┐
           │  ONNX 嵌入    │
           │  BGE 重排序   │
           │  本地 CPU 推理 │
           └──────────────┘
```

---

## 项目结构

```
finance-rag-retrieval/
├── finance_rag/src/                     # 源代码（分层架构）
│   ├── api/                             # HTTP 层
│   │   ├── routes/                      # auth / chat / documents / knowledge_bases
│   │   ├── streaming/                   # SSE 流式
│   │   └── dependencies.py              # JWT 鉴权依赖
│   ├── core/                            # 核心配置与横切
│   │   ├── config.py                    # 配置中心（环境变量 + 默认值）
│   │   ├── dependencies.py              # 全局依赖注入入口
│   │   ├── exceptions.py                # 异常分类 + 友好文案
│   │   └── logger.py                    # 日志
│   ├── services/                        # 应用服务层
│   │   ├── chat_service.py              # 问答链（改写/动态K/HyDE/引用验证）
│   │   ├── document_service.py          # 文档管理（异步上传）
│   │   ├── ingestion_pipeline.py        # 入库流水线（解析/切块/嵌入/写库 三阶段）
│   │   ├── knowledge_base_service.py    # 知识库类别注册表（PostgreSQL）
│   │   ├── citation_validator.py        # 引用验证 + 拒答策略
│   │   └── task_service.py              # 异步任务状态管理
│   ├── agent/                           # Agent 组件
│   │   ├── prompts/                     # Prompt 模板
│   │   ├── state/                       # Agent 状态模型
│   │   ├── tools/                       # 检索工具
│   │   └── query_classifier.py          # 查询复杂度分类（动态 K）
│   ├── orchestration/                   # LangGraph 编排（graph / state / nodes）
│   ├── rag/                             # RAG 核心
│   │   ├── ingestion/                   # 切块（纯标准库递归切分）/ MinerU解析 / 清洗去重 / 图片描述 / 切块进程池 worker
│   │   ├── models/                      # 分类常量 + 版本号提取
│   │   └── retrieval/                   # 混合检索 + 重排序 + 父块还原
│   ├── infrastructure/                  # 基础设施适配层
│   │   ├── vector_store/                # Milvus + ONNX 嵌入
│   │   ├── storage/                     # 存储抽象（Local / S3）
│   │   ├── relational_db/               # PostgreSQL（会话 + 知识库注册表）
│   │   ├── cache/                       # Redis 缓存（预留）
│   │   ├── task_queue/                  # 任务队列协议（预留）
│   │   ├── llm/                         # LLM 适配边界
│   │   └── parsing/                     # 解析适配边界
│   ├── eval/                            # Ragas 评估 + A/B 实验框架（CLI 形态）
│   ├── schemas/                         # Pydantic 请求/响应模型
│   ├── utils/                           # 审计 / 日志 / 指标
│   └── main.py                          # FastAPI 入口
├── config/                             # 保留目录（运行时业务数据已迁移至 PostgreSQL）
├── frontend/                           # Vue3 + Vite
├── tests/                              # 单元/集成测试
├── scripts/                            # seed_data.py / ab_rag.py 等脚本
├── docker-compose.yml
└── Dockerfile
```

---

## 快速开始

### 前置条件

- Python 3.11+
- Docker（用于 Milvus、MinIO）
- DeepSeek API Key（[获取地址](https://platform.deepseek.com)）

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，至少填入 DEEPSEEK_API_KEY
```

关键配置项：
```ini
DEEPSEEK_API_KEY=sk-xxx          # 必填：DeepSeek API Key
JWT_ADMIN_PASSWORD=your-password  # 必填：登录密码
# 查询改写默认复用 DASHSCOPE_API_KEY 走 qwen-turbo 轻量模型（大幅降低改写延迟），
# 未配置时回退 DeepSeek 主模型，再回退纯规则改写；可经 QUERY_REWRITE_MODEL 等覆盖
```

### 3. 启动 Milvus（Docker Compose）

```bash
docker-compose up -d etcd minio milvus
```

> 如果本机已有独立 Milvus 实例，设置 `MILVUS_URI` 即可，无需启动 Compose 服务。

### 4. 启动服务

```bash
python main.py
```

访问 http://localhost:8000 进入前端页面。

### 5. 上传文档

登录后进入"文档管理"页面，上传 `.md` / `.txt` / `.pdf` 文件。系统自动解析、清洗（页码/版权/导航噪声过滤 + 段落级 SimHash 去重）、切块、嵌入并存入 Milvus。

上传时可选择文档分类（内置四类或自定义类别），分类作为集合的 `category` 标量字段写入；与已入库文档高度相似的文件会被拦截（可配置）。

分类可在"知识库管理"页面统一管理：内置四类与自定义类别一样支持**编辑、改名、删除**（改名会联动迁移集合内已有文档的分类；有文档的类别需先清空才能删除）。所有类别共用同一个 Milvus 集合，只是逻辑分类视图。

### 6. 版本保留（可选）

文档统一通过「文档管理」页在线上传入库；默认同一文档重新入库时**覆盖**旧记录。
如需保留历史版本（如合同变更追溯），设置 `ENABLE_VERSIONING=true`（旧集合需先
调用 `KnowledgeBase.rebuild_collection()` 重建 schema）：

- 同一 source 的多个版本共存于集合中，`is_current` 标记最新版本；
- 检索与文档列表只命中最新版本，历史版本可追溯（`GET /api/documents?include_versions=true`）；
- 支持按版本删除：`DELETE /api/documents/{source}?version=v2`（不带 version 删除全部版本）；
- `MAX_VERSIONS_PER_DOC` 限制每文档版本数（0 = 无限），超出自动清理最旧版本。

### 文档分类与元数据过滤

知识库文档分四类，分类写入集合的 `category` 标量字段（另有 `date` / `version` / `is_current` 等字段）。内置四类：

| 分类值 | 中文标签 |
|--------|----------|
| `investment_research` | 投研类 |
| `compliance_risk` | 合规风控类 |
| `business_operations` | 业务运营类 |
| `management` | 管理类 |

聊天页支持在检索时叠加结构化过滤条件（分类多选 + 日期范围），Milvus 原生过滤表达式示例：

```
category in ["compliance_risk", "investment_research"] and date >= "2024-01-01"
```

API 调用时通过 `filters.metadata` 传入：

```json
{
  "query": "...",
  "filters": {
    "metadata": {
      "category": ["compliance_risk"],
      "date": {"gte": "2024-01-01"}
    }
  }
}
```

### Docker Compose 全栈启动

```bash
docker-compose up -d    # 启动 etcd + minio + milvus + app
```

---

## API 端点

### 鉴权
| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/auth/login` | 用户登录，返回 JWT |

### 问答
| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/chat` | 非流式问答，含引用验证 + 拒答策略（响应含 `citation_validation` / `answer_rejected` / `low_confidence`） |
| POST | `/api/chat/stream` | SSE 流式问答（`done` 事件同样携带上述字段） |

请求体可选参数：`k` / `rerank_top_n`（不传时用默认值；开启动态 K 后按问题复杂度自动调整）、`use_rerank`、`filters`、`strategy`。

### 知识库管理（类别）
| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/knowledge-bases` | 类别列表（内置四类 + 自定义，含文档数/切块数统计） |
| POST | `/api/knowledge-bases` | 新建自定义类别（`{"name", "display_name", "description"}`） |
| PATCH | `/api/knowledge-bases/{name}` | 修改显示名/描述/类别值（`name` 字段改名时联动迁移集合内文档分类） |
| DELETE | `/api/knowledge-bases/{name}` | 删除类别（有文档 409；内置类别同样可删；不删除集合） |

### 文档
| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/documents/upload` | 批量上传文档（可选 `category` 表单字段）。**入队即返回**；入库队列满或流水线关闭时返回 **503** |
| POST | `/api/documents/upload-async` | 异步上传（返回 task_id，可选 `category`）。同样受 503 背压保护 |
| GET | `/api/documents` | 文档列表（含分类；`?include_versions=true` 返回历史版本明细） |
| DELETE | `/api/documents/{source}` | 删除文档（`?version=v2` 仅删指定版本，默认全部） |
| GET | `/api/kb/stats` | 知识库统计 |
| GET | `/api/tasks/{task_id}` | 异步任务状态（含 `progress.stage`：`parsed`/`chunked`/`written`） |

> **上传的两种路径**：默认走**流式常驻流水线**（`INGEST_PIPELINE_ENABLED=true`），
> 上传完成即入队并立刻返回 task_id，前端轮询 `/api/tasks/{task_id}` 看进度；
> 关闭开关则回退到逐文件串行入库（结果一致，用于对照与排障）。
> 收到 503 表示流水线暂时打满，客户端应退避重试——已入队的文件在重试时会命中
> 内容指纹而被增量跳过，不会重复入库。

### 评估

> 评估为 CLI 形态，不提供 HTTP API。RAG 优化点的 A/B 对比统一入口为
> `scripts/ab_rag.py`（Ragas 指标 + 检索指标），使用方式见下文
> 「A/B 测试」章节；评估逻辑位于 `finance_rag/src/eval/ragas_eval.py`。

### 监控
| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/health` | 健康检查（含 Milvus 状态） |
| GET | `/metrics` | Prometheus 指标 |

入库流水线的自定义指标（用于定位入库存吞吐瓶颈）：

| 指标 | 类型 | 说明 |
|------|------|------|
| `finance_rag_ingest_queue_depth{stage}` | Gauge | 各阶段入口队列深度（`pending`/`chunk`/`embed`/`write`），持续上涨即为瓶颈段 |
| `finance_rag_ingest_stage_seconds{stage}` | Histogram | 单文档各阶段耗时 |
| `finance_rag_ingest_queue_rejected_total` | Counter | 队列满被拒绝的提交数（背压触发次数） |
| `finance_rag_ingest_worker_restarts_total{stage}` | Counter | 阶段 worker 重启次数（>0 说明曾自愈） |

---

## 配置参考

必填与常用配置见 `.env.example`。主要分组：

| 组 | 说明 |
|----|------|
| LLM | DeepSeek API 配置（含生成温度） |
| 嵌入模型 | ONNX INT8 本地嵌入 |
| 向量数据库 | Milvus 连接 + 集合配置 |
| 文档切块 | 层级父子切块参数 + 语义分块开关 |
| 增量与版本 | 变更检测模式 + 版本保留开关 |
| 混合检索 | RRF 融合 + BM25 |
| 重排序 | BGE Reranker |
| 问答链路 | Top-K、查询改写、动态 K、HyDE |
| Ragas 评估 | 超时、重试、并行度 |
| 幻觉治理 | 引用验证 + 拒答阈值 |
| 安全配置 | CORS + JWT |
| 云存储 | Local / S3 切换 |

---

## 检索优化特性

以下优化点均可通过环境变量独立开关（默认保守），逐个启用后可用 `scripts/ab_rag.py` 做 A/B 对比验证效果（详见「A/B 测试」章节）。

内容清洗（规则过滤页码/版权声明/导航链接等噪声）、文档内段落级 SimHash 去重、
文档间 SimHash 去重（海明距离 ≤ 3 拒绝入库）已固化为入库固定流程，不可通过配置关闭。

图片处理：普通 PDF 解析时会自动提取嵌入图片并保存到存储后端；配置了
`DASHSCOPE_API_KEY`（默认 DashScope Qwen-VL，可换任意 OpenAI 兼容视觉端点）时，
同时用多模态模型为每张图生成中文描述，描述文本随切块进入知识库。未配置 key
时仅保存图片、描述为占位提示；提取失败/描述失败均只告警，不阻断入库。
视觉模型调用对超时/限流/5xx 会自动重试（`IMAGE_CAPTION_MAX_RETRIES`，默认 2）。

异常降级与重试：向量库连接/查询失败会与「知识库无内容」区分对待并返回友好中文提示；
嵌入模型失败、对象存储失败、LLM 超时/限流（自动重试）等均在对应环节 try/except
降级或给出可读报错，不把底层异常串直接暴露给前端。

### 入库流水线（asyncio.Queue + Executor 三阶段）

在线上传的入库路径是一条**三阶段流水线**：某个文档切块完成即可进入 Embedding 队列，
不必等待同批其它文档解析结束（`INGEST_PIPELINE_ENABLED=true`，默认开启）。

```
┌──────────┐  parse_queue  ┌────────────┐ chunk_queue ┌─────────────┐ write_queue ┌──────────┐
│ 解析阶段 │ ────────────► │  切块阶段  │ ──────────► │  Embedding  │ ──────────► │ 写库阶段 │
│ MinerU   │               │ 清洗/去重  │             │  阶段       │             │ 向量写入 │
│ 线程池   │               │ 层级切分   │             │  线程池     │             │ await IO │
└──────────┘               │ 进程池     │             └─────────────┘             └──────────┘
```

| 阶段 | 执行体 | 为什么这样选 |
|------|--------|--------------|
| 解析 | `ThreadPoolExecutor`，受 `MINERU_MAX_CONCURRENCY` 限流 | MinerU 加载 GB 级 ONNX/GPU 权重，进程池化会让内存成倍增长 |
| 切块 | `ProcessPoolExecutor`（spawn） | 规则清洗 / SimHash 去重 / 标题注入 / 层级切分是**纯 Python**，受 GIL 限制，进程池才是正解（21 万字符文档实测约 3.4s） |
| Embedding | 专用 `ThreadPoolExecutor` | ONNX 是 C 层实现，推理期间释放 GIL；ONNX 会话本身串行化推理（与 `BGEReranker` 同一约定） |
| 写库 | 事件循环内 `await` | Milvus 是网络 IO；只有「删旧行 + 插新行」的原子区需要写锁串行 |

关键行为：

- **背压**：阶段间队列有界（`INGEST_QUEUE_MAXSIZE`），解析槽位保持到入队成功才释放，
  队列满时不再启动新的解析，内存上界约为 `INGEST_QUEUE_MAXSIZE × MAX_UPLOAD_SIZE_MB`；
- **失败隔离**：任一阶段异常只让该文件失败（task 置 `failed` 并给出中文原因），整批继续；
- **降级路径**：文档短于 `INGEST_CHUNK_MIN_CHARS` 时内联切块；进程池不可用
  （受限容器/沙箱禁止创建子进程或管道）时自动降级为内联切块并停止重试；
- **阶段进度**：`GET /api/tasks/{task_id}` 额外返回 `progress`（`parsed` / `chunked` / `embedded` / `written`）；
- **回退开关**：`INGEST_PIPELINE_ENABLED=false` 恢复原「逐文件串行」入库路径（功能等价）。

调参键：`INGEST_PIPELINE_ENABLED`、`INGEST_CHUNK_WORKERS`（0 = 自动）、
`INGEST_CHUNK_MIN_CHARS`、`INGEST_QUEUE_MAXSIZE`、`INGEST_EMBED_WORKERS`、
`DOCUMENT_PARSE_WORKERS`、`MINERU_MAX_CONCURRENCY`。

> 注意：开启语义分块（`ENABLE_SEMANTIC_CHUNKER=true`）时，切块进程池的每个 worker
> 会各自加载一份句向量模型，内存按 worker 数翻倍，建议显式设置 `INGEST_CHUNK_WORKERS=1`。

### 冷启动优化

进程冷启动曾有一个隐蔽的**间接重依赖导入**问题，已修复：

```
chunker.py 的 from langchain_text_splitters import RecursiveCharacterTextSplitter
  └─ langchain_text_splitters/__init__.py 会导入全部 splitter 子类
     └─ langchain_text_splitters.sentence_transformers
        └─ sentence_transformers → transformers → torch（+ datasets、nltk）
```

也就是说，只为用 `RecursiveCharacterTextSplitter` 一个类，每次进程启动都要多付约
**11 秒**（约 5000 个模块）。全仓只有这一处调用，Web 服务、`scripts/*`、
A/B 子进程、切块进程池 worker 全都在白付这笔开销。

三项优化：

| 优化 | 做法 | 效果 |
|------|------|------|
| 切断 splitters 依赖 | `finance_rag/src/rag/ingestion/recursive_splitter.py`：纯标准库等价实现（参数只保留用到的部分），切块结果**逐字节一致** | 该链路上不再导入 torch/transformers |
| 应用入口惰性化 | `routes/chat.py` 的 service 导入改到请求时（`_chat_fns()`） | `import finance_rag.src.main` 不再连带 langchain |
| 预热后台化 | lifespan 用 `asyncio.create_task` + `asyncio.to_thread` 在后台完成「重依赖导入 + 嵌入/重排序模型加载」，`/api/health/ready` 仍作为门禁 | 端口绑定与 accept 不再被阻塞 |

实测（本机 Windows / Python 3.13 / `import finance_rag.src.main`）：

| 指标 | 优化前 | 优化后 |
|------|--------|--------|
| 应用入口导入 | 12.6s | **1.8s** |
| 切块模块导入 | 10.9s | **0.4s** |
| 导入后是否加载 torch / transformers | 是 | **否** |
| 服务可 accept 时间 | 等预热完成（约 +8s 起） | **导入完成即可**（约 2s） |

重依赖仍在后台加载（约 6s），首个请求前的就绪性由 `/api/health/ready` 把守；
`WARMUP_ENABLED=false` 可完全跳过预热。回归守卫见
`tests/unit/test_recursive_splitter.py`（含 `sys.modules` 断言，防止再次把
`langchain_text_splitters` 写回热路径）。

> 切分器等价性测试默认跳过（需导入重依赖），显式执行：
> `python -m pytest tests/unit/test_recursive_splitter.py -m slow`
> 当前在 README / 评估语料 / `chunker.py` 全文 + 边界用例 × 18 组参数上共 285 项比对全部一致。

| 特性 | 开关（默认） | 说明 |
|------|-------------|------|
| 语义分块 | `ENABLE_SEMANTIC_CHUNKER=false` | 本地句子嵌入检测语义断点按主题切分（首次使用需下载小模型） |
| 版本保留 | `ENABLE_VERSIONING=false` | 历史版本共存、检索仅命中最新版（旧集合需先重建 schema） |
| 动态 K | `DYNAMIC_K=false` | 按问题复杂度（规则分类）调整召回深度，档位见 `DYNAMIC_K_MAP` |
| HyDE | `ENABLE_HYDE=false` | LLM 生成假设回答增强召回（每次额外 1 次 LLM 调用） |
| 拒答机制 | `ENABLE_REFUSAL=true` | 检索为空 / 来源相关性不足 / 引用校验低分时拒答（流式追加警示） |
| 生成温度 | `LLM_TEMPERATURE=0.1` | 较低温度降低随机性、减少幻觉 |

---

## A/B 测试（RAG 优化点对比）

所有 RAG 优化点均以**配置开关**形式组织为 A/B 实验：每个实验一对臂（基线 vs 优化开启），臂在独立子进程中执行（环境变量覆盖生效），逐题跑「检索 → 生成 → Ragas 评分」，最后输出指标对比报告。

### 快速开始

```bash
# 列出全部内置实验
python scripts/ab_rag.py --list

# 基于当前知识库自动生成分层测试集（约 7:2:1，人工抽查后作为评测基准）
python scripts/ab_rag.py --generate-testset --count 40

# 跑单个实验（fast = 3 项核心指标；--full = 6 项）
python scripts/ab_rag.py --experiment hyde_on_off --fast --sample 5

# 跑多个 / 全部实验
python scripts/ab_rag.py --experiment dynamic_k_on_off hyde_on_off --fast
python scripts/ab_rag.py --all --fast --sample 5

# 自定义测试集 / 抽样种子
python scripts/ab_rag.py --experiment hyde_on_off \
    --testset path/to/qa.md --sample 10 --seed 7
```

结果落盘 `scripts/results/ab/<实验名>/<时间戳>/`：`report.json`（指标对比）
/ `report.md` / `per_query.csv`（逐题明细）/ 各臂 `<arm>.json`。

### 内置实验目录

| 实验名 | 对比内容 | 类型 |
|---|---|---|
| `hybrid_vs_dense` | 稠密检索 vs RRF 混合检索 | 运行时 |
| `rerank_on_off` | BGE 重排序开/关 | 运行时 |
| `dynamic_k_on_off` | 动态 K 开/关 | 运行时 |
| `hyde_on_off` | HyDE 检索增强开/关 | 运行时 |
| `langgraph_on_off` | 标准 RAG vs Agentic RAG | 运行时 |
| `refusal_on_off` | 拒答机制开/关 | 运行时 |
| `temperature_high_vs_low` | 生成温度 0.7 vs 0.1 | 运行时 |

所有实验均为运行时实验，共享默认集合 `finance_kb`。
版本保留（`ENABLE_VERSIONING`）影响历史版本可检性而非单次问答质量，
不设 A/B 实验；内容清洗与 SimHash 去重已固化为入库固定流程（非开关），
同样不设 A/B 实验。*

### Ragas 指标

| 模式 | 指标 |
|---|---|
| `--fast`（默认） | faithfulness（忠实度）、answer_relevancy（回答相关性）、context_precision（上下文精确率） |
| `--full` | 上述 3 项 + context_recall（上下文召回率）、answer_correctness（答案正确性）、context_entity_recall（实体召回） |

**文档级本地检索指标**：hit_rate@5（至少命中一篇）、MRR@5、NDCG@5、
`doc_coverage`（覆盖比例，多跳取 0/0.5/1.0）、`doc_recall`（全部命中）。

**块级证据指标**（按测试集标注的证据 chunk id 计算）：
`evidence_hit`（top-5 是否含任一证据块）、`evidence_coverage`（证据块覆盖比例）、
`evidence_all_in_topk`（全部证据块是否都在 top-5）。证据块排名 `evidence_rank`
写入 `per_query.csv`，未命中记 `k+1`。

其余：跨文档多跳 `multi_hop_hit`（= `doc_recall`，相关文档全部召回）、
负样本拒绝 `negative_rejection`，以及加权综合分 composite_score。报告附 Δ、
相对变化与胜出臂，并按问题类型分层汇总（`metrics_by_type`，**带样本数 n**）；
双臂执行顺序按 seed 随机化以降低顺序偏差。

#### 指标口径与坑（重要）

- **`hit_rate` 在多跳问题上会饱和**：只要命中 2 篇相关文档中的任意 1 篇就是 1.0，
  因此它无法反映多跳覆盖程度；`context_recall` 由 Ragas LLM judge 打分，
  对 top-5 上下文普遍给 1.0，同样饱和。**要看多跳检索能力，请用
  `doc_coverage` / `evidence_coverage` / `evidence_rank`。**
- **测试集未标注证据块的条目，块级指标为 n/a（空），不计入均值**——不会伪装成 0.0。
- **自动生成的跨文档多跳题证据可能不可信**：`_generate_multi_hop` 是把随机抽到的
  两篇文档配对出题，第二篇常常与问题无关，此时 `multi_hop_hit=0` **不能归因为检索失败**。
  报告会单列 `[注意] 疑似标注不可信的条目`（`suspect_annotations`），
  用 `--report-diagnostics` 可在终端直接打印明细。判断顺序应该是：
  先看这类条目占多少 → 再看 `evidence_rank` 是否落在 top-15 候选内却被 rerank 截断
  （那才是重排序问题）→ 最后才调检索参数。
- **`--sample N` 按 `question_type` 分层抽样**：保证 N 不小于类型数时每类至少 1 题，
  避免小样本下整类缺失或同一题重复入集。


### 测试集

自动生成的评估集按 **约 7:2:1** 分层组织，每个条目标注证据 chunk id：

| 类型 | 占比 | 说明 |
|---|---|---|
| `single_hop` 单跳事实 | ~70% | 单块即可回答的事实题，标注 1 个证据 chunk id |
| `multi_hop` 跨文档多跳 | ~20% | 需综合两个不同文档的信息，标注 2 个 chunk id |
| `negative` 负样本陷阱 | ~10% | 前提错误/超范围问题，期望系统拒绝而非编造 |

- 生成：`python scripts/ab_rag.py --generate-testset --count 40`（默认
  `evaluation_qa_generated.md`；`--count` 目标题数、`--per-doc` 每文档候选块数）；
- 条目格式：`## N. 问题` + `- 问题类型：...` + `- 相关文档：...` +
  `- 证据chunk：id1, id2`（负样本为 `- 期望行为：...`）+ `### 标准答案`；
- 多跳条目**优先枚举不同文档组合并去重**（含 query 去重），避免 source 较少时
  反复抽到同一对文档、产出多条完全相同的题目；
- 运行前自动剔除「相关文档不在知识库」的条目（`--allow-missing` 保留）；
- `per_query.csv` 除 `question_type` / `chunk_ids` 外，还输出实测检索结果
  `retrieved_doc_stems` / `retrieved_chunk_ids` 与 `evidence_rank`，
  便于直接对比「标注证据 vs 实际召回」；`report.md` 附分类型指标（带 n）与
  疑似标注不可信条目清单。

### 成本参考

fast 模式每题每臂约 5-7 次 LLM 调用（改写/HyDE + 生成 + 3 项 judge），
full 模式约 8-12 次。`--sample 5` 单实验约 10-25 分钟。

---

## 技术栈

| 层 | 技术 |
|----|------|
| 后端框架 | FastAPI + Uvicorn |
| 向量数据库 | Milvus 2.4+（标量过滤 + BM25 Function 稀疏向量） |
| LLM | DeepSeek (langchain-deepseek) |
| 嵌入模型 | BAAI/bge-small-zh-v1.5（ONNX INT8 量化） |
| 重排序 | BAAI/bge-reranker-v2-m3 |
| 文档解析 | 本地 MinerU + Python 规则清洗 + SimHash 去重 |
| 去重 | SimHash（纯标准库，字符 n-gram + md5 投票） |
| 评估框架 | Ragas 0.4.3 |
| 前端 | Vue 3 + Element Plus + Vite |
| 存储 | 本地文件 / oss |
| 监控 | Prometheus + prometheus-fastapi-instrumentator |
| 鉴权 | PyJWT |

---

## 开发

```bash
# 安装开发依赖
pip install -r requirements.txt
pip install pytest pytest-asyncio httpx

# 运行测试（覆盖清洗/去重/增量/版本/动态K/拒答/类别注册表/入库流水线/集合写锁/断崖检测/A-B 框架等 162 个用例）
python -m pytest tests/ -v

# 前端开发
cd frontend
npm install
npm run dev              # Vite dev server (port 5173)

# A/B 测试：RAG 优化点对比（Ragas 指标；详见「A/B 测试」章节）
python scripts/ab_rag.py --list
python scripts/ab_rag.py --experiment hyde_on_off --fast --sample 5
python scripts/ab_rag.py --generate-testset --count 40
```
