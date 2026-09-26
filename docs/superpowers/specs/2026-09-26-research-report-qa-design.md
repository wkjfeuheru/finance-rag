# 研报问答改造设计

## 背景与目标

业务痛点：金融分析师查看研报、整合数据和信息耗时长。目标是通过本问答系统提高"查资料"效率。

改造载体：把知识库内容换成研报，按「个股 / 行业 / 宏观」维度建立元数据，并据此判定 RAG 链路需要改造的部分。

本文档是七轮需求澄清后的共识记录。**未决项一律显式列出（见「假设」与「风险」），不做静默假设。**

## 非目标

- 不做多租户、权限模型、用户表（沿用单用户 JWT）。
- 不做 `rating` / `target_price` 抽取（研报首页评级格式不统一，抽取错误率高，且非"查资料"必需）。
- 不做跨券商口径归一化（不做 2026E/FY26、归母净利/净利、亿/百万的换算），只输出原文口径 + 出处 + 报告日期。
- 不做会话历史持久化（前端 store 保持内存态）。
- 不设默认时间窗（不静默丢弃旧报告）。
- 不新增"标的工作台"页面。

## 目标用户与验收

| 项 | 结论 |
| --- | --- |
| 用户 | 单分析师工作台 |
| 真实用户试用 | 无。由实施方扮演分析师 |
| baseline 场景 | 跨券商对比同一指标 |
| 验收口径 | 仅自动指标（下述硬阈值） |
| 业务目标 | "查资料 30min → ≤5min" 降级为方向性目标，**不进验收门槛** |
| 前端改造 | 保留，但**不计入验收门槛** |

### 硬阈值（100 篇语料、20 题 gold 集上计算）

| 指标 | 阈值 |
| --- | --- |
| `hit_rate@5` | ≥ 0.90 |
| `evidence_rank ≤ 3` 占比 | ≥ 0.80 |
| 引用可溯源率（答案中带 `[N]` 且通过 `CitationValidator` 的比例） | ≥ 0.95 |
| 负样本拒答正确率 | = 1.00 |

题集构成：20 题 = 10 题跨券商对比 + 8 题单跳定位 + 2 题负样本。gold 集要求 `review_status == "approved"`。

## 语料与合规前提

- 语料由需求方提供真实研报 PDF，**50–100 篇起**，目标规模 500–5000 篇（千份级）。这是整个计划的先决条件。
- 标的域：A 股个股 + 行业 + 宏观。行业分类走申万（默认 2021 版一级/二级）。
- 文本只发检索片段给外部 LLM API（沿用现状）。
- **图片允许全量外发视觉模型**（DashScope `qwen-vl-max`），需新增外发审计日志。

## 1. 数据模型变更

### Milvus schema（`finance_rag/src/infrastructure/vector_store/milvus_kb.py:225-244`）

现有 schema 为 `enable_dynamic_field=False`，新增字段必须重建集合。

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `security_code` | VARCHAR(32) | **单值**，A 股 6 位代码 |
| `security_name` | VARCHAR(64) | 证券简称 |
| `industry_l1` | VARCHAR(32) | 申万一级 |
| `industry_l2` | VARCHAR(32) | 申万二级 |
| `report_type` | VARCHAR(16) | 个股 / 行业 / 宏观 |
| `broker` | VARCHAR(64) | 券商 |
| `block_type` | VARCHAR(16) | text / table / image |
| `meta_source` | VARCHAR(16) | regex / llm / manual |
| `needs_review` | BOOL | 低置信抽取待人工确认 |
| `start_page` | INT64 | 起始页（1-based） |
| `end_page` | INT64 | 结束页 |
| `image_key` | VARCHAR(256) | 图片在对象存储的 key |

沿用字段：`source` / `title` / `chunk` / `parent_id` / `chunk_key` / `content_hash` / `heading_path` / `category` / `date` / `version` / `is_current` / `is_deleted`。

### 单值 `security_code` 的兜底

一篇行业研报覆盖 N 只股票，没有合法单值代码。约定：

- 个股研报：填 `security_code` + `security_name` + `industry_l1/l2`，`report_type=个股`。
- 行业研报：`security_code` 留空，填 `industry_l1/l2`，`report_type=行业`。
- 宏观研报：只填 `report_type=宏观`。

于是"查某只票"与"查某行业"两条路都通：单值的限制由 industry 维度兜住。多标的对比查询用 `security_code in [...]`，单值字段支持。

### PostgreSQL 新增 `table_chunks`

现 `parent_chunks`（`finance_rag/src/infrastructure/relational_db/parent_store.py:25-36`）是纯 `Text` content、不存 JSONB。新增独立表：

- 复合主键 `(collection, id)`，与 `parent_chunks` 同构；
- `payload` (JSONB)：二维数组形式的结构化表体；
- `markdown` (Text)：原始 Markdown 表；
- `row_count` / `source` / `heading_path`。

建表复用 `parent_store` 的自愈模式（`_ensure_heading_path_column`，`parent_store.py:53-71`）。

`parent_chunks` 不存元数据，元数据修正无需同步它。

## 2. 元数据抽取与人工修正

### 抽取链路

```
文件名正则 ──命中──> 直接赋值
     │
     └─未命中─> 单次 LLM 调用（结构化输出）
                    │
                    └─> 结果 + meta_source + 置信度写库
```

- 正则目标：文件名中的 `名称(代码)` 模式、6 位代码、券商、日期。研报文件名规范，命中率高。
- LLM 目标：`security_code` / `security_name` / `report_type` / `broker` / `industry_l1` / `industry_l2`。
- **行业必须从申万 enum 中选**：清单写进 prompt，并校验「二级必须属于所选一级」，非法值丢弃，抽不到留空。
  - 理由：过滤走精确匹配（`HybridRetriever._build_filter`，`hybrid_retriever.py:463-487` 的 `==` / `in`）。LLM 自由输出"白酒"时，`industry_l1 == "食品饮料"` 的过滤会**静默漏召**，而这正是单值 `security_code` 的兜底路径。
- 留空值不参与过滤，因此不会误收窄召回。

### 人工修正

- `PATCH /api/documents/{source}/metadata`：只做 Milvus 标量 upsert，**不重新嵌入**（复用现成的 `fetch_existing_vectors`）。
- 前端文档列表加"待确认"标记，详情弹层编辑。
- 提交后 `meta_source = manual`、`needs_review = false`。

## 3. 检索链路

### 检索策略

全库检索 + 自动元数据过滤，**高置信才施加**：

- 抽到唯一标的 → 施加过滤，并在前端做成**可见可改的 chip**。
- 抽不到或不唯一 → 不加任何过滤（宁可召回宽）。
- 误抽的代价是"用错误条件检索 → 静默答出错误范围的结论"，比查不到更严重，因此不做"抽到就施加"。

### 代码改动点

| 位置 | 改动 |
| --- | --- |
| `chat_service._normalize_metadata_filters`（`chat_service.py:318`） | 扩白名单到新字段，**同步加值校验**：`security_code` 走 `^\d{6}$`、`industry_*` 走申万 enum、`report_type` 走 enum |
| `chat_service.infer_metadata_filters`（`chat_service.py:368`） | prompt 同步扩字段 |
| `hybrid_retriever.output_fields` | 补新字段 |
| `schemas/chat.py::SourceInfo` | 补 `security_code` / `industry_l1` / `report_type` / `start_page` / `end_page` / `block_type` / `rerank_score` |
| `chat_service.merge_docs` | 排序由纯 score 改为 `(score, date desc)` tie-break |
| `ANSWER_PROMPT` | 强制每条结论标注报告日期与出处 |

`_build_filter` 本身**字段无关，不需要改**。

### 时效性

不加默认时间窗（避免静默丢旧报告）。`date` 做成前端可见 chip，可一键收窄。

### 性能

全库 hybrid 检索后在 CPU 上跑 `bge-reranker-v2-m3`，候选集随语料线性增长。千份级下需限定重排候选集 `top-N`。

## 4. 表格链路

### 存储

- 向量库只存「表头 + 首行」作索引（`block_type="table"`，`parent_id` 指向 `table_chunks` 的行 id）。
- 整表存 PG `table_chunks`（JSONB + 原始 Markdown）。

### 展开（关键：绕开 3000 陷阱）

`_expand_to_parents`（`hybrid_retriever.py:539-545`）只在父块长度 ∈ `[50, 3000]` 时替换子块内容，**超过 3000 的父块保留子块内容**（即表头 + 首行）。盈利预测表整表几乎必然 > 3000 字符，若走原路径，整表方案会被静默吃掉。因此：

- 新增 `_expand_tables`，走 `table_chunks` 仓储，**不受 `PARENT_MAX_CHARS` 约束**。
- `_expand_to_parents` 跳过 `block_type="table"` / `"image"` 的行，避免无效查询。

### 注入

- 单表上限 **8000 字符**，超限按行截断。
- 截断时在 sources 标 `truncated: true`，并在答案中提示"表已截断，可点击查看完整表"，**不静默删数据**。

## 5. 图片链路

现状：README:317-321 声称"自动提取嵌入图片 + 多模态模型生成中文描述"，但**代码里不存在**——`IMAGE_CAPTION_*` 8 个配置键（`config.py:113-120`）零消费者，全仓无 `fitz`/`pymupdf` 引用，MinerU 以 `image_analysis=False` 且只 dump markdown（`mineru_parser.py:54-61`）。唯一碰图片的两处是 `chunker.py:102`（那行坏代码）与 `document_service.py:687 _upload_extracted_images`。

本次让它变成真的：

- 用 `pymupdf` 抽取嵌入图片（顺带拿到页码），完整图入对象存储（`images/{stem}/{n}.png`）。
- 复用 `IMAGE_CAPTION_*`（qwen-vl-max）生成中文描述。
- caption 文本作为**独立 image chunk** 存入向量库作索引：`block_type="image"`、`image_key`、`content` = caption、`start_page`。
- 新增视觉模型**外发审计日志**（挂 `utils/audit.py`）。

## 6. 页码溯源

- chunk 存 `start_page` / `end_page`。
- 新增 `GET /api/documents/{source}/page/{page}`：用 `pymupdf`（已在 `requirements.txt:18`，全仓零引用）把该页渲染成 PNG。需先 `storage.download_to_path` 落地（兼容 OSS 后端）再渲染。
- 前端弹层展示，实现"3 步内跳回原文"。
- 原始 PDF 已存 `docs/{filename}`（`document_service.py:624`），**入库路径无需改动**。

**风险**：chunk→page 映射依赖 MinerU 的 `content_list`（现为 `f_dump_content_list=False`，需打开）。匹配不上的 chunk **页码留空，不乱填**。

## 7. 合规模块删除

现状：`compliance_service` 传 `collection_names=COMPLIANCE_CATEGORIES`（`['compliance_risk']`），而 `iter_active_kbs` 因 `exists` 键缺失恒回退 `[finance_kb]`，过滤后为空 → **零检索**，问答路径直接返回拒答。该模块已不可用。

### 删除

`api/routes/compliance.py`、`services/compliance_service.py`、`services/compliance_rules.py`、`schemas/compliance.py`、`frontend/src/views/ComplianceView.vue` 及其路由与菜单项、`citation_validator.py` 的 `validate_clause_citations` / `validate_finding_evidence`、合规相关测试与 eval 用例、`chunker._split_compliance_articles` 分支、`COMPLIANCE_*` 配置键、`main.py` 的合规路由挂载。

### 保留

- `CitationValidator.validate`（通用，研报要用）。
- `compliance_risk` 分类值（只是分类值；删了要动已 seed 的 PG 记录，收益为零）。

### 顺带清理

- `chat_service.iter_active_kbs` 的 `exists` 死代码（`chat_service.py:403-424`）。
- 从未被读取的 `ENABLE_METADATA_FILTER` 开关（仅出现在 `hybrid_retriever.py:281` 与 `milvus_kb.py:853` 的 docstring）。
- `scripts/seed_data.py` 中不在 `DOCUMENT_CATEGORIES` 常量表里的 `it_technology`。

## 8. 交付节奏

1. **修 `chunker.py:102` 的 `AttributeError`**。注意修法不是删掉 `images` 参数，而是让 `images` 真的有值（与第 5 节合并）。
2. 20 篇打通端到端（元数据 + 表格 + 图片 + 页码）。
3. **扩到 100 篇建 baseline**。20 篇时全库检索几乎必中，`hit_rate@5` 会接近 1.0，改造收益被掩盖。
4. 按 baseline 定点改造检索链路。
5. 全量入 500–5000 篇。

### 交给 baseline 判定的改造候选（不凭感觉开刀）

`ENABLE_LANGGRAPH`、rerank 开关与 `top-N`、`k`、`nprobe`、`ENABLE_HYDE`、`CHAT_ENABLE_QUERY_REWRITE`、`DYNAMIC_K`、语义分块。判据：研报语料上跑现有 L0/L1/L2/L3/L5 分层（`scripts/eval_recall.py`、`scripts/ab_rag.py`），哪层掉哪个指标就改哪层。

## 9. 验收

1. 100 篇语料、20 题 gold 集上达到第「硬阈值」表的全部数值。
2. 20 篇阶段端到端跑通：元数据抽取、整表展开、图片 caption 入索引、页码跳转四项均可演示。
3. baseline 结果落盘可复核（沿用 `scripts/results/` 约定）。
4. 合规模块删除后全量测试通过，无残留引用。

## 10. 假设（实施方填写，非需求方明确表述）

1. 题集构成 10 跨券商对比 + 8 单跳 + 2 负样本（见「目标用户与验收」）——需求方未逐项确认。
2. `ENABLE_VERSIONING` 保持 `false`，研报按 `source` 覆盖。
3. 研报 Markdown 的层级结构能被现有 `_inject_heading_structure` 正确处理，不做改动。
4. `compliance_risk` 之外的三个内置分类保留；研报入库归 `investment_research`。
5. `date` 沿用现有 `resolve_document_date`（文件名正则优先），不新增抽取。
6. "30min" 为估值而非实测。既然验收只看自动指标，该数字不进门槛，也不单独计时校准。

## 11. 风险与先决条件

| 风险 | 影响 | 处置 |
| --- | --- | --- |
| 未提供真实研报语料 | baseline 与全部抽取/表格/图片链路都无法验证 | 先决条件，第 1 步前必须就位 |
| `pymilvus` 实际装 3.0.0，compose pin 服务端 `v2.4.0` | 兼容性未知 | 起服务实测一次 |
| chunk→page 映射依赖 MinerU `content_list` | 页码可能缺失 | 匹配不上则留空，不乱填 |
| 元数据 LLM 抽取系统性错误 | 千份级全量重跑代价高（含视觉模型费用） | 先 20 篇 → 100 篇再全量 |
| 表格展开绕过 3000 上限后上下文膨胀 | 挤掉其它证据 | 单表 8000 字符上限 + 按行截断 |
| 行业 enum 校验未生效 | 过滤静默漏召 | enum 校验作为硬约束，单测覆盖 |

## 变更文件清单

| 层 | 文件 |
| --- | --- |
| Schema | `infrastructure/vector_store/milvus_kb.py`、`infrastructure/relational_db/table_store.py`（新增） |
| 解析 | `rag/ingestion/mineru_parser.py`、`rag/ingestion/chunker.py` |
| 抽取 | `rag/ingestion/`（新增元数据抽取模块）、`rag/ingestion/`（新增图片 caption 模块） |
| 检索 | `rag/retrieval/hybrid_retriever.py`、`rag/retrieval/parent_store.py` |
| 服务 | `services/chat_service.py`、`services/citation_validator.py`、`services/document_service.py` |
| API | `api/routes/documents.py`、`api/routes/chat.py`、`schemas/chat.py`、`schemas/document.py`、`main.py` |
| 配置 | `core/config.py`（删 `COMPLIANCE_*`、`ENABLE_METADATA_FILTER`） |
| 删除 | 合规模块 4 文件 + 前端视图 + 相关测试与 eval 用例 |
| 前端 | `views/ChatView.vue`、`views/DocumentsView.vue`、`api/index.js`、`router/index.js`、`App.vue` |
| 脚本 | `scripts/migrate_kb_schema.py` |
| 文档 | `README.md` |
