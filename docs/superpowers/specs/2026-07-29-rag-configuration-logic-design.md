# RAG 配置逻辑完善设计

## 目标

修复项目根目录 `.env` 加载不明确、中文 tokenizer 选择逻辑容易误解、BGE reranker 缺少全局开关，以及各优化开关布尔值解析不一致的问题。稠密向量检索继续使用 DashScope `text-embedding-v2`，不修改 Milvus 向量维度或已有 collection。

## 配置来源与优先级

应用从 `finance_rag/config.py` 所在位置推导仓库根目录，并显式加载根目录下的 `.env`。`.env.example` 仅作为模板，不参与运行时加载。

配置优先级固定为：

1. 启动进程中已经存在的环境变量；
2. 仓库根目录 `.env`；
3. 代码默认值。

加载 `.env` 时不覆盖已有进程环境变量。配置仍在模块首次导入时解析，因此修改环境变量后需要重启服务。

## 统一布尔值解析

`config.py` 提供单一布尔解析函数，供所有功能开关使用。解析不区分大小写，并忽略首尾空白：

- 真值：`true`、`1`、`yes`、`on`
- 假值：`false`、`0`、`no`、`off`

变量未设置时使用调用方提供的默认值。非空但不属于上述集合的值属于配置错误，异常必须包含变量名与原始值，避免拼写错误被静默解释为关闭。

统一迁移的开关包括：

- `USE_ZH_TOKENIZER`
- `ENABLE_RERANKER`
- `CHAT_ENABLE_QUERY_REWRITE`
- `ENABLE_SMART_CHUNKER`
- `ENABLE_MULTI_STAGE_RETRIEVAL`
- `ENABLE_METADATA_FILTER`
- `ENABLE_FINANCIAL_EXPERT_PROMPT`
- `ENABLE_CITATION_VALIDATION`

## 中文 tokenizer 选择

中文开关优先级高于显式 tokenizer 名称：

- `USE_ZH_TOKENIZER=true` 时，`DOCLING_CHUNK_TOKENIZER` 强制为 `BAAI/bge-large-zh-v1.5`；即使环境中另设 `DOCLING_CHUNK_TOKENIZER` 也忽略它。
- `USE_ZH_TOKENIZER=false` 时，读取 `DOCLING_CHUNK_TOKENIZER`；未设置则使用 `sentence-transformers/all-MiniLM-L6-v2`。

该模型只用于 Docling 切块 tokenizer，不替换 DashScope embedding。

## BGE reranker 控制

新增 `ENABLE_RERANKER`，默认值为 `true`，保持聊天 API 当前默认执行 BGE 重排序的行为。

最终执行条件为：

```text
effective_use_rerank = ENABLE_RERANKER and request_use_rerank
```

全局关闭时，任何请求都不能加载或调用 BGE reranker；全局开启时，请求仍可通过 `use_rerank=false` 跳过重排序。普通混合检索与多阶段检索必须使用相同的有效值，SSE 仅在有效重排序开启时发出 `reranking` 状态。

`RERANKER_MODEL`、`RERANKER_DEVICE` 与延迟加载机制保持不变。运行时模型加载或推理失败仍回退到原始排序并记录警告，不把可选增强组件故障升级为整个问答请求失败。

## 模块边界

- `finance_rag/config.py`：负责根目录 `.env` 加载、类型解析、默认值和派生配置。
- `finance_rag/chat.py`：组合全局与请求级 reranker 开关，并把唯一的有效值传递给检索链路。
- `finance_rag/retrieve.py`：继续负责 BGE 模型延迟加载和失败回退，不重复读取环境变量。
- `.env.example`：记录所有支持的配置、默认行为和优先级，但不包含真实密钥。

现有模块继续导入配置常量，避免本次范围演变成全项目配置对象重构。

## 错误处理

- 非法布尔配置在 `config.py` 初始化阶段快速失败，并明确指出错误变量。
- 缺少 `.env` 本身不是错误；系统仍可完全通过进程环境变量运行。
- `.env.example` 不会被自动当作配置使用。
- BGE reranker 的下载、加载或推理异常继续记录 warning 并回退。

## 测试策略

新增独立配置测试，通过隔离环境并重新导入配置模块验证：

1. 项目根目录 `.env` 能被显式定位，且不覆盖进程变量；
2. 合法布尔值及大小写、空白均正确解析；
3. 非法布尔值快速失败并包含变量名和值；
4. 中文开关开启时强制选择 `BAAI/bge-large-zh-v1.5`；
5. 中文开关关闭时显式 tokenizer 生效，缺省时回到 MiniLM；
6. 各优化开关均通过统一解析路径得到预期值。

新增聊天链路测试验证：

1. 全局关闭时，请求传入 `use_rerank=true` 仍不触发重排序或 reranking 状态；
2. 全局开启且请求开启时触发重排序；
3. 全局开启但请求关闭时不触发重排序；
4. 普通与多阶段检索接收到相同的有效开关值。

测试不下载模型、不调用 DashScope、DeepSeek 或 Milvus；外部边界使用最小替身，断言应用自身传递出的有效行为。

## 非目标

- 不把稠密向量模型替换为 BGE。
- 不修改 `EMBEDDING_DIM=1536`。
- 不删除、迁移或重建 Milvus collection。
- 不引入 Pydantic Settings 或新的运行时依赖。
- 不改变 BGE reranker 失败时的降级策略。
