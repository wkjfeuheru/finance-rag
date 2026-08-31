# 生产部署配置调整计划

## Summary

目标是将当前 Docker Compose 单机部署从开发配置调整为可上线的生产配置，重点覆盖：镜像可构建、密钥与鉴权、服务网络暴露、应用健康检查、对象存储与数据库连接、依赖可复现性，以及生产环境验证。

本计划默认部署拓扑为：公网流量先经过云负载均衡或 Nginx/Caddy 等反向代理，只有应用 HTTP 入口对外开放，PostgreSQL、Redis、Milvus、etcd、MinIO 仅在 Docker 内部网络或受控管理网络可访问。

## Current State Analysis

1. `Dockerfile` 的后端阶段复制 `config/`，但仓库根目录当前没有该目录；镜像构建会在此步骤失败。当前实际配置入口是 `finance_rag/src/core/config.py`。
2. `docker-compose.yml` 内置了 MinIO、PostgreSQL 默认账号密码，Redis 无认证，并将多个基础设施端口映射到宿主机；应用端口也直接发布，未体现反向代理边界。
3. `config.py` 在缺少 `JWT_SECRET` 时只告警，`dependencies.py` 随后将请求识别为匿名用户并放行，生产环境存在鉴权绕过风险。
4. Compose 中关键环境变量缺失时会传入空值，`ALLOWED_ORIGINS` 默认仍为 localhost；应用没有生产环境标识和关键配置启动校验。
5. `main.py` 公开 `/metrics`，健康检查返回 Milvus 异常信息并执行建集合操作；未配置生产主机白名单或明确的代理边界。
6. Dockerfile 默认 root 用户、单 Uvicorn worker，镜像基础标签和后端依赖未完全锁定；`.dockerignore` 未排除全部数据和测试目录。
7. 当前对象存储仅支持本地文件和 S3/MinIO，未提供阿里云 OSS 后端；生产需要改为唯一使用阿里云 OSS。PostgreSQL 连接池没有生产连接上限、回收和超时配置。

## Proposed Changes

### 1. 修复并收紧容器构建

**文件：`Dockerfile`**

- 删除对不存在的 `config/` 的复制，或仅在确认新增并实际使用统一配置目录后再保留；本计划采用当前 `finance_rag/src/core/config.py`，因此删除该复制步骤。
- 固定 Node、Python 基础镜像到明确的小版本，避免可变标签导致构建结果漂移。
- 生产镜像仅安装运行依赖；保留当前前端构建阶段，但避免将测试/评估依赖带入运行镜像。
- 增加非 root 运行用户，并确保 `/app/files`、`/app/onnx_cache`、`/app/data/state` 对该用户可写。
- 启动命令明确生产参数；由 Compose 或编排平台决定 worker 数量，至少配置健康检查和优雅停止策略。

### 2. 增加生产配置边界与启动校验

**文件：`finance_rag/src/core/config.py`**

- 增加明确的 `APP_ENV`，生产部署设置为 `production`。
- 生产环境启动时强制校验 `JWT_SECRET`、`JWT_ADMIN_PASSWORD`、`DEEPSEEK_API_KEY`、`DATABASE_URL`、`MILVUS_URI`、`REDIS_URL` 等必需配置；缺失或仍为开发默认值时直接失败。
- 对 `JWT_SECRET` 增加最小长度/随机性要求；生产环境不允许空密钥。
- 将对象存储后端固定为 `oss`；生产对象存储使用阿里云 OSS，要求配置 OSS endpoint、bucket、region、AccessKey 或 RAM 临时凭证，并禁止使用默认密钥。
- 删除本地、S3/MinIO 的生产回退路径；应用配置缺失或 OSS SDK 不可用时直接启动失败。
- 生产环境要求 `ALLOWED_ORIGINS` 为实际 HTTPS 前端域名，禁止通配来源。
- 保持开发环境可使用本地默认值，但通过 `APP_ENV` 显式区分，避免开发兼容逻辑泄漏到生产。

**文件：`finance_rag/src/api/dependencies.py`**

- 删除或限制“缺少 JWT_SECRET 时放行”的逻辑，使生产环境永远要求 Bearer token。
- 更新模块说明和依赖行为，确保鉴权失败统一返回 401。

### 3. 重构生产 Compose 编排

**文件：`docker-compose.yml`**

- 所有账号、密码、API key、JWT 配置改为从部署平台 Secret 或生产 `.env` 注入；不得保留 `minioadmin`、`finance` 等可预测默认凭据。
- PostgreSQL 的 `DATABASE_URL` 改为引用生产数据库用户名、密码、主机和 TLS 参数；Redis 配置密码并在 `REDIS_URL` 中携带认证信息。
- 生产环境移除 MinIO 服务及其卷、端口和依赖；应用唯一通过阿里云 OSS endpoint 访问远程对象存储。
- 将 `STORAGE_BACKEND=oss`、`OSS_ENDPOINT`、`OSS_BUCKET`、`OSS_REGION`、`OSS_ACCESS_KEY_ID`、`OSS_ACCESS_KEY_SECRET` 注入应用；优先使用 RAM 最小权限账号或临时凭证，OSS endpoint 使用 HTTPS。
- 删除 etcd、MinIO、PostgreSQL、Redis、Milvus 的宿主机端口映射；如确需运维访问，仅绑定 `127.0.0.1` 或独立管理网络。
- 从 Compose 和应用配置中彻底删除 MinIO、S3 endpoint、S3 bucket、S3 access key、S3 secret key、S3 secure 等变量和实现。
- 应用仅通过反向代理需要的端口对外提供服务，并配置 `restart: unless-stopped`、健康检查、停止宽限时间和合理资源限制。
- 为基础设施补充依赖健康条件和网络隔离；为应用设置 `APP_ENV=production`、真实 `ALLOWED_ORIGINS`、生产存储和模型缓存路径。
- 明确持久卷备份范围：PostgreSQL、Milvus、MinIO、应用上传文件、模型缓存和状态文件分别纳入备份/恢复策略。

### 4. 收紧应用 HTTP 安全与探针

**文件：`finance_rag/src/main.py`**

- 从生产环境配置读取受信任主机列表，增加 `TrustedHostMiddleware`；TLS 由反向代理终止时，按实际代理拓扑配置 HTTPS/代理头策略。
- 将 CORS 限制为生产前端域名、必要 HTTP 方法和必要请求头。
- `/metrics` 仅允许内网监控或由反向代理访问控制保护，避免公网匿名暴露。
- 将健康检查拆分为无副作用的存活探针和内部依赖就绪探针；对外响应不返回 Milvus、数据库等内部异常文本。
- 确保健康探针不执行 `ensure_collection()` 这类可能修改状态的初始化操作。

### 5. 更新生产环境变量模板

**文件：`.env.example`**

- 将 JWT 密钥、管理员密码、API key 和数据库连接改为空值或明确的必填占位符，避免复制模板后直接使用已知凭据。
- 增加 `APP_ENV=production`、`ALLOWED_ORIGINS=https://<production-domain>`、生产 `DATABASE_URL`、带密码的 `REDIS_URL`、`STORAGE_BACKEND=oss`、`OSS_ENDPOINT`、`OSS_BUCKET`、`OSS_REGION`、`OSS_ACCESS_KEY_ID`、`OSS_ACCESS_KEY_SECRET` 等上线必填项。
- 删除所有 MinIO/S3 配置示例；将 Milvus 地址改为 Compose 服务名或实际生产地址，而不是 localhost。
- 明确生产模型缓存、上传目录、日志级别、JWT 有效期和文件上传限制。
- 保留开发示例，但在注释中区分开发与生产，避免将本模板误当作 Secret 存储。

### 6. 缩小构建上下文并统一依赖来源

**文件：`.dockerignore`**

- 增加排除 `.venv/`、`data/`、`files/`、`tests/`、`scripts/results/` 等本地数据、测试和构建产物目录，避免敏感金融文档进入 Docker 构建上下文。

**文件：`requirements.txt`、`pyproject.toml`**

- 确定唯一的生产依赖来源，建议以 `pyproject.toml` 的基础依赖为准，开发和评估依赖放入可选组。
- 生成并提交锁定版本或带 hash 的生产依赖清单；生产镜像不安装 pytest、ragas、datasets 等非运行依赖。
- 对当前未固定的 `langchain-deepseek` 及其他 `>=` 依赖建立可复现版本约束，并在 CI 中执行干净环境构建。

### 7. 接入阿里云 OSS 并补充数据库生产参数

**文件：`finance_rag/src/infrastructure/storage/oss.py`（新增）**

- 基于 `oss2` SDK 实现现有 `StorageBackend` 接口：上传、下载、文本下载、删除、存在性检查、哈希获取、下载到本地和同步删除。
- 使用 `OSS_ENDPOINT`、`OSS_BUCKET`、`OSS_REGION`、`OSS_ACCESS_KEY_ID`、`OSS_ACCESS_KEY_SECRET` 配置；endpoint 强制 HTTPS，bucket 由部署基础设施预创建。
- 应用账号仅授予目标 bucket 的对象读写/删除权限，不授予创建 bucket、修改策略等管理权限；优先使用 RAM 最小权限或临时凭证。
- 上传对象按 OSS 生产策略设置内容类型、服务端加密和必要元数据；保留 OSS 错误到领域 `StorageError` 的映射。

**文件：`finance_rag/src/infrastructure/storage/__init__.py`**

- 将存储工厂固定为 `AliyunOSSStorageBackend`，移除 local/S3 分支和本地回退逻辑。
- OSS SDK 缺失或初始化失败时直接报错，不得静默切换到其他存储。

**文件：`finance_rag/src/infrastructure/storage/s3.py`（删除）**

- 删除 S3/MinIO 实现，确保代码库不再包含 boto3 对象存储路径。

**文件：`pyproject.toml`、`requirements.txt`**

- 增加 `oss2` 运行依赖并移除 `boto3`；清理仅服务于 S3/MinIO 的依赖和配置。

- bucket 在部署基础设施阶段预创建并设置加密、版本、生命周期和访问策略；应用启动时只检查可访问性。
- 上传对象时按生产策略设置内容类型、服务端加密、生命周期和必要元数据。

**文件：`finance_rag/src/infrastructure/relational_db/postgres.py`、`finance_rag/src/core/config.py`**

- 增加可配置的连接池大小、最大溢出、连接回收时间、连接超时和执行超时。
- 生产数据库连接启用 TLS，并依据数据库服务商要求配置证书校验参数。

## Required Production Values

部署平台至少需要提供以下真实值，并通过 Secret 管理：

- `APP_ENV=production`
- `JWT_SECRET`：随机生成，至少 32 字符，禁止复用示例值
- `JWT_ADMIN_USERNAME`、`JWT_ADMIN_PASSWORD`
- `DEEPSEEK_API_KEY`，以及实际使用的模型名
- `DATABASE_URL`：生产 PostgreSQL，使用强密码和 TLS
- `REDIS_URL`：带认证信息，且仅内网可达
- `MILVUS_URI`、必要时的 `MILVUS_TOKEN`
- `ALLOWED_ORIGINS`：真实 HTTPS 前端域名
- `STORAGE_BACKEND=oss`
- `OSS_ENDPOINT`：阿里云 OSS HTTPS endpoint
- `OSS_BUCKET`、`OSS_REGION`
- `OSS_ACCESS_KEY_ID`、`OSS_ACCESS_KEY_SECRET`，或对应的 RAM 临时凭证
- `HF_ENDPOINT`、模型缓存卷和持久化上传目录

## Assumptions & Decisions

- 采用 Docker Compose 单主机部署，反向代理和 TLS 证书不由当前应用容器负责。
- 当前实际代码结构以 `finance_rag/` 为准，不新增与其重复的根目录 `config/`。
- 不在本阶段加入完整用户体系、密钥轮换平台、Kubernetes 清单或云厂商专属资源；只完成当前项目上线所需的配置收紧和运行参数。
- 数据迁移、Milvus 集合兼容性、对象存储历史数据迁移和备份恢复演练属于上线验证的一部分，不通过默认配置自动处理。

## Model Deployment Plan

### Model Inventory

当前项目包含两类本地模型：

1. **嵌入模型**：默认 `BAAI/bge-large-zh-v1.5`，向量维度为 1024。应用首次使用时通过 `optimum` 导出 ONNX，并使用 `onnxruntime` 动态 INT8 量化；后续从 `ONNX_CACHE_DIR` 加载，固定使用 `CPUExecutionProvider`。
2. **重排序模型**：默认 `BAAI/bge-reranker-v2-m3`，通过 `sentence-transformers.CrossEncoder` 加载；由 `RERANKER_DEVICE` 选择 CPU 或 CUDA，当前应用启动生命周期中会调用 `warmup()` 预热。

### Recommended Production Topology

- 生产第一阶段采用 CPU 推理，与当前 ONNX INT8 嵌入实现保持一致；`RERANKER_DEVICE=cpu`，避免将 CUDA 配置到不含 GPU 的镜像。
- 将模型下载、ONNX 导出、INT8 量化和 tokenizer 缓存从运行时迁移到镜像构建阶段或一次性模型准备阶段。业务容器启动时只加载已准备好的本地模型，禁止多个实例同时执行首次导出。
- 构建一个独立的模型准备步骤：联网下载 Hugging Face 模型，完成嵌入模型 ONNX 导出/量化和重排序模型缓存，然后执行一次离线加载与推理检查。
- 将模型缓存制作成版本化镜像层或只读模型卷；不要把模型缓存写入临时容器层。`HF_HOME`、`ONNX_CACHE_DIR` 和 `HF_ENDPOINT` 必须在构建/准备阶段与运行阶段保持一致。
- 生产应用容器挂载模型缓存只读目录，单独挂载可写的上传目录和运行状态目录；模型目录不允许业务进程修改。
- 由于 BGE 重排序模型约占用较大内存，默认采用单 worker；如需提高并发，优先横向扩容并逐实例评估内存，而不是盲目增加 Uvicorn workers。每个进程都会独立持有模型实例。

### CPU Deployment

- 镜像安装 `onnxruntime`，不要安装 `onnxruntime-gpu`；保持 `RERANKER_DEVICE=cpu`。
- 根据机器核数配置 `OMP_NUM_THREADS`、`MKL_NUM_THREADS` 和嵌入 `EMBED_BATCH_SIZE`，通过压测确定值，避免多个请求造成 CPU 过载。
- 设置模型缓存目录为持久化路径，例如 `/app/model_cache`；缓存卷在发布前预热完成。
- 启动探针必须等待模型加载/预热完成后再将实例加入流量；模型加载失败应使实例进入未就绪状态，而不是接受请求后才失败。

### GPU Deployment

- GPU 方案需要单独构建 CUDA 基础镜像，安装与 CUDA/cuDNN 版本匹配的 PyTorch、`sentence-transformers` 和 `onnxruntime-gpu`；不能直接复用 CPU 镜像并仅修改 `RERANKER_DEVICE=cuda`。
- GPU 节点由编排平台显式分配，例如每个容器绑定一张 GPU；设置 `RERANKER_DEVICE=cuda`，并在启动检查中验证 `torch.cuda.is_available()`。
- 当前嵌入器仍固定使用 `CPUExecutionProvider`，因此 GPU 只会加速重排序模型；是否采用 GPU 需要以实际 QPS、延迟和显存占用压测结果决定。
- GPU 方案单独维护镜像标签、驱动/CUDA 兼容矩阵和回滚版本，不在 CPU 生产镜像中隐式切换。

### Model Versioning and Rollback

- 固定 `EMBEDDING_MODEL`、`EMBEDDING_DIM`、`RERANKER_MODEL` 的版本或 commit，禁止生产环境使用未固定的模型 latest 状态。
- 嵌入模型变更会改变向量空间；变更 `EMBEDDING_MODEL` 或 `EMBEDDING_DIM` 前必须新建 Milvus collection，完成全量重嵌入、抽样检索评估和切换，再保留旧 collection 供回滚。
- 重排序模型变更不改变 Milvus 向量维度，但仍需进行离线评估和线上灰度；模型缓存、镜像和配置必须可回滚到上一版本。
- 发布记录至少包含模型标识、模型文件校验值、量化方式、运行设备、镜像版本和评估结果。

### Model Operations and Monitoring

- 监控模型加载耗时、首次请求延迟、推理耗时、CPU/内存/显存、模型缓存命中率和下载失败次数。
- 对模型下载和准备步骤设置超时、重试及失败即终止发布策略；不要在生产请求路径中静默切换到未经评估的模型。
- 将模型准备、应用启动和业务请求日志分开，避免下载/量化日志淹没业务告警。
- 对嵌入向量维度、tokenizer、ONNX 输入输出和重排序结果做启动自检；自检失败时实例不可就绪。

### Files and Configuration

- `Dockerfile`：增加模型准备阶段或调用独立准备脚本，固定模型缓存路径，并区分 CPU/GPU 镜像。
- `docker-compose.yml`：增加只读模型缓存卷、模型准备依赖、CPU 线程参数、单 worker 默认值和模型健康/就绪检查。
- `finance_rag/src/core/config.py`：增加模型版本、离线模式、缓存目录、模型准备策略、设备和线程数配置，并在生产环境校验模型配置。
- `finance_rag/src/infrastructure/vector_store/onnx_embedder.py`：支持生产离线加载，禁止运行时自动下载或导出；增加模型文件和向量维度启动校验。
- `finance_rag/src/rag/retrieval/hybrid_retriever.py`：保留 CPU/CUDA 设备选择，但生产启动时应将预热失败视为不可就绪，而不是仅记录告警。
- `finance_rag/src/main.py`：将模型预热结果纳入 readiness 状态，区分存活和就绪探针。
- `.env.example`：增加 `MODEL_CACHE_DIR`、`HF_HOME`、`HF_HUB_OFFLINE`、`TRANSFORMERS_OFFLINE`、`RERANKER_DEVICE`、`EMBED_BATCH_SIZE`、`OMP_NUM_THREADS`、`MKL_NUM_THREADS` 和固定模型版本示例。

## Verification Steps

1. 使用生产环境变量执行干净的 Docker 构建，确认 Dockerfile 不再引用不存在路径，且容器以非 root 用户运行。
2. 在缺少 `JWT_SECRET`、管理员密码或 API key 时启动应用，确认生产环境直接失败；使用正确 Secret 后启动成功。
3. 执行 `docker compose config`，确认没有开发默认凭据、localhost 服务地址或不必要的基础设施宿主机端口暴露。
4. 启动完整栈并检查应用存活、依赖就绪、登录、带 token 访问和无 token 访问行为。
5. 从公网入口验证 CORS、HTTPS、Host 白名单、`/metrics` 访问控制以及健康接口不泄露内部异常。
6. 验证 PostgreSQL、Redis、Milvus、阿里云 OSS 的 TLS/认证连接，确认 OSS bucket 已预创建且应用账号无法执行 bucket 管理操作；上传下载一份测试文档并验证对象权限、加密和删除行为。
7. 执行备份与恢复演练，确认数据库、向量数据、上传文件和应用状态文件能够恢复。
8. 运行后端测试、静态检查和镜像漏洞扫描，记录模型首次加载、接口延迟、资源使用和日志告警结果。
