"""项目配置中心。

配置优先级（高 → 低）：
1. 启动进程中已存在的环境变量；
2. 仓库根目录 ``.env``（不覆盖已有进程变量）；
3. 下方代码默认值。

修改环境变量后需重启服务生效。``.env.example`` 仅作模板，不参与运行时加载。
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


# ==========================================================================
# 1. 路径常量与环境加载
# ==========================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_project_env(project_root: Path = PROJECT_ROOT) -> None:
    """加载仓库根目录 ``.env``，不覆盖已有进程环境变量。"""
    load_dotenv(dotenv_path=project_root / ".env", override=False)


# ==========================================================================
# 2. 类型解析工具
# ==========================================================================

_TRUE_VALUES = frozenset({"true", "1", "yes", "on"})
_FALSE_VALUES = frozenset({"false", "0", "no", "off"})


def env_bool(name: str, default: bool) -> bool:
    """解析布尔环境变量，不区分大小写。

    非法值快速失败并包含变量名与原始值。
    """
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


def env_positive_float(name: str, default: float) -> float:
    """解析正浮点数环境变量，值必须 > 0。"""
    value = float(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"Environment variable {name} must be greater than 0")
    return value


load_project_env()

# 运行环境配置
APP_ENV = os.getenv("APP_ENV", "development").strip().lower()

# LLM 服务配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek:deepseek-v4-pro")
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "120"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "1"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))

# 嵌入服务与向量数据库
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))
MODEL_CACHE_DIR = os.getenv(
    "MODEL_CACHE_DIR", str(Path.home() / ".cache" / "finance_rag" / "models")
)
ONNX_CACHE_DIR = os.getenv("ONNX_CACHE_DIR", str(Path(MODEL_CACHE_DIR) / "onnx"))
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "25"))
HF_ENDPOINT = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")
HF_HOME = os.getenv("HF_HOME", str(Path(MODEL_CACHE_DIR) / "hf_cache"))
MODEL_OFFLINE = env_bool("MODEL_OFFLINE", APP_ENV == "production")
HF_HUB_OFFLINE = env_bool("HF_HUB_OFFLINE", MODEL_OFFLINE)
TRANSFORMERS_OFFLINE = env_bool("TRANSFORMERS_OFFLINE", MODEL_OFFLINE)
if HF_ENDPOINT:
    os.environ.setdefault("HF_ENDPOINT", HF_ENDPOINT)
os.environ.setdefault("HF_HOME", HF_HOME)
os.environ.setdefault("HF_HUB_OFFLINE", "1" if HF_HUB_OFFLINE else "0")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1" if TRANSFORMERS_OFFLINE else "0")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "")
MILVUS_TIMEOUT_SECONDS = float(os.getenv("MILVUS_TIMEOUT_SECONDS", "15"))
KB_COLLECTION_NAME = os.getenv("KB_COLLECTION_NAME", "finance_kb")
MILVUS_NLIST = int(os.getenv("MILVUS_NLIST", "128"))
MILVUS_NPROBE = int(os.getenv("MILVUS_NPROBE", "10"))
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg://finance:finance@localhost:5432/finance_rag")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
REDIS_CACHE_PREFIX = os.getenv("REDIS_CACHE_PREFIX", "finance_rag")
REDIS_TIMEOUT_SECONDS = float(os.getenv("REDIS_TIMEOUT_SECONDS", "2"))

# 文档图片处理
IMAGE_CAPTION_BASE_URL = os.getenv("IMAGE_CAPTION_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
IMAGE_CAPTION_MODEL = os.getenv("IMAGE_CAPTION_MODEL", "qwen-vl-max")
# 未单独配置时复用 DashScope key（与 QUERY_REWRITE_API_KEY 同一约定）
IMAGE_CAPTION_API_KEY = os.getenv("IMAGE_CAPTION_API_KEY", "") or DASHSCOPE_API_KEY
IMAGE_CAPTION_TIMEOUT_SECONDS = float(os.getenv("IMAGE_CAPTION_TIMEOUT_SECONDS", "30"))
IMAGE_CAPTION_MAX_IMAGES = int(os.getenv("IMAGE_CAPTION_MAX_IMAGES", "10"))
IMAGE_CAPTION_MAX_RETRIES = int(os.getenv("IMAGE_CAPTION_MAX_RETRIES", "2"))
IMAGE_CAPTION_MIN_DIM_PX = int(os.getenv("IMAGE_CAPTION_MIN_DIM_PX", "80"))
IMAGE_CAPTION_MIN_BYTES = int(os.getenv("IMAGE_CAPTION_MIN_BYTES", "5000"))
IMAGE_CAPTION_PROMPT = os.getenv("IMAGE_CAPTION_PROMPT", "这是一份金融文档中嵌入的图片。请用中文详细描述图片内容：如果是图表，说明图表类型、标题、关键数据与趋势；如果是表格，转述主要行列内容；如果是印章/签名/截图/扫描件，说明其性质与可见文字。只输出描述本身，不超过 200 字。")

# 文档解析：MinerU 是唯一的二进制文档解析器，通过 Python API 调用
MINERU_METHOD = os.getenv("MINERU_METHOD", "auto")
MINERU_BACKEND = os.getenv("MINERU_BACKEND", "pipeline")
MINERU_FORMULA_ENABLE = env_bool("MINERU_FORMULA_ENABLE", True)
MINERU_TABLE_ENABLE = env_bool("MINERU_TABLE_ENABLE", True)
MINERU_OUTPUT_DIR = os.getenv("MINERU_OUTPUT_DIR", str(PROJECT_ROOT / "assets" / "state" / "mineru"))
MINERU_MAX_CONCURRENCY = max(1, int(os.getenv("MINERU_MAX_CONCURRENCY", "2")))
MINERU_KEEP_ARTIFACTS_ON_ERROR = env_bool("MINERU_KEEP_ARTIFACTS_ON_ERROR", False)
MINERU_SUPPORTED_EXTENSIONS = tuple(
    item.strip().lower() for item in os.getenv(
        "MINERU_SUPPORTED_EXTENSIONS", ".pdf,.docx,.pptx,.xlsx,.xls,.md,.txt"
    ).split(",") if item.strip()
)
CLEANER_RULES_VERSION = os.getenv("CLEANER_RULES_VERSION", "mineru-cleaner-v1")
CHUNK_RULES_VERSION = os.getenv("CHUNK_RULES_VERSION", "hierarchical-v2")
EMBEDDING_VERSION = os.getenv("EMBEDDING_VERSION", EMBEDDING_MODEL)
OBJECT_EVENT_SECRET = os.getenv("OBJECT_EVENT_SECRET", "")
OBJECT_EVENT_PREFIX = os.getenv("OBJECT_EVENT_PREFIX", "docs/")
OBJECT_EVENT_MAX_RETRIES = max(0, int(os.getenv("OBJECT_EVENT_MAX_RETRIES", "5")))
VERSION_RETENTION_DAYS = max(0, int(os.getenv("VERSION_RETENTION_DAYS", "0")))

# 文档切块配置
CHUNK_TOKENIZER = os.getenv("CHUNK_TOKENIZER", "BAAI/bge-large-zh-v1.5")
DOCLING_CHUNK_MAX_TOKENS = int(os.getenv("DOCLING_CHUNK_MAX_TOKENS", "512"))
PARENT_MAX_TOKENS = int(os.getenv("PARENT_MAX_TOKENS", "2000"))
CHILD_OVERLAP_TOKENS = int(os.getenv("CHILD_OVERLAP_TOKENS", "50"))
ENABLE_SEMANTIC_CHUNKER = env_bool("ENABLE_SEMANTIC_CHUNKER", False)
SEMANTIC_BREAK_THRESHOLD = float(os.getenv("SEMANTIC_BREAK_THRESHOLD", "0.45"))
SEMANTIC_MIN_TOKENS = int(os.getenv("SEMANTIC_MIN_TOKENS", "100"))
SEMANTIC_EMBED_MODEL = os.getenv("SEMANTIC_EMBED_MODEL", "BAAI/bge-small-zh-v1.5")
SEMANTIC_DEVICE = os.getenv("SEMANTIC_DEVICE", "cpu")
ENABLE_VERSIONING = env_bool("ENABLE_VERSIONING", False)
MAX_VERSIONS_PER_DOC = int(os.getenv("MAX_VERSIONS_PER_DOC", "0"))


def _parse_dynamic_k_map() -> dict[str, dict[str, int]]:
    """解析动态 K 档位配置（JSON），非法时回退默认档位。"""
    raw = os.getenv("DYNAMIC_K_MAP", '{"simple": {"k": 3, "rerank_top_n": 2}, "normal": {"k": 8, "rerank_top_n": 4}, "complex": {"k": 20, "rerank_top_n": 6}}')
    fallback = {"simple": {"k": 3, "rerank_top_n": 2}, "normal": {"k": 8, "rerank_top_n": 4}, "complex": {"k": 20, "rerank_top_n": 6}}
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and all(isinstance(v, dict) and "k" in v for v in parsed.values()):
            return {k: {"k": int(v["k"]), "rerank_top_n": int(v.get("rerank_top_n", v["k"]))} for k, v in parsed.items()}
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    return fallback


DYNAMIC_K = env_bool("DYNAMIC_K", False)
DYNAMIC_K_MAP = _parse_dynamic_k_map()
ENABLE_HYDE = env_bool("ENABLE_HYDE", False)
HYDE_WEIGHT = float(os.getenv("HYDE_WEIGHT", "0.5"))
RRF_K = int(os.getenv("RRF_K", "60"))
RUNTIME_STATE_DIR = PROJECT_ROOT / "assets" / "state"
ENABLE_RERANKER = env_bool("ENABLE_RERANKER", True)
# 启动预热：true 时在后台线程预热「重依赖导入 + 嵌入/重排序模型」，
# 不阻塞 uvicorn 绑定端口；就绪性由 /api/health/ready 报告。
# false 时完全跳过预热（模型在首次调用时惰性加载，首个请求会明显变慢）。
WARMUP_ENABLED = env_bool("WARMUP_ENABLED", True)
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
RERANKER_DEVICE = os.getenv("RERANKER_DEVICE", "cpu")
RERANKER_CLIFF_THRESHOLD = float(os.getenv("RERANKER_CLIFF_THRESHOLD", "0.35"))
# 断崖截断的下限：检测到断崖时**至少保留**这么多条送进重排序。
#
# 判据是「相邻粗排分数相对落差 > threshold」才截断。阈值 0.35 最初按重排分数
# 的陡落差标定（首条 0.947、末条 0.005）。粗排分（RRF / 稠密）相邻落差通常更小，
# 只有头部和尾部明显拉开时才会触发。下限取 3，避免一次截到只剩 1 条候选。
RERANKER_CLIFF_MIN_RESULTS = max(1, int(os.getenv("RERANKER_CLIFF_MIN_RESULTS", "3")))
MILVUS_FINGERPRINT_PATH = os.getenv("MILVUS_FINGERPRINT_PATH", str(RUNTIME_STATE_DIR / ".milvus_fingerprint"))
CHAT_TOP_K = int(os.getenv("CHAT_TOP_K", "5"))
CHAT_RERANK_TOP_K = int(os.getenv("CHAT_RERANK_TOP_K", "3"))
CHAT_ENABLE_QUERY_REWRITE = env_bool("CHAT_ENABLE_QUERY_REWRITE", False)
ENABLE_LANGGRAPH = env_bool("ENABLE_LANGGRAPH", False)
LANGGRAPH_MAX_REFLECT_ROUNDS = min(3, max(1, int(os.getenv("LANGGRAPH_MAX_REFLECT_ROUNDS", "2"))))
TENANT_ID = os.getenv("TENANT_ID", "default")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", str(PROJECT_ROOT / "files"))
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "20"))
DOCUMENT_PARSE_WORKERS = int(os.getenv("DOCUMENT_PARSE_WORKERS", "4"))
PDF_PARSE_TIMEOUT_SECONDS = env_positive_float("PDF_PARSE_TIMEOUT_SECONDS", 300)

# ---------------------------------------------------------------------------
# 入库流水线（进程单例常驻队列 + 分阶段 Executor）
# ---------------------------------------------------------------------------
# 常驻四段流水线：文档上传完成即入队（submit），入队即返回，不等同批其它文档。
#
#   submit ──► in_queue ──► [解析 worker] ──► parse_queue ──► [切块 worker]
#                                                                  │
#   写库 ◄── [写入 worker] ◄── write_queue ◄── [Embedding worker] ◄── chunk_queue
#
# 各阶段执行体与并发上限：
#   解析       ThreadPoolExecutor，worker 数 == MINERU_MAX_CONCURRENCY
#              （MinerU 加载 GB 级 ONNX/GPU 权重，进程池化会成倍放大内存）
#   切块       ProcessPoolExecutor（spawn），worker 数 == 进程池 worker 数
#              （清洗/SimHash/切分是纯 Python，受 GIL 限制，进程池才是正解）
#   Embedding  专用 ThreadPoolExecutor（ONNX 为 C 层；注意 OnnxEmbedder
#              内置类级推理锁，实际 session.run 仍串行）
#   写入       多消费者 + 两阶段（锁外存储上传可并行，锁内向量库原子区串行）
#
# 关闭时回退到原有的「逐文件串行」入库路径。
INGEST_PIPELINE_ENABLED = env_bool("INGEST_PIPELINE_ENABLED", True)

# 切块进程池 worker 数；0 = 自动（min(4, max(1, cpu_count // 2))）。
# 注意：开启 ENABLE_SEMANTIC_CHUNKER 时每个 worker 会各自加载句向量模型，
# 内存占用按 worker 数翻倍，建议此时显式设为 1。
INGEST_CHUNK_WORKERS = max(0, int(os.getenv("INGEST_CHUNK_WORKERS", "0")))

# 切块内联阈值：markdown 长度小于该值时不进进程池（进程往返开销大于收益）。
INGEST_CHUNK_MIN_CHARS = max(0, int(os.getenv("INGEST_CHUNK_MIN_CHARS", "60000")))

# 阶段间队列容量（背压上限）：每个队列同时滞留的待入库文档数。
# 队列满时 submit() 立即失败（抛 IngestionQueueFullError，接口返回 503），
# 把压力回传给上游而不是无限缓冲。
INGEST_QUEUE_MAXSIZE = max(1, int(os.getenv("INGEST_QUEUE_MAXSIZE", "16")))

# Embedding 线程池 / 消费者数（实际 session.run 仍串行，线程池用于重叠分批提交）。
INGEST_EMBED_WORKERS = max(1, int(os.getenv("INGEST_EMBED_WORKERS", "2")))

# 写入阶段消费者数。写库分两阶段：阶段 A（原文件/图片/解析 md 落存储）是网络 IO，
# 多消费者可并行；阶段 B（向量库「删旧行 + 插新行」+ 文档间 SimHash 去重 + 指纹）
# 由写锁强制串行，不受此值影响。设为 1 即回到完全串行。
# 存储后端为本地文件系统时收益有限；为 OSS/S3 远程对象存储时收益明显。
INGEST_WRITE_WORKERS = max(1, int(os.getenv("INGEST_WRITE_WORKERS", "2")))

# 常驻 worker 监督间隔：检查是否有阶段 worker 意外退出并重启。
# 常驻 worker 一旦死亡，其上游队列会填满并导致全量上传阻塞，因此必须有兜底。
INGEST_SUPERVISOR_INTERVAL_SECONDS = max(0.1, float(os.getenv("INGEST_SUPERVISOR_INTERVAL_SECONDS", "1.0")))

# 各阶段队列深度采样间隔（Prometheus gauge）。
INGEST_METRICS_INTERVAL_SECONDS = max(0.5, float(os.getenv("INGEST_METRICS_INTERVAL_SECONDS", "2.0")))

# 关闭时等待在途文档结算的最长秒数；超时则强制取消并结算为失败。
INGEST_DRAIN_TIMEOUT_SECONDS = max(0.0, float(os.getenv("INGEST_DRAIN_TIMEOUT_SECONDS", "10.0")))


def resolve_chunk_workers() -> int:
    """解析切块进程池 worker 数：显式配置优先，否则按 CPU 数自动推导。"""
    if INGEST_CHUNK_WORKERS > 0:
        return INGEST_CHUNK_WORKERS
    return min(4, max(1, (os.cpu_count() or 2) // 2))


def resolve_parse_concurrency() -> int:
    """解析阶段常驻 worker 数（= MinerU 并发上限）。

    用 worker 数量直接表达"同时最多几个 MinerU 解析"，取代原先的
    ``Semaphore(MINERU_MAX_CONCURRENCY)``：worker 持有文档直到成功入队，
    因此下游背压会自然传导到解析阶段，少一个可出错的运行期机制。
    """
    return max(1, MINERU_MAX_CONCURRENCY)


def resolve_chunk_concurrency() -> int:
    """切块阶段常驻消费者数。

    必须与进程池 worker 数一致，否则单消费者会让进程池退化为串行执行
    （消费协程 ``await run_in_executor`` 逐个等待，池内 worker 空转）。
    """
    return max(1, resolve_chunk_workers())


def resolve_write_concurrency() -> int:
    """写入阶段常驻消费者数。

    阶段 A（存储上传）可并行，阶段 B（向量库原子区）由写锁串行，
    因此该值只影响存储 IO 的并发度，不影响写入正确性。
    """
    return max(1, INGEST_WRITE_WORKERS)


ENABLE_METADATA_LLM = env_bool("ENABLE_METADATA_LLM", True)
# 原文页渲染分辨率：72dpi 读研报太糊，默认 144dpi
PAGE_RENDER_DPI = int(os.getenv("PAGE_RENDER_DPI", "144"))
ENABLE_CITATION_VALIDATION = env_bool("ENABLE_CITATION_VALIDATION", False)
CITATION_SIMILARITY_THRESHOLD = float(os.getenv("CITATION_SIMILARITY_THRESHOLD", "0.4"))
ENABLE_REFUSAL = env_bool("ENABLE_REFUSAL", True)
REFUSAL_SCORE_THRESHOLD = float(os.getenv("REFUSAL_SCORE_THRESHOLD", "0.5"))
LOW_CONFIDENCE_THRESHOLD = float(os.getenv("LOW_CONFIDENCE_THRESHOLD", "0.8"))
REFUSAL_MIN_RERANK_SCORE = float(os.getenv("REFUSAL_MIN_RERANK_SCORE", "0.3"))
RAGAS_TIMEOUT_SECONDS = int(os.getenv("RAGAS_TIMEOUT_SECONDS", "300"))
RAGAS_MAX_RETRIES = max(1, int(os.getenv("RAGAS_MAX_RETRIES", "2")))
RAGAS_MAX_WORKERS = max(1, int(os.getenv("RAGAS_MAX_WORKERS", "4")))
ALLOWED_ORIGINS = [origin.strip() for origin in os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173,http://localhost:8000,http://127.0.0.1:8000").split(",") if origin.strip()]
TRUSTED_HOSTS = [host.strip() for host in os.getenv("TRUSTED_HOSTS", "localhost,127.0.0.1").split(",") if host.strip()]
JWT_SECRET = os.getenv("JWT_SECRET", "")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "1440"))
JWT_ADMIN_USERNAME = os.getenv("JWT_ADMIN_USERNAME", "admin")
JWT_ADMIN_PASSWORD = os.getenv("JWT_ADMIN_PASSWORD", "")

QUERY_REWRITE_BASE_URL = os.getenv("QUERY_REWRITE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
QUERY_REWRITE_MODEL = os.getenv("QUERY_REWRITE_MODEL", "qwen-turbo")
QUERY_REWRITE_API_KEY = os.getenv("QUERY_REWRITE_API_KEY", "") or DASHSCOPE_API_KEY
QUERY_REWRITE_TIMEOUT_SECONDS = float(os.getenv("QUERY_REWRITE_TIMEOUT_SECONDS", "15"))

STORAGE_BACKEND = os.getenv("STORAGE_BACKEND", "oss")
OSS_ENDPOINT = os.getenv("OSS_ENDPOINT", "https://oss-cn-hangzhou.aliyuncs.com")
OSS_BUCKET = os.getenv("OSS_BUCKET", "finance-rag-docs")
OSS_ACCESS_KEY_ID = os.getenv("OSS_ACCESS_KEY_ID", "")
OSS_ACCESS_KEY_SECRET = os.getenv("OSS_ACCESS_KEY_SECRET", "")
OSS_REGION = os.getenv("OSS_REGION", "cn-hangzhou")

# 生产环境禁止使用代码默认值或空值启动，避免服务以不完整配置对外提供能力。
def validate_production_config() -> None:
    """校验生产环境必需的安全、外部服务和存储配置。"""
    if APP_ENV != "production":
        return

    required_names = (
        "JWT_SECRET",
        "TRUSTED_HOSTS",
        "JWT_ADMIN_PASSWORD",
        "DEEPSEEK_API_KEY",
        "DATABASE_URL",
        "REDIS_URL",
        "MILVUS_URI",
        "OSS_ENDPOINT",
        "OSS_BUCKET",
        "OSS_ACCESS_KEY_ID",
        "OSS_ACCESS_KEY_SECRET",
        "OSS_REGION",
    )
    missing = [name for name in required_names if not os.getenv(name, "").strip()]
    if len(JWT_SECRET) < 32:
        missing.append("JWT_SECRET（至少 32 字符）")
    if STORAGE_BACKEND.lower() != "oss":
        missing.append("STORAGE_BACKEND=oss")
    if not OSS_ENDPOINT.lower().startswith("https://"):
        missing.append("OSS_ENDPOINT（必须使用 HTTPS）")
    if any("localhost" in origin.lower() or "127.0.0.1" in origin.lower() for origin in ALLOWED_ORIGINS):
        missing.append("ALLOWED_ORIGINS（生产环境禁止 localhost/127.0.0.1）")
    if missing:
        raise RuntimeError(f"生产配置不完整，缺少或无效配置：{', '.join(missing)}")


# 无论运行环境如何，缺少密钥都不得放行鉴权。
if not JWT_SECRET:
    logger.error("JWT_SECRET 未配置，鉴权不可用。请设置至少 32 字符的强随机字符串。")

validate_production_config()


# ---------------------------------------------------------------------------
# LLM 客户端（惰性初始化，避免 ``import config`` 时触发重导入/网络/告警）
# ---------------------------------------------------------------------------

_model: Any = None
_model_initialized = False
_rewrite_model: Any = None
_rewrite_model_initialized = False


def get_model() -> Any:
    """惰性获取主 LLM 客户端；未配置 API key 或初始化失败时返回 None。"""
    global _model, _model_initialized
    if not _model_initialized:
        _model_initialized = True
        if DEEPSEEK_API_KEY:
            try:
                from langchain.chat_models import init_chat_model

                _model = init_chat_model(
                    DEEPSEEK_MODEL,
                    api_key=DEEPSEEK_API_KEY,
                    timeout=LLM_TIMEOUT_SECONDS,
                    max_retries=LLM_MAX_RETRIES,
                    temperature=LLM_TEMPERATURE,
                )
            except Exception as exc:
                logger.warning("LLM 初始化失败（langchain-deepseek 可能未安装）：%s", exc)
    return _model


def get_rewrite_model() -> Any:
    """惰性获取查询改写轻量模型；未配置 key 或初始化失败时返回 None。"""
    global _rewrite_model, _rewrite_model_initialized
    if not _rewrite_model_initialized:
        _rewrite_model_initialized = True
        if QUERY_REWRITE_API_KEY:
            try:
                from langchain.chat_models import init_chat_model

                _rewrite_model = init_chat_model(
                    f"openai:{QUERY_REWRITE_MODEL}",
                    api_key=QUERY_REWRITE_API_KEY,
                    base_url=QUERY_REWRITE_BASE_URL,
                    timeout=QUERY_REWRITE_TIMEOUT_SECONDS,
                    max_retries=1,
                    temperature=0.1,
                )
            except Exception as exc:
                logger.warning("查询改写轻量模型初始化失败，将回退到主模型/规则改写：%s", exc)
    return _rewrite_model
