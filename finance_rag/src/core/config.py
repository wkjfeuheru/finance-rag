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
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY", "")
SCAN_PDF_TEXT_THRESHOLD = int(os.getenv("SCAN_PDF_TEXT_THRESHOLD", "50"))

# 文档图片处理
IMAGE_CAPTION_BASE_URL = os.getenv("IMAGE_CAPTION_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
IMAGE_CAPTION_MODEL = os.getenv("IMAGE_CAPTION_MODEL", "qwen-vl-max")
IMAGE_CAPTION_TIMEOUT_SECONDS = float(os.getenv("IMAGE_CAPTION_TIMEOUT_SECONDS", "30"))
IMAGE_CAPTION_MAX_IMAGES = int(os.getenv("IMAGE_CAPTION_MAX_IMAGES", "10"))
IMAGE_CAPTION_MAX_RETRIES = int(os.getenv("IMAGE_CAPTION_MAX_RETRIES", "2"))
IMAGE_CAPTION_MIN_DIM_PX = int(os.getenv("IMAGE_CAPTION_MIN_DIM_PX", "80"))
IMAGE_CAPTION_MIN_BYTES = int(os.getenv("IMAGE_CAPTION_MIN_BYTES", "5000"))
IMAGE_CAPTION_PROMPT = os.getenv("IMAGE_CAPTION_PROMPT", "这是一份金融文档中嵌入的图片。请用中文详细描述图片内容：如果是图表，说明图表类型、标题、关键数据与趋势；如果是表格，转述主要行列内容；如果是印章/签名/截图/扫描件，说明其性质与可见文字。只输出描述本身，不超过 200 字。")

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
RUNTIME_STATE_DIR = PROJECT_ROOT / "data" / "state"
ENABLE_RERANKER = env_bool("ENABLE_RERANKER", True)
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
RERANKER_DEVICE = os.getenv("RERANKER_DEVICE", "cpu")
RERANKER_CLIFF_THRESHOLD = float(os.getenv("RERANKER_CLIFF_THRESHOLD", "0.35"))
RERANKER_CLIFF_MIN_RESULTS = max(1, int(os.getenv("RERANKER_CLIFF_MIN_RESULTS", "1")))
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
ENABLE_METADATA_FILTER = env_bool("ENABLE_METADATA_FILTER", False)
ENABLE_CITATION_VALIDATION = env_bool("ENABLE_CITATION_VALIDATION", False)
CITATION_SIMILARITY_THRESHOLD = float(os.getenv("CITATION_SIMILARITY_THRESHOLD", "0.4"))
ENABLE_REFUSAL = env_bool("ENABLE_REFUSAL", True)
REFUSAL_SCORE_THRESHOLD = float(os.getenv("REFUSAL_SCORE_THRESHOLD", "0.5"))
LOW_CONFIDENCE_THRESHOLD = float(os.getenv("LOW_CONFIDENCE_THRESHOLD", "0.8"))
REFUSAL_MIN_RERANK_SCORE = float(os.getenv("REFUSAL_MIN_RERANK_SCORE", "0.3"))
COMPLIANCE_REQUIRE_CLAUSE = env_bool("COMPLIANCE_REQUIRE_CLAUSE", True)
COMPLIANCE_CLAUSE_MIN_SCORE = float(os.getenv("COMPLIANCE_CLAUSE_MIN_SCORE", "0.5"))
COMPLIANCE_CATEGORIES = [c.strip() for c in os.getenv("COMPLIANCE_CATEGORIES", "compliance_risk").split(",") if c.strip()]
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
