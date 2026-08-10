"""项目配置中心。

配置优先级（高 → 低）：
1. 启动进程中已存在的环境变量；
2. 仓库根目录 ``.env``（不覆盖已有进程变量）；
3. 下方代码默认值。

修改环境变量后需重启服务生效。``.env.example`` 仅作模板，不参与运行时加载。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


# ===========================================================================
# 1. 路径常量与环境加载
# ===========================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_project_env(project_root: Path = PROJECT_ROOT) -> None:
    """加载仓库根目录 ``.env``，不覆盖已有进程环境变量。"""
    load_dotenv(dotenv_path=project_root / ".env", override=False)


# ===========================================================================
# 2. 类型解析工具
# ===========================================================================

_TRUE_VALUES = frozenset({"true", "1", "yes", "on"})
_FALSE_VALUES = frozenset({"false", "0", "no", "off"})
DEFAULT_CHUNK_TOKENIZER = "sentence-transformers/all-MiniLM-L6-v2"
ZH_CHUNK_TOKENIZER = "BAAI/bge-large-zh-v1.5"


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


def resolve_chunk_tokenizer(use_zh: bool, configured: str | None) -> str:
    """中文开关优先级高于显式 tokenizer 名称。

    - ``USE_ZH_TOKENIZER=true`` → 强制 ``BAAI/bge-large-zh-v1.5``
    - ``USE_ZH_TOKENIZER=false`` → 读取 ``DOCLING_CHUNK_TOKENIZER``，缺省回退 MiniLM
    """
    if use_zh:
        return ZH_CHUNK_TOKENIZER
    return configured or DEFAULT_CHUNK_TOKENIZER


def safe_parse_json(text: str, default: dict | None = None) -> dict:
    """解析 LLM JSON 输出，容忍 markdown 代码围栏。"""
    if default is None:
        default = {}

    content = (text or "").strip()
    if "```json" in content:
        content = content.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in content:
        parts = content.split("```")
        if len(parts) >= 2:
            content = parts[1]

    try:
        parsed: Any = json.loads(content.strip())
    except json.JSONDecodeError:
        return default

    return parsed if isinstance(parsed, dict) else default


# 模块导入时立即加载 .env，确保后续所有 os.getenv 能读到值
load_project_env()


# ===========================================================================
# 3. LLM 服务配置 (DeepSeek)
# ===========================================================================

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek:deepseek-v4-pro")
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "120"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))

# ===========================================================================
# 4. 嵌入服务与向量数据库 (ONNX INT8 本地嵌入 / Milvus)
# ===========================================================================

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))
ONNX_CACHE_DIR = os.getenv(
    "ONNX_CACHE_DIR",
    str(Path.home() / ".cache" / "finance_rag" / "onnx"),
)
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "25"))
# HuggingFace 镜像端点（国内网络无法直连 huggingface.co 时使用）
HF_ENDPOINT = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")
if HF_ENDPOINT:
    os.environ.setdefault("HF_ENDPOINT", HF_ENDPOINT)
# HuggingFace 本地缓存目录（嵌入模型 + reranker 共用）
os.environ.setdefault("HF_HOME", str(Path(ONNX_CACHE_DIR) / "hf_cache"))

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "")
MILVUS_TIMEOUT_SECONDS = float(os.getenv("MILVUS_TIMEOUT_SECONDS", "15"))
KB_COLLECTION_NAME = os.getenv("KB_COLLECTION_NAME", "finance_kb")
MILVUS_NLIST = int(os.getenv("MILVUS_NLIST", "128"))
MILVUS_NPROBE = int(os.getenv("MILVUS_NPROBE", "10"))

# LlamaParse（扫描版 PDF 云端 OCR 解析）
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY", "")
SCAN_PDF_TEXT_THRESHOLD = int(os.getenv("SCAN_PDF_TEXT_THRESHOLD", "50"))

# ===========================================================================
# 5. 文档切块配置 (层级父子切块)
# ===========================================================================

USE_ZH_TOKENIZER = env_bool("USE_ZH_TOKENIZER", False)
DOCLING_CHUNK_TOKENIZER = resolve_chunk_tokenizer(
    USE_ZH_TOKENIZER,
    os.getenv("DOCLING_CHUNK_TOKENIZER"),
)
DOCLING_CHUNK_MAX_TOKENS = int(os.getenv("DOCLING_CHUNK_MAX_TOKENS", "512"))
PARENT_MAX_TOKENS = int(os.getenv("PARENT_MAX_TOKENS", "2000"))
CHILD_OVERLAP_TOKENS = int(os.getenv("CHILD_OVERLAP_TOKENS", "50"))

# ===========================================================================
# 6. 混合检索配置 (稠密 + 稀疏 BM25 + RRF 融合)
# ===========================================================================

RRF_K = int(os.getenv("RRF_K", "60"))
PARENT_STORE_PATH = os.getenv(
    "PARENT_STORE_PATH",
    str(Path.home() / ".cache" / "finance_rag" / "parent_store.db"),
)

MULTI_STAGE_TOP_K = int(os.getenv("MULTI_STAGE_TOP_K", "20"))
MULTI_STAGE_EXPAND_COUNT = int(os.getenv("MULTI_STAGE_EXPAND_COUNT", "3"))

# ===========================================================================
# 7. BGE 重排序配置
# ===========================================================================

ENABLE_RERANKER = env_bool("ENABLE_RERANKER", True)
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
RERANKER_DEVICE = os.getenv("RERANKER_DEVICE", "cpu")

# 断崖检测：相邻 rerank_score 相对下降超过此阈值时截断
RERANKER_CLIFF_THRESHOLD = float(os.getenv("RERANKER_CLIFF_THRESHOLD", "0.35"))
RERANKER_CLIFF_MIN_RESULTS = max(1, int(os.getenv("RERANKER_CLIFF_MIN_RESULTS", "1")))

# 增量构建：文件哈希指纹存储路径
MILVUS_FINGERPRINT_PATH = os.getenv(
    "MILVUS_FINGERPRINT_PATH",
    os.path.join(os.getenv("UPLOAD_DIR", "files"), ".milvus_fingerprint"),
)

# ===========================================================================
# 8. 问答链路配置 (Chat)
# ===========================================================================

CHAT_TOP_K = int(os.getenv("CHAT_TOP_K", "5"))
CHAT_RERANK_TOP_K = int(os.getenv("CHAT_RERANK_TOP_K", "3"))
CHAT_ENABLE_QUERY_REWRITE = env_bool("CHAT_ENABLE_QUERY_REWRITE", True)

# 查询改写去重阈值
QUERY_REWRITE_SIMILARITY_THRESHOLD = min(
    1.0,
    max(0.0, float(os.getenv("QUERY_REWRITE_SIMILARITY_THRESHOLD", "0.85"))),
)

# LangGraph Agentic RAG 开关
ENABLE_LANGGRAPH = env_bool("ENABLE_LANGGRAPH", False)
LANGGRAPH_MAX_REFLECT_ROUNDS = min(3, max(1, int(os.getenv("LANGGRAPH_MAX_REFLECT_ROUNDS", "2"))))

# 多租户隔离
TENANT_ID = os.getenv("TENANT_ID", "default")
QUERY_REWRITE_MAX_ATTEMPTS = max(
    1, int(os.getenv("QUERY_REWRITE_MAX_ATTEMPTS", "3"))
)

# ===========================================================================
# 9. 文档上传与管理
# ===========================================================================

UPLOAD_DIR = os.getenv(
    "UPLOAD_DIR", str(PROJECT_ROOT / "files")
)
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "20"))
DOCUMENT_PARSE_WORKERS = int(os.getenv("DOCUMENT_PARSE_WORKERS", "4"))
PDF_PARSE_TIMEOUT_SECONDS = env_positive_float("PDF_PARSE_TIMEOUT_SECONDS", 180)

# ===========================================================================
# 10. RAG 优化开关（默认全部关闭，逐个启用验证效果）
# ===========================================================================

ENABLE_SMART_CHUNKER = env_bool("ENABLE_SMART_CHUNKER", False)
ENABLE_MULTI_STAGE_RETRIEVAL = env_bool("ENABLE_MULTI_STAGE_RETRIEVAL", False)
ENABLE_METADATA_FILTER = env_bool("ENABLE_METADATA_FILTER", False)
ENABLE_FINANCIAL_EXPERT_PROMPT = env_bool("ENABLE_FINANCIAL_EXPERT_PROMPT", False)
ENABLE_CITATION_VALIDATION = env_bool("ENABLE_CITATION_VALIDATION", False)

# 引用验证相似度阈值
CITATION_SIMILARITY_THRESHOLD = float(
    os.getenv("CITATION_SIMILARITY_THRESHOLD", "0.4")
)

# ===========================================================================
# 11. 策略评估 (Ragas)
# ===========================================================================

RAGAS_TIMEOUT_SECONDS = int(os.getenv("RAGAS_TIMEOUT_SECONDS", "150"))
RAGAS_MAX_RETRIES = max(1, int(os.getenv("RAGAS_MAX_RETRIES", "2")))
RAGAS_MAX_WORKERS = max(1, int(os.getenv("RAGAS_MAX_WORKERS", "2")))

# ===========================================================================
# 12. 安全配置 (CORS + JWT 单用户鉴权)
# ===========================================================================

# CORS 允许的来源（逗号分隔）
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,http://localhost:8000,http://127.0.0.1:8000",
    ).split(",")
    if origin.strip()
]

# JWT 单用户鉴权
JWT_SECRET = os.getenv("JWT_SECRET", "")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "1440"))
JWT_ADMIN_USERNAME = os.getenv("JWT_ADMIN_USERNAME", "admin")
JWT_ADMIN_PASSWORD = os.getenv("JWT_ADMIN_PASSWORD", "")

if not JWT_SECRET:
    import logging
    logging.getLogger(__name__).warning(
        "JWT_SECRET 未配置，鉴权将放行。请在 .env 中设置强随机字符串（至少 32 字符）。"
    )


# ===========================================================================
# 13. LLM 模型实例（延迟初始化，仅配置了 API_KEY 时创建）
# ===========================================================================

model = None
if DEEPSEEK_API_KEY:
    try:
        from langchain.chat_models import init_chat_model
    except ImportError:
        init_chat_model = None  # type: ignore[assignment]

    if init_chat_model is not None:
        try:
            model = init_chat_model(
                DEEPSEEK_MODEL,
                api_key=DEEPSEEK_API_KEY,
                timeout=LLM_TIMEOUT_SECONDS,
                max_retries=LLM_MAX_RETRIES,
            )
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(
                "LLM 初始化失败（langchain-deepseek 可能未安装）：%s", exc
            )
