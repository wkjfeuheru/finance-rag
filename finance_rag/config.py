from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


def safe_parse_json(text: str, default: dict | None = None) -> dict:
    """Parse LLM JSON output with markdown-fence tolerance."""
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


load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "")
MILVUS_TIMEOUT_SECONDS = float(os.getenv("MILVUS_TIMEOUT_SECONDS", "15"))
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek:deepseek-v4-pro")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
RERANKER_DEVICE = os.getenv("RERANKER_DEVICE", "cpu")

# --- Agentic RAG 问答知识库配置 ---
KB_COLLECTION_NAME = os.getenv("KB_COLLECTION_NAME", "finance_kb")
DOCLING_CHUNK_TOKENIZER = os.getenv(
    "DOCLING_CHUNK_TOKENIZER",
    "BAAI/bge-large-zh-v1.5" if os.getenv("USE_ZH_TOKENIZER", "false").lower() == "true"
    else "sentence-transformers/all-MiniLM-L6-v2",
)
DOCLING_CHUNK_MAX_TOKENS = int(os.getenv("DOCLING_CHUNK_MAX_TOKENS", "512"))
HYBRID_DENSE_WEIGHT = float(os.getenv("HYBRID_DENSE_WEIGHT", "0.7"))
HYBRID_SPARSE_WEIGHT = float(os.getenv("HYBRID_SPARSE_WEIGHT", "0.3"))
CHAT_TOP_K = int(os.getenv("CHAT_TOP_K", "5"))
CHAT_RERANK_TOP_K = int(os.getenv("CHAT_RERANK_TOP_K", "3"))
CHAT_ENABLE_QUERY_REWRITE = os.getenv("CHAT_ENABLE_QUERY_REWRITE", "true").lower() == "true"
RAGAS_TIMEOUT_SECONDS = int(os.getenv("RAGAS_TIMEOUT_SECONDS", "150"))
RAGAS_MAX_RETRIES = max(1, int(os.getenv("RAGAS_MAX_RETRIES", "2")))
RAGAS_MAX_WORKERS = max(1, int(os.getenv("RAGAS_MAX_WORKERS", "2")))
QUERY_REWRITE_SIMILARITY_THRESHOLD = min(
    1.0,
    max(
        0.0,
        float(os.getenv("QUERY_REWRITE_SIMILARITY_THRESHOLD", "0.8")),
    ),
)
QUERY_REWRITE_MAX_ATTEMPTS = max(
    1,
    int(os.getenv("QUERY_REWRITE_MAX_ATTEMPTS", "3")),
)
UPLOAD_DIR = os.getenv("UPLOAD_DIR", str(Path(__file__).resolve().parents[1] / "files"))
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "20"))
DOCUMENT_PARSE_WORKERS = int(os.getenv("DOCUMENT_PARSE_WORKERS", "4"))

# === RAG 优化开关（默认关闭，保持原有行为） ===
USE_ZH_TOKENIZER = os.getenv("USE_ZH_TOKENIZER", "false").lower() == "true"
ENABLE_SMART_CHUNKER = os.getenv("ENABLE_SMART_CHUNKER", "false").lower() == "true"
ENABLE_MULTI_STAGE_RETRIEVAL = os.getenv("ENABLE_MULTI_STAGE_RETRIEVAL", "false").lower() == "true"
ENABLE_METADATA_FILTER = os.getenv("ENABLE_METADATA_FILTER", "false").lower() == "true"
ENABLE_FINANCIAL_EXPERT_PROMPT = os.getenv("ENABLE_FINANCIAL_EXPERT_PROMPT", "false").lower() == "true"
ENABLE_CITATION_VALIDATION = os.getenv("ENABLE_CITATION_VALIDATION", "false").lower() == "true"

# 多阶段检索参数
MULTI_STAGE_TOP_K = int(os.getenv("MULTI_STAGE_TOP_K", "20"))
MULTI_STAGE_EXPAND_COUNT = int(os.getenv("MULTI_STAGE_EXPAND_COUNT", "3"))

# 检索参数调优
MILVUS_NPROBE = int(os.getenv("MILVUS_NPROBE", "10"))
MILVUS_NLIST = int(os.getenv("MILVUS_NLIST", "128"))

# 引用验证相似度阈值
CITATION_SIMILARITY_THRESHOLD = float(os.getenv("CITATION_SIMILARITY_THRESHOLD", "0.4"))

model = None
if DEEPSEEK_API_KEY:
    from langchain.chat_models import init_chat_model

    model = init_chat_model(
        DEEPSEEK_MODEL,
        api_key=DEEPSEEK_API_KEY,
        timeout=float(os.getenv("LLM_TIMEOUT_SECONDS", "120")),
        max_retries=int(os.getenv("LLM_MAX_RETRIES", "2")),
    )
