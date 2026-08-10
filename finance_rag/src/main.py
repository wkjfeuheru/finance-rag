"""FastAPI 应用入口 —— 金融 Agentic RAG 知识库平台。

检索链路统一收口到 :class:`KnowledgeBase`，保留：
* 健康检查
* 流式 / 非流式问答
* 文档上传 / 列表 / 删除
* 知识库统计
* 策略评估（ragas）：切换 RRF 混合检索 / 纯向量检索、是否启用重排序，输出评估指标
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from prometheus_fastapi_instrumentator import Instrumentator

from config.settings import ALLOWED_ORIGINS
from finance_rag.src.infrastructure.vector_store.milvus_kb import get_knowledge_base

# 配置应用日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastAPI 应用实例
# ---------------------------------------------------------------------------

app = FastAPI(
    title="金融 Agentic RAG 知识库平台",
    description="层级切块 + 稠密/稀疏混合检索 + BGE 重排序的金融问答平台",
    version="2.0.0",
)

# CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus 指标暴露（/metrics 端点）
Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=True)

# ---------------------------------------------------------------------------
# 路由注册
# ---------------------------------------------------------------------------

# 鉴权路由（前缀：/api/auth）
from finance_rag.src.api.routes.auth import router as auth_router

app.include_router(auth_router, prefix="/api/auth", tags=["auth"])

# 聊天路由（前缀：/api/chat）
from finance_rag.src.api.routes.chat import router as chat_router

app.include_router(chat_router, prefix="/api/chat", tags=["chat"])

# 文档管理路由（前缀：/api —— 含 /documents、/tasks、/kb）
from finance_rag.src.api.routes.documents import router as documents_router

app.include_router(documents_router, prefix="/api", tags=["documents"])

# 评估路由（前缀：/api —— 含 /test-queries、/evaluate-*、/eval-runs）
from finance_rag.src.api.routes.eval import router as eval_router

app.include_router(eval_router, prefix="/api", tags=["eval"])

# ---------------------------------------------------------------------------
# 根路由 —— 前端 SPA 入口
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the Vue3 SPA frontend."""
    dist_html = Path(__file__).resolve().parents[2] / "frontend" / "dist" / "index.html"
    if dist_html.exists():
        return HTMLResponse(dist_html.read_text(encoding="utf-8"))
    return HTMLResponse(
        "<h2>前端未构建。</h2>"
        "<p>开发模式请运行 <code>cd frontend && npm run dev</code>（端口 5173）。</p>"
        "<p>生产部署请运行 <code>cd frontend && npm run build</code> 后重启后端。</p>",
        status_code=404,
    )


# ---------------------------------------------------------------------------
# 健康检查
# ---------------------------------------------------------------------------


@app.get("/api/health")
async def health():
    """Health check including Milvus connection status via KnowledgeBase."""
    milvus_ok = False
    try:
        kb = get_knowledge_base()
        kb.ensure_collection()
        milvus_ok = True
    except Exception as exc:
        milvus_ok = str(exc)

    return {
        "status": "ok",
        "milvus": milvus_ok,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# 挂载 Vue3 前端构建产物（必须在所有 API 路由之后）
# ---------------------------------------------------------------------------

_dist_dir = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if _dist_dir.exists():
    app.mount("/assets", StaticFiles(directory=str(_dist_dir / "assets")), name="frontend-assets")
    logger.info("已挂载前端静态文件：%s", _dist_dir)
