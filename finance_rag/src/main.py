"""FastAPI 应用入口 —— 金融 Agentic RAG 知识库平台。

检索链路统一收口到 :class:`KnowledgeBase`，保留：
* 健康检查
* 流式 / 非流式问答
* 文档上传 / 列表 / 删除
* 知识库统计
* 策略评估（ragas）：切换 RRF 混合检索 / 纯向量检索、是否启用重排序，输出评估指标
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from prometheus_fastapi_instrumentator import Instrumentator

from finance_rag.src.core.config import (
    ALLOWED_ORIGINS,
    ENABLE_RERANKER,
    TRUSTED_HOSTS,
    WARMUP_ENABLED,
)
from finance_rag.src.core.dependencies import get_knowledge_base
from finance_rag.src.infrastructure.vector_store.milvus_kb import get_milvus_client


_readiness = {
    "live": False,
    "ready": False,
    "models": {"embedding": False, "reranker": not ENABLE_RERANKER},
    "error": None,
}

# 配置应用日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

# 同时落盘到 logs/app.log（UTF-8），避免终端关闭后运行日志无痕可查
_LOG_DIR = Path(__file__).resolve().parents[2] / "logs"
try:
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    _file_handler = logging.FileHandler(_LOG_DIR / "app.log", encoding="utf-8")
    _file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    )
    _file_handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(_file_handler)
except OSError:  # 落盘失败不阻断启动
    pass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 应用生命周期
# ---------------------------------------------------------------------------

# 缓存后端预热任务：uvicorn 开始 accept 后仍在跑，健康检查据此报告就绪状态
_warmup_task: asyncio.Task | None = None


def _preload_heavy_modules() -> None:
    """导入检索问答链路的重依赖（langchain → transformers → torch）。

    顺序很关键：先导入 ``chat_service``（连带 langchain → transformers → torch），
    再导入检索器与编排模块，最后加载本地模型。全部放在后台线程里执行，
    避免阻塞 uvicorn 绑定端口。

    ``langchain_core.language_models.chat_models``（由 ``langchain_core.prompts`` /
    ``output_parsers`` 间接导入）在**导入期**就会导入 ``transformers``，而
    transformers 顶层又导入 ``torch``——这是冷启动的主要剩余开销（约 5～7s）。
    langchain_core 自身已用 ``__getattr__`` 做了惰性导入，但其内部依赖链无法避免。
    """
    import time

    from finance_rag.src.rag.retrieval.hybrid_retriever import BGEReranker
    from finance_rag.src.services import chat_service  # noqa: F401

    logger.info("重依赖导入完成（含 langchain/transformers/torch）")
    logger.info("预热 ONNX 嵌入模型...")
    started = time.perf_counter()
    get_knowledge_base().get_embeddings().warmup()
    _readiness["models"]["embedding"] = True
    logger.info("嵌入模型预热完成（%.1fs）", time.perf_counter() - started)

    if ENABLE_RERANKER:
        logger.info("预热 BGE 重排序模型...")
        started = time.perf_counter()
        BGEReranker().warmup()
        _readiness["models"]["reranker"] = True
        logger.info("重排序模型预热完成（%.1fs）", time.perf_counter() - started)


async def _warmup_models() -> None:
    """后台预热：重依赖导入 + 嵌入/重排序模型加载。

    全部通过 ``asyncio.to_thread`` 执行，既不阻塞事件循环，也不阻塞启动。
    失败只记录状态，不影响服务存活（首个请求会自行降级或报可读错误）。
    """
    try:
        await asyncio.to_thread(_preload_heavy_modules)
        _readiness["ready"] = True
    except Exception as exc:
        _readiness["ready"] = False
        _readiness["error"] = str(exc)
        logger.exception("模型预热失败，服务保持未就绪状态")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时后台预热模型，立即让服务进入可服务状态。

    冷启动优化：把「重依赖导入 + 模型加载」移出启动阻塞路径，uvicorn 绑定端口
    后立刻 accept 连接。就绪性仍由 ``/api/health/ready`` 报告（预热未完成返回 503），
    因此不会把冷模型暴露给真实流量。
    """
    global _warmup_task

    _readiness["live"] = True
    _readiness["ready"] = False
    _readiness["models"]["embedding"] = False
    _readiness["models"]["reranker"] = not ENABLE_RERANKER
    _readiness["error"] = None

    if WARMUP_ENABLED:
        _warmup_task = asyncio.create_task(_warmup_models(), name="model-warmup")
    else:
        # 显式关闭预热：不做后台加载，直接就绪（模型在首次调用时惰性加载）
        _readiness["ready"] = True
        logger.info("WARMUP_ENABLED=false，跳过启动预热")

    yield

    _readiness["live"] = False
    _readiness["ready"] = False
    task, _warmup_task = _warmup_task, None
    if task is not None and not task.done():
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:  # pragma: no cover - 收尾阶段失败不阻断退出
            logger.warning("预热任务收尾异常", exc_info=True)
    # 入库流水线持有进程池/线程池：必须先排空在途文档再显式释放，
    # 否则 ProcessPoolExecutor 的解释器退出清理会 join 子进程并拖住进程退出。
    try:
        from finance_rag.src.services.ingestion_pipeline import (
            drain_ingestion_pipeline,
            shutdown_ingestion_pipeline,
        )

        if not await drain_ingestion_pipeline():
            logger.warning("入库流水线仍有在途文档未结算，将强制结算为失败")
        shutdown_ingestion_pipeline()
    except Exception:  # pragma: no cover - 关闭失败不阻断退出
        logger.warning("入库流水线关闭失败", exc_info=True)


# ---------------------------------------------------------------------------
# FastAPI 应用实例
# ---------------------------------------------------------------------------

app = FastAPI(
    title="金融 Agentic RAG 知识库平台",
    description="层级切块 + 稠密/稀疏混合检索 + BGE 重排序的金融问答平台",
    version="2.0.0",
    lifespan=lifespan,
)

# 主机白名单与 CORS 中间件
app.add_middleware(TrustedHostMiddleware, allowed_hosts=TRUSTED_HOSTS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "Accept"],
)

# Prometheus 指标暴露（/metrics 端点）
Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=True)

# ---------------------------------------------------------------------------
# 全局异常处理
# ---------------------------------------------------------------------------


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """兜底处理所有未捕获异常：记录完整堆栈并返回结构化中文 JSON。"""
    from finance_rag.src.core.exceptions import friendly_message

    logger.exception("未处理异常：%s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": friendly_message(exc)},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """请求参数校验失败：返回 422 结构化提示。"""
    return JSONResponse(
        status_code=422,
        content={
            "detail": "请求参数校验失败",
            "errors": exc.errors(),
        },
    )

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

# OSS 对象事件路由（前缀：/api）
from finance_rag.src.api.routes.object_events import router as object_events_router

app.include_router(object_events_router, prefix="/api", tags=["object-events"])

from finance_rag.src.api.routes.knowledge_bases import router as knowledge_bases_router

app.include_router(knowledge_bases_router, prefix="/api", tags=["knowledge-bases"])

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


@app.get("/api/health/live")
async def liveness():
    """存活检查：只确认进程仍在运行，不依赖外部服务或模型。"""
    return {"status": "ok" if _readiness["live"] else "down"}


@app.get("/api/health/ready")
async def readiness():
    """就绪检查：模型预热成功后才允许流量进入。"""
    status_code = 200 if _readiness["ready"] else 503
    return JSONResponse(status_code=status_code, content={
        "status": "ready" if _readiness["ready"] else "not_ready",
        "models": _readiness["models"],
    })


@app.get("/api/health")
async def health():
    """综合健康检查，返回存活、就绪和 Milvus 状态。"""
    milvus_ok = False
    try:
        # 仅探测 Milvus 连接，不在健康检查中创建或加载集合。
        get_milvus_client().has_collection("healthcheck")
        milvus_ok = True
    except Exception:
        logger.warning("Milvus 健康探测失败", exc_info=True)

    healthy = _readiness["live"] and _readiness["ready"] and milvus_ok is True
    return JSONResponse(
        status_code=200 if healthy else 503,
        content={
            "status": "ok" if healthy else "not_ready",
            "live": _readiness["live"],
            "ready": _readiness["ready"],
            "models": _readiness["models"],
            "milvus": milvus_ok,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )


# ---------------------------------------------------------------------------
# 挂载 Vue3 前端构建产物（必须在所有 API 路由之后）
# ---------------------------------------------------------------------------

_dist_dir = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if _dist_dir.exists():
    app.mount("/assets", StaticFiles(directory=str(_dist_dir / "assets")), name="frontend-assets")
    logger.info("已挂载前端静态文件：%s", _dist_dir)


def main() -> None:
    """启动规范 FastAPI 应用。"""
    import uvicorn

    uvicorn.run(
        "finance_rag.src.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("DEBUG", "false").lower() == "true",
        log_level="info",
    )


if __name__ == "__main__":
    main()
