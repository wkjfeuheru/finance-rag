"""FastAPI application for the financial Agentic RAG platform.

检索链路统一收口到 :class:`KnowledgeBase`，保留：
* 健康检查
* 流式 / 非流式问答
* 文档上传 / 列表 / 删除
* 知识库统计
* 策略评估（ragas）：调整稠密/稀疏权重、是否启用重排序，输出评估指标
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from finance_rag.logging_config import set_request_id

# 配置应用日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="金融 Agentic RAG 知识库平台",
    description="层级切块 + 稠密/稀疏混合检索 + BGE 重排序的金融问答平台",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str = Field(..., description="user | assistant")
    content: str


class ChatRequest(BaseModel):
    query: str
    history: list[ChatMessage] = Field(default_factory=list)
    use_rewrite: bool | None = None
    use_rerank: bool = True
    k: int = Field(default=5, ge=1, le=20)
    rerank_top_n: int = Field(default=3, ge=1, le=10)


class SourceInfo(BaseModel):
    index: int
    title: str
    source: str
    chunk: int | None = None
    score: float
    preview: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceInfo] = []
    rewritten_query: str


class UploadResponse(BaseModel):
    filename: str
    source: str
    title: str
    chunk_count: int
    parent_count: int
    size_mb: float


class UploadFailure(BaseModel):
    filename: str
    error: str


class BatchUploadResponse(BaseModel):
    total: int
    success_count: int
    failure_count: int
    successes: list[UploadResponse]
    failures: list[UploadFailure]


# --- 评估相关模型 ---

class EvaluateStrategyRequest(BaseModel):
    """策略评估请求。"""
    dense_weight: float = Field(default=0.7, ge=0.0, le=1.0, description="稠密向量权重")
    sparse_weight: float = Field(default=0.3, ge=0.0, le=1.0, description="稀疏向量权重")
    use_rerank: bool = Field(default=False, description="是否启用 BGE 重排序")
    rerank_top_n: int = Field(default=3, ge=1, le=10, description="重排序返回结果数")
    k: int = Field(default=5, ge=1, le=20, description="检索深度")


class EvaluateSingleQueryRequest(BaseModel):
    """单条查询评估请求。"""
    query: str = Field(..., min_length=1, max_length=2000, description="手动输入的测试查询")
    dense_weight: float = Field(default=0.7, ge=0.0, le=1.0, description="稠密向量权重")
    sparse_weight: float = Field(default=0.3, ge=0.0, le=1.0, description="稀疏向量权重")
    use_rerank: bool = Field(default=False, description="是否启用 BGE 重排序")
    rerank_top_n: int = Field(default=3, ge=1, le=10, description="重排序返回结果数")
    k: int = Field(default=5, ge=1, le=20, description="检索深度")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the Vue3 SPA frontend."""
    dist_html = Path(__file__).resolve().parents[1] / "frontend" / "dist" / "index.html"
    if dist_html.exists():
        return HTMLResponse(dist_html.read_text(encoding="utf-8"))
    return HTMLResponse(
        "<h2>前端未构建。</h2>"
        "<p>开发模式请运行 <code>cd frontend && npm run dev</code>（端口 5173）。</p>"
        "<p>生产部署请运行 <code>cd frontend && npm run build</code> 后重启后端。</p>",
        status_code=404,
    )


@app.get("/api/health")
async def health():
    """Health check including Milvus connection status via KnowledgeBase."""
    milvus_ok = False
    try:
        from finance_rag.knowledge_base import get_knowledge_base
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


# ===========================================================================
# Agentic RAG 问答 API
# ===========================================================================

# --- SSE 流式问答 ---

@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    """SSE 流式问答：逐 token 返回答案，含检索来源。"""
    from sse_starlette.sse import EventSourceResponse

    from finance_rag.chat import chat_stream as chat_stream_fn

    history = [{"role": m.role, "content": m.content} for m in req.history]

    async def event_generator():
        try:
            async for event in chat_stream_fn(
                req.query,
                history,
                use_rewrite=req.use_rewrite,
                use_rerank=req.use_rerank,
                k=req.k,
                rerank_top_n=req.rerank_top_n,
            ):
                yield {"event": event["type"], "data": json.dumps(event, ensure_ascii=False)}
        except Exception as exc:
            logger.exception("流式问答异常：%s", exc)
            yield {
                "event": "error",
                "data": json.dumps({"type": "error", "message": str(exc)}, ensure_ascii=False),
            }

    return EventSourceResponse(event_generator())


# --- 非流式问答 ---

@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """非流式问答：返回完整答案 + 来源。"""
    from finance_rag.chat import chat as chat_fn

    history = [{"role": m.role, "content": m.content} for m in req.history]

    try:
        result = await chat_fn(
            req.query,
            history,
            use_rewrite=req.use_rewrite,
            use_rerank=req.use_rerank,
            k=req.k,
            rerank_top_n=req.rerank_top_n,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return ChatResponse(
        answer=result["answer"],
        sources=[SourceInfo(**s) for s in result["sources"]],
        rewritten_query=result["rewritten_query"],
    )


# --- 文档上传 ---

@app.post("/api/documents/upload", response_model=BatchUploadResponse)
async def upload_documents(files: list[UploadFile] = File(...)):
    """上传文档到知识库（支持 .md / .txt / .pdf）。"""
    from finance_rag.document_manager import get_document_manager

    dm = get_document_manager()
    if not files:
        raise HTTPException(status_code=400, detail="No files were uploaded")

    successes = []
    failures = []
    try:
        results = await dm.upload_documents(files)
        for file, result in zip(files, results):
            if isinstance(result, BaseException):
                if not isinstance(result, ValueError):
                    logger.error(
                        "Document upload failed: %s",
                        result,
                        exc_info=(type(result), result, result.__traceback__),
                    )
                failures.append({
                    "filename": file.filename or "untitled",
                    "error": str(result),
                })
            else:
                successes.append(result)
    finally:
        for file in files:
            await file.close()

    return BatchUploadResponse(
        total=len(files),
        success_count=len(successes),
        failure_count=len(failures),
        successes=[UploadResponse(**item) for item in successes],
        failures=[UploadFailure(**item) for item in failures],
    )


# --- 文档列表 ---

@app.get("/api/documents")
async def list_documents():
    """列出知识库所有文档。"""
    from finance_rag.document_manager import get_document_manager

    dm = get_document_manager()
    return dm.list_documents()


# --- 文档删除 ---

@app.delete("/api/documents/{source:path}")
async def delete_document(source: str):
    """删除指定文档的 Milvus 向量记录，保留本地原文件。"""
    from finance_rag.document_manager import get_document_manager

    dm = get_document_manager()
    try:
        result = dm.delete_document(source)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return result


# --- 知识库统计 ---

@app.get("/api/kb/stats")
async def kb_stats():
    """知识库统计信息（文档数/块数/集合状态）。"""
    from finance_rag.document_manager import get_document_manager

    dm = get_document_manager()
    return dm.get_stats()


# ===========================================================================
# 策略评估 API（ragas）
# ===========================================================================

@app.get("/api/test-queries")
async def get_test_queries():
    """获取测试查询列表及标准答案状态。

    标准答案已固化在 ``files/docs/evaluation_qa.md``，无需调用大模型生成。
    """
    from finance_rag.evaluation import get_test_set_loader

    loader = get_test_set_loader()
    test_set = loader.load_test_set()
    has_gt = loader.has_ground_truth()

    return {
        "queries": [entry.to_dict() for entry in test_set],
        "count": len(test_set),
        "has_ground_truth": has_gt,
    }


@app.post("/api/evaluate-strategy")
async def evaluate_strategy(req: EvaluateSingleQueryRequest):
    """评估前端手动输入的单条测试查询并返回 Ragas 指标。"""
    from finance_rag.evaluation import (
        StrategyConfig,
        get_strategy_evaluator,
        get_test_set_loader,
    )

    loader = get_test_set_loader()
    if not loader.has_ground_truth():
        raise HTTPException(
            status_code=400,
            detail="标准答案缺失，请检查 files/docs/evaluation_qa.md 文件",
        )

    query = req.query.strip()
    test_set = loader.load_test_set()
    entry = next((item for item in test_set if item.query.strip() == query), None)
    if entry is None:
        raise HTTPException(
            status_code=400,
            detail="未找到该查询对应的标准答案，请先将查询及标准答案添加到 files/docs/evaluation_qa.md",
        )

    config = StrategyConfig(
        dense_weight=req.dense_weight,
        sparse_weight=req.sparse_weight,
        use_rerank=req.use_rerank,
        rerank_top_n=req.rerank_top_n,
        k=req.k,
    )
    try:
        logger.info("评估手动输入的测试查询 query=%s", query)
        return get_strategy_evaluator().evaluate(config, entry)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("策略评估失败：%s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

# ---------------------------------------------------------------------------
# 挂载 Vue3 前端构建产物（必须在所有 API 路由之后）
# ---------------------------------------------------------------------------

_dist_dir = Path(__file__).resolve().parents[1] / "frontend" / "dist"
if _dist_dir.exists():
    app.mount("/assets", StaticFiles(directory=str(_dist_dir / "assets")), name="frontend-assets")
    print(f"[api] 已挂载前端静态文件：{_dist_dir}")
