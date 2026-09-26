"""Agentic RAG 问答 API 路由（流式 + 非流式）。"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sse_starlette.sse import EventSourceResponse

from finance_rag.src.api.dependencies import get_current_user
from finance_rag.src.api.streaming import encode_event
from finance_rag.src.schemas.chat import ChatRequest, ChatResponse, SourceInfo

logger = logging.getLogger(__name__)

router = APIRouter()

# chat_service 模块级导入会连带导入 langchain_core.language_models.chat_models
# → transformers → torch（实测约 5～7s）。为让服务尽早 accept 连接，这里改为
# 请求时惰性导入；应用启动阶段由 lifespan 的后台预热任务提前完成导入。


def _chat_fns():
    """惰性获取 chat_service 的入口函数（首次调用触发重依赖导入）。"""
    from finance_rag.src.services.chat_service import chat as chat_fn
    from finance_rag.src.services.chat_service import chat_stream as chat_stream_fn

    return chat_stream_fn, chat_fn


@router.post("/stream")
async def chat_stream(
    req: ChatRequest,
    current_user: Annotated[str, Depends(get_current_user)],
):
    """SSE 流式问答：逐 token 返回答案，含检索来源。"""
    history = [{"role": m.role, "content": m.content} for m in req.history]
    chat_stream_fn, _ = _chat_fns()

    async def event_generator():
        try:
            async for event in chat_stream_fn(
                req.query,
                history,
                use_rerank=req.use_rerank,
                k=req.k,
                rerank_top_n=req.rerank_top_n,
                filters=req.filters,
                infer_filters=req.infer_filters,
            ):
                yield encode_event(event)
        except Exception as exc:
            from finance_rag.src.core.exceptions import friendly_message

            logger.exception("流式问答异常：%s", exc)
            yield encode_event({"type": "error", "message": friendly_message(exc)})

    return EventSourceResponse(event_generator())


@router.post("", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    current_user: Annotated[str, Depends(get_current_user)],
):
    """非流式问答：返回完整答案 + 来源。"""
    history = [{"role": m.role, "content": m.content} for m in req.history]
    _, chat_fn = _chat_fns()

    try:
        result = await chat_fn(
            req.query,
            history,
            use_rerank=req.use_rerank,
            k=req.k,
            rerank_top_n=req.rerank_top_n,
            filters=req.filters,
            infer_filters=req.infer_filters,
        )
    except Exception as exc:
        from finance_rag.src.core.exceptions import friendly_message

        raise HTTPException(status_code=500, detail=friendly_message(exc))

    return ChatResponse(
        answer=result["answer"],
        sources=[SourceInfo(**s) for s in result["sources"]],
        rewritten_query=result["rewritten_query"],
        citation_validation=result.get("citation_validation"),
        answer_rejected=bool(result.get("answer_rejected", False)),
        low_confidence=bool(result.get("low_confidence", False)),
    )
