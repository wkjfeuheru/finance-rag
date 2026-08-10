"""Agentic RAG 问答 API 路由（流式 + 非流式）。"""

from __future__ import annotations

import json
import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sse_starlette.sse import EventSourceResponse

from finance_rag.src.api.dependencies import get_current_user
from finance_rag.src.application.chat_service import chat_stream as chat_stream_fn, chat as chat_fn
from finance_rag.src.schemas.chat import ChatRequest, ChatResponse, SourceInfo

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/stream")
async def chat_stream(
    req: ChatRequest,
    current_user: Annotated[str, Depends(get_current_user)],
):
    """SSE 流式问答：逐 token 返回答案，含检索来源。"""
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
                strategy=req.strategy,
                filters=req.filters,
            ):
                yield {"event": event["type"], "data": json.dumps(event, ensure_ascii=False)}
        except Exception as exc:
            logger.exception("流式问答异常：%s", exc)
            yield {
                "event": "error",
                "data": json.dumps({"type": "error", "message": str(exc)}, ensure_ascii=False),
            }

    return EventSourceResponse(event_generator())


@router.post("", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    current_user: Annotated[str, Depends(get_current_user)],
):
    """非流式问答：返回完整答案 + 来源。"""
    history = [{"role": m.role, "content": m.content} for m in req.history]

    try:
        result = await chat_fn(
            req.query,
            history,
            use_rewrite=req.use_rewrite,
            use_rerank=req.use_rerank,
            k=req.k,
            rerank_top_n=req.rerank_top_n,
            strategy=req.strategy,
            filters=req.filters,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return ChatResponse(
        answer=result["answer"],
        sources=[SourceInfo(**s) for s in result["sources"]],
        rewritten_query=result["rewritten_query"],
        citation_validation=result.get("citation_validation"),
    )
