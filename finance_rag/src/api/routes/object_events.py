"""Object-storage event endpoint for asynchronous document reindexing."""
from __future__ import annotations

import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, Request

from finance_rag.src.api.dependencies import get_current_user
from finance_rag.src.core.config import MINERU_SUPPORTED_EXTENSIONS
from finance_rag.src.infrastructure.storage import get_storage
from finance_rag.src.services.object_event_adapter import accepts_key, parse_object_event, verify_signature
from finance_rag.src.services.document_service import get_document_manager

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/object-events")
async def object_event(
    request: Request,
    current_user: Annotated[str, Depends(get_current_user)],
    x_object_event_signature: Annotated[str | None, Header()] = None,
):
    """Accept an OSS event and enqueue the existing document processing path."""
    raw = await request.body()
    if not verify_signature(raw, x_object_event_signature):
        raise HTTPException(status_code=401, detail="对象事件签名无效")
    try:
        event = parse_object_event(await request.json())
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"对象事件格式无效：{exc}") from exc
    if not accepts_key(event.key):
        return {"accepted": False, "reason": "filtered"}
    suffix = Path(event.key).suffix.lower()
    if suffix not in MINERU_SUPPORTED_EXTENSIONS:
        return {"accepted": False, "reason": "unsupported_extension"}
    if "Removed" in event.event_type or "Delete" in event.event_type:
        # Deletion is deliberately soft at the service boundary.
        get_document_manager().soft_delete_source(Path(event.key).name)
        return {"accepted": True, "event_id": event.event_id, "action": "soft_delete"}

    storage = get_storage()
    temp_path = Path(tempfile.gettempdir()) / f".event-{event.event_id}{suffix}"
    await storage.download_to_path(event.key, temp_path)
    try:
        dm = get_document_manager()
        task = asyncio.create_task(dm.process_object_event(
            temp_path, Path(event.key).name, event.version_id or event.etag or event.event_id,
        ))
        return {"accepted": True, "event_id": event.event_id, "action": "index", "task": id(task)}
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


__all__ = ["router"]
