"""知识库管理 API 路由（增删改查逻辑"知识库类别"）。

知识库类别共用同一个 Milvus 集合（finance_kb），类别值作为文档分类元数据；
本路由只管理注册表（显示名/描述/自定义类别），不操作集合本身。
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from finance_rag.src.api.dependencies import get_current_user
from finance_rag.src.services.knowledge_base_service import (
    KnowledgeBaseNotEmptyError,
    get_kb_registry,
)
from finance_rag.src.schemas.knowledge_base import (
    KnowledgeBaseCreate,
    KnowledgeBaseInfo,
    KnowledgeBaseUpdate,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/knowledge-bases", response_model=list[KnowledgeBaseInfo])
async def list_knowledge_bases(
    current_user: Annotated[str, Depends(get_current_user)],
):
    """列出所有知识库类别（内置四类 + 自定义，含文档数/切块数）。"""
    return get_kb_registry().list_knowledge_bases()


@router.post("/knowledge-bases", response_model=KnowledgeBaseInfo)
async def create_knowledge_base(
    payload: KnowledgeBaseCreate,
    current_user: Annotated[str, Depends(get_current_user)],
):
    """创建自定义知识库类别（不创建 Milvus 集合，仅注册分类值）。"""
    registry = get_kb_registry()
    try:
        return registry.create_knowledge_base(
            payload.name,
            display_name=payload.display_name,
            description=payload.description,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("创建知识库类别失败：%s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.patch("/knowledge-bases/{name}", response_model=KnowledgeBaseInfo)
async def update_knowledge_base(
    name: str,
    payload: KnowledgeBaseUpdate,
    current_user: Annotated[str, Depends(get_current_user)],
):
    """修改知识库类别（显示名/描述/类别值；改名会联动迁移集合内文档分类）。"""
    registry = get_kb_registry()
    try:
        entry = registry.update_knowledge_base(
            name,
            display_name=payload.display_name,
            description=payload.description,
            new_name=payload.name,
        )
        # 合并统计信息返回完整结构（改名后按新名称查统计）
        lookup = entry.get("name", name)
        for item in registry.list_knowledge_bases():
            if item["name"] == lookup:
                return item
        return KnowledgeBaseInfo(
            name=lookup,
            display_name=entry.get("display_name", lookup),
            description=entry.get("description", ""),
            created_at=entry.get("created_at", ""),
            document_count=0,
            chunk_count=0,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("更新知识库类别失败：%s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.delete("/knowledge-bases/{name}")
async def delete_knowledge_base(
    name: str,
    current_user: Annotated[str, Depends(get_current_user)],
):
    """删除知识库类别（有文档 409；内置与自定义均可删除；不删除集合）。"""
    registry = get_kb_registry()
    try:
        return registry.delete_knowledge_base(name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except KnowledgeBaseNotEmptyError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except Exception as exc:
        logger.exception("删除知识库类别失败：%s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
