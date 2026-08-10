"""鉴权 API 路由。"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from finance_rag.src.utils.audit import log as audit_log
from finance_rag.src.api.dependencies import (
    LoginRequest,
    TokenResponse,
    authenticate_user,
    create_access_token,
)

router = APIRouter()


@router.post("/login", response_model=TokenResponse)
async def login(req: LoginRequest):
    """单用户登录，校验 .env 配置的账号密码后签发 JWT。"""
    if not authenticate_user(req.username, req.password):
        audit_log("login", user=req.username, result="failure", detail="密码错误")
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    token = create_access_token(req.username)
    audit_log("login", user=req.username, result="success")
    return TokenResponse(access_token=token)
