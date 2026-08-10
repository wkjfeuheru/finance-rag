"""JWT 鉴权模块测试。"""

import os
import time

import pytest
import pytest_asyncio

# 设置环境变量（必须在导入前）
os.environ["JWT_SECRET"] = "test-secret-for-pytest-at-least-32-chars"
os.environ["JWT_ADMIN_USERNAME"] = "admin"
os.environ["JWT_ADMIN_PASSWORD"] = "test-password"

from finance_rag.src.api.dependencies import (
    authenticate_user,
    create_access_token,
    decode_access_token,
    get_current_user,
)


class TestAuthenticateUser:
    def test_correct_credentials(self):
        assert authenticate_user("admin", "test-password") is True

    def test_wrong_password(self):
        assert authenticate_user("admin", "wrong") is False

    def test_wrong_username(self):
        assert authenticate_user("attacker", "test-password") is False

    def test_empty_credentials(self):
        assert authenticate_user("", "") is False


class TestJWT:
    def test_create_and_decode(self):
        token = create_access_token("admin")
        user = decode_access_token(token)
        assert user == "admin"

    def test_expired_token(self):
        """直接构造过期 token 测试过期校验。"""
        import jwt as pyjwt
        from datetime import datetime, timedelta, timezone
        from finance_rag.src.api.dependencies import decode_access_token
        from config.settings import JWT_SECRET, JWT_ALGORITHM

        payload = {
            "sub": "admin",
            "iat": int((datetime.now(timezone.utc) - timedelta(days=1)).timestamp()),
            "exp": int((datetime.now(timezone.utc) - timedelta(hours=1)).timestamp()),
        }
        expired_token = pyjwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
        with pytest.raises(Exception) as exc_info:
            decode_access_token(expired_token)
        assert exc_info.value.status_code == 401

    def test_invalid_token(self):
        with pytest.raises(Exception) as exc_info:
            decode_access_token("not.a.valid.token")
        assert exc_info.value.status_code == 401


class TestGetCurrentUser:
    @pytest.mark.asyncio
    async def test_no_token(self):
        """未带 token 应返回 401"""
        with pytest.raises(Exception) as exc_info:
            await get_current_user(None)
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_valid_token(self):
        token = create_access_token("admin")
        user = await get_current_user(token)
        assert user == "admin"