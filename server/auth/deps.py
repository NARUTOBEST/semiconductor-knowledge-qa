# -*- coding: utf-8 -*-
"""FastAPI 认证依赖:从 Authorization 头提取并验证 JWT。

用法(在需要保护的路由上):
    @router.post("/chat")
    async def chat(request: Request, user=Depends(get_current_user)):
        ...
"""
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from .service import verify_token
from .db import get_user_by_username

_security = HTTPBearer(auto_error=False)


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_security),
):
    """验证 Bearer token,返回 {"username", "role"}。

    失败时抛 401,FastAPI 自动返回 JSON {"detail": "..."}。
    """
    if not credentials:
        raise HTTPException(status_code=401, detail="未登录")

    payload = verify_token(credentials.credentials)
    if not payload:
        raise HTTPException(status_code=401, detail="登录已过期,请重新登录")

    username = payload.get("sub")
    user = get_user_by_username(username)
    if not user:
        raise HTTPException(status_code=401, detail="用户不存在")

    return {"username": user["username"], "role": user["role"]}
