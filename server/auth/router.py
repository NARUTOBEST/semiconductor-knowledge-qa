# -*- coding: utf-8 -*-
"""认证路由:POST /login + POST /register + GET /me。

注册:用户自助注册,默认 role=user。
登录:验证密码后签发 JWT。
/me:用当前 token 换取用户信息(前端用于刷新页面后恢复登录态)。
"""
import time
import logging
from collections import defaultdict
from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel

from .db import get_user_by_username, create_user, count_users
from .service import hash_password, verify_password, create_token
from .deps import get_current_user

logger = logging.getLogger("auth")
router = APIRouter()

# ---- 登录/注册 IP 限流(单进程内存版)----
_login_attempts = defaultdict(list)   # {ip: [timestamp, ...]}
_LOGIN_MAX_ATTEMPTS = 5               # 1 分钟内最多 5 次
_LOGIN_WINDOW_SECONDS = 60


def _check_login_rate_limit(request: Request):
    """同一 IP 1 分钟内最多 5 次登录/注册尝试,超出返回 429。"""
    ip = request.client.host if request.client else "unknown"
    now = time.time()
    # 清理过期记录
    recent = [t for t in _login_attempts[ip] if now - t < _LOGIN_WINDOW_SECONDS]
    if len(recent) >= _LOGIN_MAX_ATTEMPTS:
        raise HTTPException(status_code=429, detail="尝试过于频繁,请 1 分钟后再试")
    recent.append(now)
    _login_attempts[ip] = recent


class LoginRequest(BaseModel):
    username: str
    password: str


class RegisterRequest(BaseModel):
    username: str
    password: str


@router.post("/register")
def register(req: RegisterRequest, request: Request):
    """用户自助注册。首个用户自动成为 admin。"""
    _check_login_rate_limit(request)
    # ---- 输入校验 ----
    username = req.username.strip()
    password = req.password

    if len(username) < 3 or len(username) > 20:
        raise HTTPException(status_code=400, detail="用户名长度需为 3-20 个字符")
    if not username.replace("_", "").isalnum():
        raise HTTPException(status_code=400, detail="用户名只能包含字母、数字、下划线")
    if len(password) < 6:
        raise HTTPException(status_code=400, detail="密码至少 6 个字符")

    # ---- 查重 ----
    if get_user_by_username(username):
        raise HTTPException(status_code=409, detail="用户名已存在")

    # ---- 首个用户自动 admin ----
    role = "admin" if count_users() == 0 else "user"

    # ---- 写入 ----
    pw_hash = hash_password(password)
    if not create_user(username, pw_hash, role):
        raise HTTPException(status_code=409, detail="用户名已存在")

    token = create_token(username, role)
    logger.info(f"注册成功: {username} (role={role})")
    return {"token": token, "username": username, "role": role}


@router.post("/login")
def login(req: LoginRequest, request: Request):
    """用户登录,验证密码后签发 JWT。"""
    _check_login_rate_limit(request)
    username = req.username.strip()
    user = get_user_by_username(username)

    # 用户名或密码错误统一返回相同提示,不泄露用户名是否存在
    if not user or not verify_password(req.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="用户名或密码错误")

    token = create_token(user["username"], user["role"])
    logger.info(f"登录成功: {user['username']}")
    return {
        "token": token,
        "username": user["username"],
        "role": user["role"],
    }


@router.get("/me")
def me(user=Depends(get_current_user)):
    """用当前 token 换取用户信息(前端刷新页面后恢复登录态)。"""
    return {"username": user["username"], "role": user["role"]}
