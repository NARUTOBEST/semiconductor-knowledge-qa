# -*- coding: utf-8 -*-
"""认证路由:POST /login + POST /register + GET /me。

注册:用户自助注册,默认 role=user。
登录:验证密码后签发 JWT。
/me:用当前 token 换取用户信息(前端用于刷新页面后恢复登录态)。
"""
import time
import uuid
import logging
from collections import defaultdict
from fastapi import APIRouter, HTTPException, Depends, Request, Header
from pydantic import BaseModel

import config as C
from .db import get_user_by_username, create_user, count_users, delete_user
from .service import hash_password, verify_password, create_token
from .deps import get_current_user

logger = logging.getLogger("auth")
router = APIRouter()

# ---- 登录/注册 IP 限流(状态外置 Redis,多 worker 共享;不可用回退进程内存)----
_login_attempts = defaultdict(list)   # 内存回退:{ip: [timestamp, ...]}
_LOGIN_MAX_ATTEMPTS = 5               # 1 分钟内最多 5 次
_LOGIN_WINDOW_SECONDS = 60


def _check_login_rate_limit(request: Request):
    """同一 IP 1 分钟内最多 5 次登录/注册尝试,超出返回 429。

    主路径走 Redis ZSET 滑动窗口(键 rl:login:{ip},多 worker 部署下全局
    一致,防爆破强度不因多进程打折);Redis 不可用回退进程内存(=单 worker
    时代行为,多 worker 下各算各的——降级可用但不精确,见 state_store)。
    """
    ip = request.client.host if request.client else "unknown"
    if _redis_record_attempt(ip):
        return
    now = time.time()
    # 清理过期记录
    recent = [t for t in _login_attempts[ip] if now - t < _LOGIN_WINDOW_SECONDS]
    if len(recent) >= _LOGIN_MAX_ATTEMPTS:
        raise HTTPException(status_code=429, detail="尝试过于频繁,请 1 分钟后再试")
    recent.append(now)
    _login_attempts[ip] = recent


def _redis_record_attempt(ip: str) -> bool:
    """Redis 侧记账。True=已记账(或已 429);False=Redis 不可用,走内存回退。"""
    try:
        from support import state_store
    except ImportError:
        return False
    r = state_store.get_state_redis()
    if r is None:
        return False
    key = f"rl:login:{ip}"
    now = time.time()
    try:
        with r.pipeline() as p:
            p.zremrangebyscore(key, "-inf", now - _LOGIN_WINDOW_SECONDS)
            p.zcard(key)
            count = int(p.execute()[1])
        if count >= _LOGIN_MAX_ATTEMPTS:
            # 与内存版语义一致:被拒的尝试不记账
            raise HTTPException(status_code=429, detail="尝试过于频繁,请 1 分钟后再试")
        member = f"{now}:{uuid.uuid4().hex}"   # member 唯一,避免 zadd 覆盖同分成员
        r.zadd(key, {member: now})
        r.expire(key, _LOGIN_WINDOW_SECONDS)   # 整键 TTL 兜底,防长期不活跃 IP 残留
        return True
    except HTTPException:
        raise
    except Exception:
        state_store.note_fail()
        return False


class LoginRequest(BaseModel):
    username: str
    password: str


class RegisterRequest(BaseModel):
    username: str
    password: str


def _bootstrap_admin_authorized(bootstrap_token: str | None) -> bool:
    """校验首个管理员注册是否被授权。

    不再"首个注册者自动成 admin"(任何人抢先注册即可提权)。首个用户要成为 admin,
    必须在请求头携带与服务端 ``ADMIN_BOOTSTRAP_TOKEN`` 一致的引导令牌(部署方通过
    环境变量注入,用后即弃)。未配置令牌或令牌不符时,首个用户只能是普通 user。
    """
    expected = getattr(C, "ADMIN_BOOTSTRAP_TOKEN", "") or ""
    if not expected:
        return False
    return bool(bootstrap_token) and bootstrap_token == expected


@router.post("/register")
def register(req: RegisterRequest, request: Request,
             x_bootstrap_token: str | None = Header(default=None)):
    """用户自助注册。默认 role=user;首个用户需携带正确的引导令牌才成为 admin。"""
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

    # ---- 角色:仅在"系统尚无用户"且"携带正确引导令牌"时才授予 admin ----
    is_first_user = count_users() == 0
    if is_first_user and _bootstrap_admin_authorized(x_bootstrap_token):
        role = "admin"
    else:
        role = "user"

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


class DeleteAccountRequest(BaseModel):
    password: str


@router.delete("/me")
def delete_me(req: DeleteAccountRequest,
              user=Depends(get_current_user)):
    """账号注销:删除账号并级联清除该用户全部个人数据(被遗忘权)。

    需在请求体携带当前密码做二次确认,防止 token 被盗后被恶意注销。
    级联范围:
      - auth.db users 行;
      - auth.db conversations 该用户全部会话;
      - working 库该用户命名空间(``username|%``)下所有 checkpoint;
      - short 库该用户命名空间下 session_events 流水。
    记忆层清理失败不阻断账号删除(残留由运维据日志补偿),但会在返回中标注。
    """
    username = user["username"]

    # 二次确认:必须校验当前密码
    row = get_user_by_username(username)
    if not row or not verify_password(req.password, row["password_hash"]):
        raise HTTPException(status_code=401, detail="密码错误,无法注销")

    # 1) 记忆两层级联(须在删 conversations 之前:它要枚举会话 id 定位 checkpoint)
    purge = {"checkpoints_deleted": 0, "short_events_deleted": 0}
    purge_error = None
    try:
        from memories.orchestration import delete_user_artifacts
        purge = delete_user_artifacts(username)
    except Exception as e:
        purge_error = f"{type(e).__name__}: {e}"
        logger.exception(f"account purge memory failed for {username}")

    # 2) 会话表
    conv_deleted = 0
    try:
        from chat.conversation.db import delete_all_for_user
        conv_deleted = delete_all_for_user(username)
    except Exception:
        logger.exception(f"account purge conversations failed for {username}")

    # 3) 用户行
    if not delete_user(username):
        raise HTTPException(status_code=404, detail="用户不存在或已注销")

    logger.info(
        f"账号注销: {username} | 会话 {conv_deleted} "
        f"checkpoint {purge['checkpoints_deleted']} 短期流水 {purge['short_events_deleted']}"
        + (f" | 记忆清理异常: {purge_error}" if purge_error else "")
    )
    return {
        "ok": True,
        "deleted": {
            "conversations": conv_deleted,
            "checkpoints": purge["checkpoints_deleted"],
            "short_events": purge["short_events_deleted"],
        },
        "warning": purge_error,
    }
