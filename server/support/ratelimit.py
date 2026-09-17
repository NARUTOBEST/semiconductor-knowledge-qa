# -*- coding: utf-8 -*-
"""速率限制:双层锁(per-user 1 + 全局并发 N),状态外置 Redis。

两层:
  第1层(per-user): 每用户同时 1 个请求,超出直接 429
  第2层(global):  全局并发 N 个,等待 QUEUE_TIMEOUT 秒,超时 429

存储:默认 Redis(所有 worker 共享同一把锁/信号量,多 worker 语义正确,应用
重启状态不丢;进程崩溃持有的槽位由 TTL 自动过期回收,不泄漏);Redis 不可用
时自动回退进程内存(=外置前的单 worker 行为,见 state_store)。

token 传递:acquire_* 返回槽位 token,release_* 凭 token 释放(不依赖线程
局部变量——SSE 同步生成器在 starlette 线程池里迭代,每次 next 可能换线程)。

键设计(前缀 rl:,均带 TTL 兜底):
  rl:user:<username>  值=token;SET NX PX 获取,释放时 token 比对后删(防误删他人新锁)
  rl:global           ZSET:member=token, score=到期时间戳;清过期+计数+占位
                      在 WATCH 事务内原子完成(不用 Lua,兼容 fakeredis 测试)
"""
import os
import sys
import time
import uuid
import threading
import logging

from fastapi import HTTPException

_HERE = os.path.dirname(os.path.abspath(__file__))            # server/support/
_PROJECT = os.path.dirname(os.path.dirname(_HERE))            # project root
_CONFIG = os.path.join(_PROJECT, "config")
if _CONFIG not in sys.path:
    sys.path.insert(0, _CONFIG)

import config as C
from support import state_store

logger = logging.getLogger("ratelimit")

# ---- 内存回退状态(monkeypatch 友好,测试通过 conftest 替换)----
_global_sem = threading.Semaphore(C.RATE_LIMIT_GLOBAL_CONCURRENT)
_user_locks = {}                       # {username: True}
_lock = threading.Lock()               # 保护 _user_locks

# 槽位 TTL(秒):须大于单请求最长耗时;worker 崩溃未释放时到期自动回收。
_SLOT_TTL = float(os.getenv("RATE_LIMIT_SLOT_TTL", "600"))

_BUSY_USER = "该用户已有进行中的请求,请等待"
_BUSY_GLOBAL = "系统繁忙,请稍后重试"


def _r():
    return state_store.get_state_redis()


# ---- 第1层: per-user ----

def acquire_user_slot(username: str):
    """获取用户级锁,同一用户已有请求则 429。

    :returns: 槽位 token(Redis 模式);内存回退模式返回 None(release 按 username 释放)。
    """
    r = _r()
    if r is not None:
        token = uuid.uuid4().hex
        try:
            ok = r.set(f"rl:user:{username}", token, nx=True,
                       px=int(_SLOT_TTL * 1000))
        except Exception:
            state_store.note_fail()
            r = None
        else:
            if not ok:
                raise HTTPException(status_code=429, detail=_BUSY_USER)
            return token
    # 内存回退(Redis 关闭或当场失败)
    with _lock:
        if username in _user_locks:
            raise HTTPException(status_code=429, detail=_BUSY_USER)
        _user_locks[username] = True
    return None


def release_user_slot(username: str, token: str = None):
    """释放用户级锁。带 token = Redis 模式获取的槽:只走 Redis(失败靠 TTL 回收,
    绝不误碰内存回退态);token 为空 = 内存回退模式获取的槽。"""
    if token is not None:
        r = _r()
        if r is not None:
            try:
                _compare_del(r, f"rl:user:{username}", token)
            except Exception:
                state_store.note_fail()   # TTL 兜底回收,不影响内存回退态
        return
    with _lock:
        _user_locks.pop(username, None)


def _compare_del(r, key, token) -> bool:
    """GET==token 则 DEL(WATCH 事务保证原子;不用 Lua 以兼容 fakeredis)。"""
    from redis.exceptions import WatchError
    with r.pipeline() as p:
        while True:
            try:
                p.watch(key)
                cur = p.get(key)
                if cur != token:
                    p.unwatch()
                    return False
                p.multi()
                p.delete(key)
                p.execute()
                return True
            except WatchError:
                continue


# ---- 第2层: global ----

def acquire_global_slot(username: str):
    """获取全局并发槽位,等待 QUEUE_TIMEOUT 秒,超时 429。

    :returns: 槽位 token(Redis 模式);内存回退模式返回 None。
    """
    r = _r()
    if r is not None:
        deadline = time.time() + C.RATE_LIMIT_QUEUE_TIMEOUT
        try:
            while True:
                token = _zset_take_slot(r)
                if token:
                    return token
                if time.time() >= deadline:
                    raise HTTPException(status_code=429, detail=_BUSY_GLOBAL)
                time.sleep(0.05)
        except HTTPException:
            raise
        except Exception:
            state_store.note_fail()
    # 内存回退
    acquired = _global_sem.acquire(timeout=C.RATE_LIMIT_QUEUE_TIMEOUT)
    if not acquired:
        raise HTTPException(status_code=429, detail=_BUSY_GLOBAL)
    return None


def _zset_take_slot(r):
    """清过期 + 计数 + 占位(WATCH 事务内原子);抢不到返回 None。"""
    from redis.exceptions import WatchError
    key = "rl:global"
    token = uuid.uuid4().hex
    now = time.time()
    with r.pipeline() as p:
        while True:
            try:
                p.watch(key)
                p.zremrangebyscore(key, "-inf", now)   # 回收崩溃残留的过期槽
                if p.zcard(key) >= C.RATE_LIMIT_GLOBAL_CONCURRENT:
                    p.unwatch()
                    return None
                p.multi()
                p.zadd(key, {token: now + _SLOT_TTL})
                p.execute()
                return token
            except WatchError:
                continue


def release_global_slot(token: str = None):
    """释放全局槽位。带 token = Redis 模式获取的槽:只走 Redis(失败靠 TTL 回收,
    防止误 release 内存信号量);token 为空 = 内存回退模式获取的槽。"""
    if token is not None:
        r = _r()
        if r is not None:
            try:
                r.zrem("rl:global", token)
            except Exception:
                state_store.note_fail()   # TTL 兜底回收
        return
    _global_sem.release()


def release_all(username: str, user_token: str = None, global_token: str = None):
    """释放用户锁 + 全局槽位(在 finally 中调用)。"""
    release_global_slot(global_token)
    release_user_slot(username, user_token)
