# -*- coding: utf-8 -*-
"""速率限制:双层锁(per-user 1 + 全局信号量)。

两层:
  第1层(per-user): 每用户同时 1 个请求,超出直接 429
  第2层(global):  全局信号量 N 个,等待 QUEUE_TIMEOUT 秒,超时 429

注意:本实现基于 threading 内存锁,仅适用于单 worker 部署。
多 worker 场景需改用共享存储(Redis / SQLite)实现跨进程限流。
"""
import os
import sys
import time
import threading
import logging

from fastapi import HTTPException

_HERE = os.path.dirname(os.path.abspath(__file__))            # server/support/
_PROJECT = os.path.dirname(os.path.dirname(_HERE))            # project root
_CONFIG = os.path.join(_PROJECT, "config")
if _CONFIG not in sys.path:
    sys.path.insert(0, _CONFIG)

import config as C

logger = logging.getLogger("ratelimit")

# ---- 模块级状态(monkeypatch 友好,测试通过 conftest 替换)----
_global_sem = threading.Semaphore(C.RATE_LIMIT_GLOBAL_CONCURRENT)
_user_locks = {}                       # {username: True}
_lock = threading.Lock()               # 保护 _user_locks


# ---- 第1层: per-user ----

def acquire_user_slot(username: str):
    """获取用户级锁,同一用户已有请求则 429。"""
    with _lock:
        if username in _user_locks:
            raise HTTPException(
                status_code=429,
                detail="该用户已有进行中的请求,请等待",
            )
        _user_locks[username] = True


def release_user_slot(username: str):
    """释放用户级锁。"""
    with _lock:
        _user_locks.pop(username, None)


# ---- 第2层: global ----

def acquire_global_slot(username: str):
    """获取全局信号量,等待 QUEUE_TIMEOUT 秒,超时 429。"""
    acquired = _global_sem.acquire(timeout=C.RATE_LIMIT_QUEUE_TIMEOUT)
    if not acquired:
        raise HTTPException(
            status_code=429,
            detail="系统繁忙,请稍后重试",
        )


def release_global_slot(token: str = None):
    """释放全局信号量(token 参数为兼容性保留,内存版忽略)。"""
    _global_sem.release()


def release_all(username: str, global_token: str = None):
    """释放用户锁 + 全局信号量(在 finally 中调用)。"""
    release_global_slot(global_token)
    release_user_slot(username)
