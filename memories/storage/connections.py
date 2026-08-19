# -*- coding: utf-8 -*-
"""连接工厂:三个 PG 库 + 可选 Redis。

- PG 用 psycopg2(v2),每调用新建短连接(简单可靠;后续可换连接池)。
- Redis 为可选高速缓存:连接失败不抛异常,降级为不可用,调用方据此跳过缓存。
- 连接信息全部来自 config,禁止硬编码。
"""
import contextlib
import logging
import os
import sys
import threading

import psycopg2
import psycopg2.extras  # noqa: F401  注册 jsonb/uuid 等适配
from psycopg2 import pool as pg_pool

# 本文件位于 memories/storage/;项目根在上两级(memories/storage -> memories -> 项目根)
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "config"))
import config as C  # noqa: E402

logger = logging.getLogger("agent")

# PG 连接池:每个库一个 ThreadedConnectionPool(进程级单例,懒加载)。
# 三库合计最多 3*PG_POOL_MAX 条连接,需与 PG max_connections 协调。
_POOL_MIN = int(os.getenv("PG_POOL_MIN", "1"))
_POOL_MAX = int(os.getenv("PG_POOL_MAX", "10"))
_PG_CONNECT_TIMEOUT = int(os.getenv("PG_CONNECT_TIMEOUT", "3"))

_pools: dict[str, "pg_pool.ThreadedConnectionPool"] = {}
_pool_sems: dict[str, threading.BoundedSemaphore] = {}
_pools_lock = threading.Lock()

# 等待空闲连接的最长秒数;超时抛 PoolTimeout,由上层降级/报错而非无限堆积线程。
_POOL_ACQUIRE_TIMEOUT = float(os.getenv("PG_POOL_ACQUIRE_TIMEOUT", "30"))


def _get_pool(which: str) -> "pg_pool.ThreadedConnectionPool":
    """获取(惰性创建)指定库的线程连接池及其许可信号量。线程安全。"""
    if which in _pools:
        return _pools[which]
    with _pools_lock:
        if which in _pools:  # double-checked
            return _pools[which]
        uri = _uri_for(which)
        p = pg_pool.ThreadedConnectionPool(
            _POOL_MIN, _POOL_MAX, _pool_uri(uri),
        )
        _pools[which] = p
        # 用有界信号量把 getconn 变成"阻塞等待":ThreadedConnectionPool 自身在
        # 池满时会直接抛 PoolError,信号量保证同时取连接数不超过 _POOL_MAX。
        _pool_sems[which] = threading.BoundedSemaphore(_POOL_MAX)
        return p


def pool_getconn(which: str):
    """从指定库池取一个连接,池满时阻塞等待至超时。

    配合 pool_putconn 使用。超时抛 psycopg2.pool.PoolError。
    """
    pool = _get_pool(which)
    sem = _pool_sems[which]
    if not sem.acquire(timeout=_POOL_ACQUIRE_TIMEOUT):
        raise pg_pool.PoolError(
            f"连接池 {which} 等待空闲连接超时({_POOL_ACQUIRE_TIMEOUT}s)")
    try:
        return pool.getconn()
    except Exception:
        sem.release()
        raise


def pool_putconn(which: str, conn, *, close: bool = False) -> None:
    """归还连接到池并释放许可。close=True 时丢弃坏连接并重建。"""
    pool = _get_pool(which)
    try:
        pool.putconn(conn, close=close)
    finally:
        try:
            _pool_sems[which].release()
        except ValueError:
            pass


def _pool_uri(uri: str) -> str:
    if "connect_timeout" in uri:
        return uri
    sep = "&" if "?" in uri else "?"
    return f"{uri}{sep}connect_timeout={_PG_CONNECT_TIMEOUT}"


# ---------------------------------------------------------------- Redis
_redis_client = None
_redis_checked = False
_redis_available = False


def _build_redis():
    """尝试构建 Redis 客户端;不验证连通性(交给 ping)。"""
    try:
        import redis  # redis-py
    except Exception:
        return None
    try:
        return redis.Redis(
            host=C.REDIS_HOST,
            port=C.REDIS_PORT,
            password=C.REDIS_PASSWORD or None,
            db=C.REDIS_DB,
            decode_responses=True,
            socket_connect_timeout=1.0,
            socket_timeout=1.0,
        )
    except Exception:
        return None


def get_redis():
    """返回 Redis 客户端;不可用时返回 None。

    仅探测一次(进程生命周期内)。Redis 不可用不影响主流程。
    """
    global _redis_client, _redis_checked, _redis_available
    if _redis_checked:
        return _redis_client if _redis_available else None
    _redis_checked = True
    client = _build_redis()
    if client is not None:
        try:
            client.ping()
            _redis_available = True
            _redis_client = client
        except Exception:
            _redis_available = False
            _redis_client = None
    return _redis_client


def ping_redis() -> bool:
    """显式探测 Redis 连通性(同时强制重新检测)。"""
    global _redis_checked
    _redis_checked = False
    return get_redis() is not None


# ---------------------------------------------------------------- PG
def _uri_for(which: str) -> str:
    uri = {"working": C.WORKING_PG_URI, "short": C.SHORT_PG_URI,
           "long": C.LONG_PG_URI}.get(which)
    if not uri:
        raise RuntimeError(f"{which.upper()}_PG_URI 未配置,请检查 env/env.env")
    return uri


@contextlib.contextmanager
def pg_conn(which: str, dict_row: bool = True):
    """从连接池取一个指定 PG 库连接的上下文管理器。

    which: 'working' | 'short' | 'long'
    dict_row=True 时游标返回 dict-like 行。默认 autocommit=False,由调用方提交。
    连接用毕归还连接池(失败时丢弃坏连接,避免污染池)。
    """
    conn = pool_getconn(which)
    try:
        if dict_row:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                yield conn, cur
        else:
            with conn.cursor() as cur:
                yield conn, cur
        conn.commit()
    except Exception:
        # 回滚后连接通常仍干净可复用;仅回滚本身失败才丢弃连接
        try:
            conn.rollback()
        except Exception:
            pool_putconn(which, conn, close=True)
            raise
        raise
    finally:
        pool_putconn(which, conn)
