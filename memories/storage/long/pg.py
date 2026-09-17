# -*- coding: utf-8 -*-
"""长期记忆 PostgreSQL 连接工厂(psycopg3,线程安全版)。

长期记忆是【旁路增强】:PG 不可用 / 未装驱动时 get_pg() 返回 None,调用方据此跳过
偏好的抽取与注入,聊天主流程不受影响(不抛异常)。连接用 autocommit(每条语句即提交,
适合 best-effort 写入);pgvector 的 vector 类型适配通过 register_vector 注册,
之后可用 Python list[float] 作 vector 参数、读出 numpy 向量。

线程安全:psycopg3 Connection 非线程安全,不能多线程共享。此前全局单例 _conn 在
FastAPI 线程池(召回读 / 升迁写 / 后台抽取线程)并发下会协议错位甚至崩溃。现改为
【每线程独立连接】(thread-local):调用方 API 不变,仍 conn = get_pg() 后用 cursor,
只是不同线程拿到的是各自的连接。FastAPI 线程池线程数有界(anyio 默认 40)+ 少量
后台线程,总连接数有上界(PG 默认 max_connections=100 内可控)。

失败策略:连接失败按【线程 + 全局冷却】降级——本线程失败过不再反复重试;全局最近
一次失败在冷却期内时,其他线程也先跳过(避免 PG 宕机期每个线程各撞一次连接超时)。
冷却期过后线程首次调用会重新尝试,天然实现断线自愈;force=True(ping/init)立即重连。

连接信息全部来自 config(LONG_PG_URI 或 POSTGRES_* 拼装),禁止硬编码。
"""
from __future__ import annotations

import logging
import os
import sys
import threading
import time

_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "config"))
import config as C  # noqa: E402

logger = logging.getLogger("agent")

_CONNECT_TIMEOUT = float(os.getenv("LONG_PG_CONNECT_TIMEOUT", "4"))
_FAIL_COOLDOWN = float(os.getenv("LONG_PG_FAIL_COOLDOWN", "30"))

_local = threading.local()          # 每线程独立连接
_last_fail_ts = 0.0                 # 全局最近一次连接失败时间(冷却门)
_fail_lock = threading.Lock()


def _connect():
    try:
        import psycopg
    except Exception:
        return None
    try:
        conn = psycopg.connect(
            C.LONG_PG_URI,
            autocommit=True,
            connect_timeout=_CONNECT_TIMEOUT,
            application_name="long-memory",
        )
    except Exception as e:
        logger.info("long-memory PG connect skipped: %s: %s",
                    type(e).__name__, str(e)[:120])
        return None
    # 注册 vector 类型适配(失败不致命:仅影响向量参数/结果,结构化读写仍可用)
    try:
        from pgvector.psycopg import register_vector
        register_vector(conn)
    except Exception as e:
        logger.warning("pgvector register_vector failed (vector ops disabled): %s",
                       str(e)[:120])
    return conn


def _note_fail() -> None:
    global _last_fail_ts
    with _fail_lock:
        _last_fail_ts = time.time()


def _in_cooldown() -> bool:
    return (time.time() - _last_fail_ts) < _FAIL_COOLDOWN


def get_pg(force: bool = False):
    """返回当前线程专用的 PG 连接(thread-local);不可用返回 None。

    force=True 时丢弃本线程旧连接立即重连(供 init / 健康检查 / 断线恢复用)。
    连接已 closed/broken 时视为失效,自动重建。
    """
    conn = getattr(_local, "conn", None)
    if force and conn is not None:
        _close_conn(conn)
        conn = None
    if conn is not None:
        if conn.closed or getattr(conn, "broken", False):
            _close_conn(conn)
            conn = None
        else:
            return conn
    _local.conn = None  # 清掉失效引用,重连失败时不残留旧连接
    if not force and _in_cooldown():
        # 全局冷却期内(最近一次连接失败):先跳过,冷却过后由下一次调用自然重试
        return None
    conn = _connect()
    if conn is None:
        _note_fail()
        _local.tried = True
        return None
    _local.tried = True
    _local.conn = conn
    return conn


def ping_pg() -> bool:
    """显式探测 PG 连通性(强制重连并 SELECT 1)。"""
    conn = get_pg(force=True)
    if conn is None:
        return False
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            return cur.fetchone() is not None
    except Exception:
        _close_conn(conn)
        _local.conn = None
        _note_fail()
        return False


def _close_conn(conn) -> None:
    """关闭单个连接,吞掉一切异常(旁路增强,关闭失败不影响主流程)。"""
    try:
        conn.close()
    except Exception:
        pass
