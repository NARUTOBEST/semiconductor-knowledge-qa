# -*- coding: utf-8 -*-
"""SQLite 用户表管理。无 ORM,直接 sqlite3,轻量零外部依赖。

表结构:
  users(id, username UNIQUE, password_hash, role, created_at)
"""
import sqlite3
import config as C


def get_conn():
    conn = sqlite3.connect(C.AUTH_DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    # WAL:读写互不阻塞(写-写仍由 timeout=10 排队),避免并发写报 database is locked。
    # WAL 对库文件持久,重复设置幂等;个别盘(网络盘)不支持时静默回退默认 journal。
    try:
        conn.execute("PRAGMA journal_mode=WAL")
    except sqlite3.OperationalError:
        pass
    return conn


def init_db():
    """创建 users 表(幂等,已存在则跳过)。"""
    conn = get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            username      TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role          TEXT NOT NULL DEFAULT 'user',
            created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def get_user_by_username(username):
    """按用户名查用户,返回 Row 或 None。"""
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM users WHERE username = ?", (username,)
    ).fetchone()
    conn.close()
    return row


def create_user(username, password_hash, role="user"):
    """插入用户。成功返回 True,用户名已存在返回 False。"""
    conn = get_conn()
    try:
        conn.execute(
            "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
            (username, password_hash, role),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def count_users():
    """返回用户总数(用于判断是否首个用户)。"""
    conn = get_conn()
    n = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    conn.close()
    return n


def delete_user(username: str) -> bool:
    """删除用户行(账号注销)。返回是否确实删除了一行。"""
    conn = get_conn()
    try:
        cur = conn.execute("DELETE FROM users WHERE username = ?", (username,))
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()
