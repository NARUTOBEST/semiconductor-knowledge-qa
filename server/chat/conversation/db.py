# -*- coding: utf-8 -*-
"""会话持久化:SQLite(复用 auth.db)。

表结构:
  conversations(id, user_id, title, messages_json, created_at, updated_at)

messages 存为 JSON 字符串(10 用户规模足够,不需要分表)。
"""
import sqlite3
import json
import os
import sys


import config as C


def get_conn():
    conn = sqlite3.connect(C.AUTH_DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def init_table():
    conn = get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id            TEXT PRIMARY KEY,
            user_id       TEXT NOT NULL,
            title         TEXT NOT NULL,
            messages_json TEXT NOT NULL DEFAULT '[]',
            created_at    INTEGER NOT NULL,
            updated_at    INTEGER NOT NULL
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_conv_user ON conversations(user_id)"
    )
    conn.commit()
    conn.close()


def upsert_conversation(conv_id, user_id, title, messages, created_at, updated_at):
    """新建或更新一条会话(按 id upsert)。

    冲突时仅当原行属同一 user_id 才更新,防止他人以已知会话 id
    覆写别人的会话(IDOR);属他人时本行被静默跳过。
    """
    conn = get_conn()
    try:
        conn.execute("""
            INSERT INTO conversations (id, user_id, title, messages_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                title = excluded.title,
                messages_json = excluded.messages_json,
                updated_at = excluded.updated_at
            WHERE conversations.user_id = excluded.user_id
        """, (conv_id, user_id, title, json.dumps(messages, ensure_ascii=False), created_at, updated_at))
        conn.commit()
    finally:
        conn.close()


def get_conversations(user_id):
    """获取某用户的全部会话(按更新时间倒序)。"""
    conn = get_conn()
    try:
        rows = conn.execute(
            "SELECT * FROM conversations WHERE user_id = ? ORDER BY updated_at DESC",
            (user_id,),
        ).fetchall()
    finally:
        conn.close()
    return [{
        "id": r["id"],
        "title": r["title"],
        "messages": json.loads(r["messages_json"]),
        "createdAt": r["created_at"],
        "updatedAt": r["updated_at"],
    } for r in rows]


def delete_conversation(conv_id, user_id):
    """删除一条会话(校验 user_id 防越权)。返回实际删除的行数。"""
    conn = get_conn()
    try:
        cur = conn.execute(
            "DELETE FROM conversations WHERE id = ? AND user_id = ?",
            (conv_id, user_id),
        )
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()
