# -*- coding: utf-8 -*-
"""auth.db(conversations 复用同库)WAL 日志模式测试:并发写不报 database is locked。

背景:默认 journal 模式下写并发排队、极端情况超 busy timeout 报 locked;
get_conn 统一开 WAL(读写互不阻塞,写-写仍由 timeout=10 排队)。
"""
import os
import sys
import threading

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "server"))

import config as C  # noqa: E402


def _mode(conn):
    return conn.execute("PRAGMA journal_mode").fetchone()[0].lower()


def test_auth_db_wal_mode():
    from auth.db import get_conn, init_db
    init_db()
    conn = get_conn()
    assert _mode(conn) == "wal"
    conn.close()


def test_conversation_db_wal_mode():
    from chat.conversation import db as cdb
    cdb.init_table()
    conn = cdb.get_conn()
    assert _mode(conn) == "wal"
    conn.close()


def test_concurrent_writes_no_locked_error():
    """多线程并发写同一库:WAL + busy timeout 下不应抛 database is locked。"""
    from auth.db import get_conn, init_db
    init_db()
    errs = []

    def writer(i):
        try:
            for _ in range(20):
                conn = get_conn()
                conn.execute(
                    "INSERT OR IGNORE INTO users(username, password_hash) VALUES (?, ?)",
                    (f"wal_user_{i}", "x"))
                conn.commit()
                conn.close()
        except Exception as e:  # noqa: BLE001
            errs.append(e)

    ts = [threading.Thread(target=writer, args=(i,)) for i in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(10)
    assert errs == []
