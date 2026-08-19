# -*- coding: utf-8 -*-
"""长期记忆语义去重测试(mock DB + embed,不连真实 PG/模型)。"""
import os
import sys
from unittest.mock import MagicMock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# _embed_text 是模块级函数;但 long_term 名字被子模块底部的单例实例覆盖,
# 直接 import 子模块会拿到实例,故从 sys.modules 取真正的模块对象来 patch。
import memories.storage.long  # noqa: E402
lt_module = sys.modules["memories.storage.long.long_term"]
long_term = lt_module.long_term  # 单例实例


def _vec():
    return np.ones(1024, dtype=np.float32)


def _fake_conn(fetch_rows=None, insert_row=None):
    """构造假的 psycopg2 连接/游标。fetch_rows=SELECT 查重结果;insert_row=RETURNING。"""
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchall.return_value = fetch_rows or []
    cur.fetchone.return_value = insert_row
    return conn, cur


def test_semantic_dup_skips_insert(monkeypatch):
    # 查重命中相似度 0.97(>= 阈值) -> 返回已有 id,不执行 INSERT
    hit = {"id": 999, "content": "我用 Windows 11", "score": 0.97}
    conn, cur = _fake_conn(fetch_rows=[hit])
    monkeypatch.setattr(long_term, "_conn", lambda: conn)
    monkeypatch.setattr(long_term, "_release", lambda conn: None)
    monkeypatch.setattr(lt_module, "_embed_text", lambda t: _vec())

    mid = long_term.add_memory("alice", "我的电脑是 Win11", memory_type="fact")

    assert mid == 999
    executed = [c.args[0] for c in cur.execute.call_args_list]
    assert not any("INSERT" in q for q in executed)
    conn.commit.assert_not_called()


def test_no_similar_inserts_new(monkeypatch):
    # 查重无命中 -> 正常 INSERT 新记忆
    conn, cur = _fake_conn(fetch_rows=[], insert_row=(123,))
    monkeypatch.setattr(long_term, "_conn", lambda: conn)
    monkeypatch.setattr(long_term, "_release", lambda conn: None)
    monkeypatch.setattr(lt_module, "_embed_text", lambda t: _vec())

    mid = long_term.add_memory("alice", "我用 Ubuntu 做开发", memory_type="fact")

    assert mid == 123
    executed = [c.args[0] for c in cur.execute.call_args_list]
    assert any("INSERT" in q for q in executed)
    conn.commit.assert_called_once()


def test_below_threshold_not_dedup(monkeypatch):
    # 最近邻相似度 0.7(< 阈值) -> 不视为重复,继续 INSERT
    row = {"id": 999, "content": " unrelated ", "score": 0.7}
    conn, cur = _fake_conn(fetch_rows=[row], insert_row=(124,))
    monkeypatch.setattr(long_term, "_conn", lambda: conn)
    monkeypatch.setattr(long_term, "_release", lambda conn: None)
    monkeypatch.setattr(lt_module, "_embed_text", lambda t: _vec())

    mid = long_term.add_memory("alice", "一条全新的事实", memory_type="fact")

    assert mid == 124
    executed = [c.args[0] for c in cur.execute.call_args_list]
    assert any("INSERT" in q for q in executed)


def test_embed_false_skips_semantic_dedup(monkeypatch):
    # embed=False 时向量为 None -> 不做语义查重,直接走精确去重 INSERT 路径
    conn, cur = _fake_conn(insert_row=(222,))
    monkeypatch.setattr(long_term, "_conn", lambda: conn)
    monkeypatch.setattr(long_term, "_release", lambda conn: None)

    mid = long_term.add_memory("alice", "兜底写入", embed=False)

    assert mid == 222
    assert cur.execute.call_count == 1
    assert "INSERT" in cur.execute.call_args.args[0]


def test_dedup_disabled(monkeypatch):
    # dedup=False -> 即便有向量也不查重
    conn, cur = _fake_conn(fetch_rows=[{"id": 1, "content": "x", "score": 0.99}],
                           insert_row=(333,))
    monkeypatch.setattr(long_term, "_conn", lambda: conn)
    monkeypatch.setattr(long_term, "_release", lambda conn: None)
    monkeypatch.setattr(lt_module, "_embed_text", lambda t: _vec())

    mid = long_term.add_memory("alice", "强行写入", dedup=False)

    assert mid == 333
    executed = [c.args[0] for c in cur.execute.call_args_list]
    assert not any("SELECT" in q for q in executed)
    assert any("INSERT" in q for q in executed)
