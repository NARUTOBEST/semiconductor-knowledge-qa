# -*- coding: utf-8 -*-
"""短期记忆流水(Redis 实现)单测:用 fakeredis,不依赖真实 redis 服务。

覆盖:seq 原子递增、事件/线程索引落键、删线程、按用户前缀枚举与级联清理、
滚动 TTL(stale_threads / delete_threads_before)。
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib  # noqa: E402

import fakeredis  # noqa: E402

# 注意:memories.storage.short 包的 __init__ 里 `from .short_term import short_term`
# 会把包属性 short_term(原子模块)覆盖成同名单例实例,故不能用
# `import ...short_term as m`(拿到的是实例)。用 importlib 从 sys.modules 取真正的模块对象,
# 以便 monkeypatch 替换其命名空间内的 get_redis(short_term.py 里 from ..connections import get_redis)。
st_module = importlib.import_module("memories.storage.short.short_term")
from memories.storage.short.short_term import ShortTermMemory  # noqa: E402

_THREADS_INDEX = "mem:threads"


def _make(monkeypatch):
    """构造一个绑到 fakeredis 的 ShortTermMemory,并返回 (memory, fake_client)。"""
    fake = fakeredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(st_module, "get_redis", lambda: fake)
    return ShortTermMemory(), fake


class TestAppend:
    def test_seq_increments(self, monkeypatch):
        m, fake = _make(monkeypatch)
        s1 = m.append_event("alice|t1", "user_message", {"q": "hi"})
        s2 = m.append_event("alice|t1", "assistant_message", {"a": "hello"})
        assert s1 == 1 and s2 == 2

    def test_event_and_index_written(self, monkeypatch):
        m, fake = _make(monkeypatch)
        m.append_event("alice|t1", "user_message", {"q": "hi"},
                       user_id="alice", session_id="sess")
        # 事件 hash
        evt = fake.hgetall("mem:evt:alice|t1:1")
        assert evt["type"] == "user_message"
        assert '"q"' in evt["payload"] and "hi" in evt["payload"]
        assert evt["uid"] == "alice" and evt["sid"] == "sess"
        # 线程索引(score 为最后活动 ts)
        assert fake.zscore(_THREADS_INDEX, "alice|t1") is not None
        # 用户索引
        assert "alice|t1" in fake.smembers("mem:user:alice")
        # TTL 已设置(30 天)
        assert fake.ttl("mem:evt:alice|t1:1") > 0


class TestDeleteThread:
    def test_delete_removes_events(self, monkeypatch):
        m, fake = _make(monkeypatch)
        m.append_event("alice|t1", "user_message", {})
        m.append_event("alice|t1", "assistant_message", {})
        n = m.delete_thread("alice|t1")
        assert n == 2
        assert fake.zscore(_THREADS_INDEX, "alice|t1") is None
        assert fake.smembers("mem:evts:alice|t1") == set()
        assert "alice|t1" not in fake.smembers("mem:user:alice")


class TestUserScope:
    def test_list_and_delete_user(self, monkeypatch):
        m, fake = _make(monkeypatch)
        m.append_event("alice|t1", "user_message", {})
        m.append_event("alice|t2", "user_message", {})
        m.append_event("bob|t3", "user_message", {})

        assert set(m.list_user_threads("alice")) == {"alice|t1", "alice|t2"}

        r = m.delete_user("alice")
        # alice 两个线程各 1 事件
        assert r["events"] == 2
        # alice 的线程/事件已清,bob 的保留
        assert m.list_user_threads("alice") == []
        assert "bob|t3" in m.list_user_threads("bob")
        assert fake.zscore(_THREADS_INDEX, "bob|t3") is not None
        assert fake.zscore(_THREADS_INDEX, "alice|t1") is None


class TestTTLStale:
    def test_stale_and_delete_before(self, monkeypatch):
        m, fake = _make(monkeypatch)
        # 一个活跃线程 + 一个被手动改成 40 天前活动的旧线程
        m.append_event("carol|fresh", "user_message", {})
        m.append_event("carol|old", "user_message", {})
        old_ts = time.time() - 40 * 86400
        fake.zadd(_THREADS_INDEX, {"carol|old": old_ts})

        stale = m.stale_threads(30)
        stale_ids = [tid for tid, _ in stale]
        assert "carol|old" in stale_ids
        assert "carol|fresh" not in stale_ids

        deleted = m.delete_threads_before(30)
        assert deleted == 1
        assert fake.zscore(_THREADS_INDEX, "carol|old") is None
        assert fake.zscore(_THREADS_INDEX, "carol|fresh") is not None


class TestRecentDialogue:
    def _seed(self, m, tid, pairs):
        """pairs: [(role, content), ...] 顺序写入 user/assistant 消息事件。"""
        for role, content in pairs:
            etype = "user_message" if role == "user" else "assistant_message"
            m.append_event(tid, etype, {"content": content})

    def test_limit_none_returns_all_in_order(self, monkeypatch):
        m, _ = _make(monkeypatch)
        pairs = []
        for i in range(10):
            pairs += [("user", f"问题{i}"), ("assistant", f"回答{i}")]
        self._seed(m, "alice|t1", pairs)
        # 夹杂的过程事件不应混入对话
        m.append_event("alice|t1", "tool_call", {"name": "search_text"})

        rows = m.recent_dialogue("alice|t1")  # limit=None = 全部
        assert len(rows) == 20
        assert rows[0] == {"role": "user", "content": "问题0"}
        assert rows[-1] == {"role": "assistant", "content": "回答9"}
        assert all(r["role"] in ("user", "assistant") for r in rows)

    def test_limit_n_returns_last_n(self, monkeypatch):
        m, _ = _make(monkeypatch)
        pairs = []
        for i in range(10):
            pairs += [("user", f"u{i}"), ("assistant", f"a{i}")]
        self._seed(m, "alice|t2", pairs)

        rows = m.recent_dialogue("alice|t2", limit=4)
        assert len(rows) == 4
        # 取最近 4 条,且按时间正序
        assert [r["content"] for r in rows] == ["u8", "a8", "u9", "a9"]

    def test_empty_thread_returns_empty(self, monkeypatch):
        m, _ = _make(monkeypatch)
        assert m.recent_dialogue("nobody|t") == []
