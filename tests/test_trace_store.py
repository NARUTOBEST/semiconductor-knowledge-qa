# -*- coding: utf-8 -*-
"""trace 包(TraceStore)测试:独立 Redis 键空间的工作流追踪记录。

覆盖:
  - 白名单:只录 tool_call/tool_result/error/error_trace/done,其余跳过;
  - 写入 + 按 seq 正序回读;done 保留完整 trace(不剥离);
  - TTL:所有键按 TRACE_TTL_DAYS 设过期;
  - payload 上限:超大 trace 被截断并打标,键不膨胀;
  - 软失败:Redis 不可用不抛异常、返回 False/[];
  - 删除:delete_thread / delete_user 级联清理;
  - 开关:TRACE_STORE_ENABLED=0 时不写。
用 fakeredis,不触网、不依赖真实 Redis。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402
import fakeredis  # noqa: E402

import config as C  # noqa: E402
import trace as trace_pkg  # noqa: E402
from trace import store as trace_store_mod  # noqa: E402
import memories.storage.connections as conn  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_breaker():
    """每个用例前后复位熔断器,避免开路状态跨用例泄漏。"""
    trace_pkg.reset_circuit()
    yield
    trace_pkg.reset_circuit()


def _patch_redis(monkeypatch):
    fake = fakeredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(conn, "get_redis", lambda: fake)
    return fake


def _rec(ev, tid="alice|t1", **kw):
    return trace_pkg.record_trace_event(ev, thread_id=tid,
                                        user_id="alice", session_id="s1", **kw)


# ---------------- 白名单 ----------------
def test_only_whitelisted_types_recorded(monkeypatch):
    _patch_redis(monkeypatch)
    for t in ["token", "status", "step_start", "user_message", "assistant_message"]:
        assert _rec({"type": t, "x": 1}) is False
    for t in ["tool_call", "tool_result", "error", "error_trace", "done"]:
        assert _rec({"type": t, "x": t}) is True
    evs = trace_pkg.get_trace_events("alice|t1")
    assert [e["type"] for e in evs] == \
        ["tool_call", "tool_result", "error", "error_trace", "done"]


def test_non_dict_or_empty_thread_skipped(monkeypatch):
    _patch_redis(monkeypatch)
    assert trace_pkg.record_trace_event("not-a-dict", thread_id="alice|t1") is False
    assert trace_pkg.record_trace_event({"type": "done"}, thread_id="") is False


# ---------------- 写入/回读 + done 保留完整 trace ----------------
def test_records_and_reads_in_order_done_keeps_trace(monkeypatch):
    _patch_redis(monkeypatch)
    full_trace = {"steps": [{"llm": {"out": "x"}}], "final_reason": "answer"}
    _rec({"type": "tool_call", "name": "search_text", "args": {"q": "ALD"}})
    _rec({"type": "tool_result", "name": "search_text", "ok": True})
    _rec({"type": "done", "trace": full_trace, "answer": "ALD 是…"})

    evs = trace_pkg.get_trace_events("alice|t1")
    assert [e["seq"] for e in evs] == [1, 2, 3]
    assert evs[0]["payload"]["name"] == "search_text"
    assert evs[0]["uid"] == "alice" and evs[0]["sid"] == "s1"
    # done 的完整 trace 必须保留(追踪存储不剥离,与短期流水相反)
    done = evs[-1]["payload"]
    assert done["trace"] == full_trace
    assert done["answer"] == "ALD 是…"


def test_since_seq_and_limit(monkeypatch):
    _patch_redis(monkeypatch)
    for i in range(5):
        _rec({"type": "tool_result", "i": i})
    assert len(trace_pkg.get_trace_events("alice|t1")) == 5
    assert [e["payload"]["i"] for e in trace_pkg.get_trace_events("alice|t1", since_seq=2)] == [2, 3, 4]
    assert len(trace_pkg.get_trace_events("alice|t1", limit=2)) == 2


# ---------------- TTL ----------------
def test_ttl_set_on_keys(monkeypatch):
    fake = _patch_redis(monkeypatch)
    monkeypatch.setattr(C, "TRACE_TTL_DAYS", 14, raising=False)
    _rec({"type": "error", "message": "boom"})
    ttl_expect = 14 * 86400
    # 事件键、集合键、计数键都应带 TTL
    assert fake.ttl("trace:evt:alice|t1:1") == ttl_expect
    assert fake.ttl("trace:evts:alice|t1") == ttl_expect
    assert fake.ttl("trace:cnt:alice|t1") == ttl_expect
    assert fake.ttl("trace:user:alice") == ttl_expect


# ---------------- payload 上限 ----------------
def test_payload_cap_truncates_huge_trace(monkeypatch):
    _patch_redis(monkeypatch)
    monkeypatch.setattr(C, "TRACE_MAX_PAYLOAD_CHARS", 2000, raising=False)
    huge = {"steps": [{"blob": "x" * 5000} for _ in range(10)]}
    _rec({"type": "done", "trace": huge, "answer": "ok"})
    # 直接读 Redis 里的 payload,确认未超限且 trace 被截断打标
    import json as _json
    fake = conn.get_redis()
    stored = fake.hget("trace:evt:alice|t1:1", "payload")
    assert len(stored) <= 2000 + 200  # 截断后留余量
    payload = _json.loads(stored)
    assert payload.get("_trace_truncated") is True


# ---------------- 软失败 ----------------
def test_soft_fail_when_redis_unavailable(monkeypatch):
    monkeypatch.setattr(conn, "get_redis", lambda: None)
    # 不抛异常;写返回 False,读返回 [],删返回 0
    assert trace_pkg.record_trace_event({"type": "done"}, thread_id="alice|t1") is False
    assert trace_pkg.get_trace_events("alice|t1") == []
    assert trace_pkg.delete_trace_thread("alice|t1") == 0
    assert trace_pkg.delete_trace_user("alice") == 0


def test_soft_fail_when_redis_raises(monkeypatch):
    def _boom():
        raise RuntimeError("redis down")
    monkeypatch.setattr(conn, "get_redis", _boom)
    assert trace_pkg.record_trace_event({"type": "error"}, thread_id="alice|t1") is False
    assert trace_pkg.get_trace_events("alice|t1") == []


# ---------------- 开关 ----------------
def test_disabled_skips_write(monkeypatch):
    fake = _patch_redis(monkeypatch)
    monkeypatch.setattr(C, "TRACE_STORE_ENABLED", False, raising=False)
    assert trace_pkg.record_trace_event({"type": "done"}, thread_id="alice|t1") is False
    assert trace_pkg.get_trace_events("alice|t1") == []


# ---------------- 删除级联 ----------------
def test_delete_thread_removes_events(monkeypatch):
    _patch_redis(monkeypatch)
    _rec({"type": "tool_call"}, tid="alice|t1")
    _rec({"type": "done"}, tid="alice|t1")
    _rec({"type": "done"}, tid="alice|t2")
    assert trace_pkg.delete_trace_thread("alice|t1") == 2
    assert trace_pkg.get_trace_events("alice|t1") == []
    assert len(trace_pkg.get_trace_events("alice|t2")) == 1
    # 线程已从用户索引移除
    assert "alice|t1" not in trace_pkg.trace_store.list_user_threads("alice")
    assert "alice|t2" in trace_pkg.trace_store.list_user_threads("alice")


def test_delete_user_removes_all_threads(monkeypatch):
    _patch_redis(monkeypatch)
    _rec({"type": "tool_call"}, tid="alice|t1")
    _rec({"type": "error"}, tid="alice|t2")
    _rec({"type": "done"}, tid="bob|t9")
    total = trace_pkg.delete_trace_user("alice")
    assert total == 2
    assert trace_pkg.get_trace_events("alice|t1") == []
    assert trace_pkg.get_trace_events("alice|t2") == []
    # 其他用户不受影响
    assert len(trace_pkg.get_trace_events("bob|t9")) == 1
    assert trace_pkg.trace_store.list_user_threads("alice") == []


# ---------------- run_path 接线:事件旁落到追踪存储,不影响 yield ----------------
def test_run_path_records_trace_and_still_yields(monkeypatch):
    fake = _patch_redis(monkeypatch)

    def _stream():
        yield {"type": "status", "message": "…"}
        yield {"type": "tool_call", "name": "search_text"}
        yield {"type": "done", "trace": {"final_reason": "answer"}}

    from agent_reasoning.ReAct.support import runner as runner_mod
    evs = list(runner_mod.run_path(
        "react", _stream(), thread_id="alice|t1", username="alice", session_id="s1"))

    # 浏览器仍收到全部事件(含 status)
    assert [e["type"] for e in evs] == ["status", "tool_call", "done"]
    # 追踪存储只录白名单(tool_call/done),不含 status
    rec_types = [e["type"] for e in trace_pkg.get_trace_events("alice|t1")]
    assert rec_types == ["tool_call", "done"]


def test_run_path_synthetic_error_also_traced(monkeypatch):
    _patch_redis(monkeypatch)

    def _boom():
        yield {"type": "status"}
        raise RuntimeError("graph boom")

    from agent_reasoning.ReAct.support import runner as runner_mod
    evs = list(runner_mod.run_path(
        "react", _boom(), thread_id="alice|t1", username="alice"))
    # 前端收到补发的 error
    assert any(e["type"] == "error" for e in evs)
    # 追踪存储也录到了这条兜底 error
    rec_types = [e["type"] for e in trace_pkg.get_trace_events("alice|t1")]
    assert "error" in rec_types


# ---------------- 先重试、再熔断 ----------------
class _FlakyClient:
    """包一层 fakeredis:指定方法前 fail_times 次抛连接错,之后恢复正常。"""

    def __init__(self, real, fail_times=0, fail_method="incr"):
        self._real = real
        self.fail_left = fail_times
        self.fail_method = fail_method
        self.calls = 0

    def __getattr__(self, name):
        attr = getattr(self._real, name)
        if name != self.fail_method:
            return attr

        def _wrapped(*a, **k):
            self.calls += 1
            if self.fail_left > 0:
                self.fail_left -= 1
                raise ConnectionError(f"redis transient ({name})")
            return attr(*a, **k)
        return _wrapped


def test_retry_recovers_after_transient_failures(monkeypatch):
    monkeypatch.setattr(C, "TRACE_RETRY_MAX", 3, raising=False)
    monkeypatch.setattr(C, "TRACE_RETRY_BACKOFF", 0.0, raising=False)
    flaky = _FlakyClient(fakeredis.FakeRedis(decode_responses=True), fail_times=2)
    monkeypatch.setattr(conn, "get_redis", lambda: flaky)

    # 前两次 incr 失败、第三次成功 -> 重试后落库成功
    assert trace_pkg.record_trace_event(
        {"type": "done"}, thread_id="alice|t1") is True
    assert flaky.calls == 3            # 确实重试了 3 次
    assert len(trace_pkg.get_trace_events("alice|t1")) == 1


def test_circuit_opens_after_retries_exhausted_then_skips(monkeypatch):
    monkeypatch.setattr(C, "TRACE_RETRY_MAX", 3, raising=False)
    monkeypatch.setattr(C, "TRACE_RETRY_BACKOFF", 0.0, raising=False)
    monkeypatch.setattr(C, "TRACE_CIRCUIT_COOLDOWN", 30.0, raising=False)
    flaky = _FlakyClient(fakeredis.FakeRedis(decode_responses=True),
                         fail_times=10 ** 6)  # 一直失败
    monkeypatch.setattr(conn, "get_redis", lambda: flaky)

    # 第一次:重试 3 次全失败 -> 熔断开路,返回 False
    assert trace_pkg.record_trace_event(
        {"type": "error"}, thread_id="alice|t1") is False
    assert flaky.calls == 3
    assert trace_store_mod._circuit_open is True

    # 冷却期内:后续操作直接跳过,不再碰 Redis(calls 不增长、不阻塞)
    assert trace_pkg.record_trace_event(
        {"type": "done"}, thread_id="alice|t1") is False
    assert trace_pkg.get_trace_events("alice|t1") == []
    assert flaky.calls == 3


def test_circuit_half_open_recovers_after_cooldown(monkeypatch):
    monkeypatch.setattr(C, "TRACE_RETRY_MAX", 2, raising=False)
    monkeypatch.setattr(C, "TRACE_RETRY_BACKOFF", 0.0, raising=False)
    # 先一直失败 -> 熔断
    flaky = _FlakyClient(fakeredis.FakeRedis(decode_responses=True),
                         fail_times=10 ** 6)
    monkeypatch.setattr(conn, "get_redis", lambda: flaky)
    assert trace_pkg.record_trace_event(
        {"type": "error"}, thread_id="alice|t1") is False
    assert trace_store_mod._circuit_open is True
    calls_at_open = flaky.calls

    # 冷却期已过(半开):换一个健康客户端,下一次操作探测成功 -> 闭合复位
    healthy = fakeredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(conn, "get_redis", lambda: healthy)
    trace_store_mod._circuit_open_until = 0.0  # 强制视为冷却已过

    assert trace_pkg.record_trace_event(
        {"type": "done"}, thread_id="alice|t1") is True
    assert trace_store_mod._circuit_open is False        # 已闭合
    assert len(trace_pkg.get_trace_events("alice|t1")) == 1
    # 半开探测只尝试一次(失败的 flaky 没再被调用)
    assert flaky.calls == calls_at_open


def test_client_unavailable_trips_without_retry(monkeypatch):
    # redis 包缺失/客户端建不出来属非瞬时:不重试,直接熔断
    monkeypatch.setattr(C, "TRACE_RETRY_MAX", 5, raising=False)
    monkeypatch.setattr(conn, "get_redis", lambda: None)
    assert trace_pkg.record_trace_event(
        {"type": "done"}, thread_id="alice|t1") is False
    assert trace_store_mod._circuit_open is True
    # 冷却期内读取也安全返回 []
    assert trace_pkg.get_trace_events("alice|t1") == []
