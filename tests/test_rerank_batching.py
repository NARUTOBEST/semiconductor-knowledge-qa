# -*- coding: utf-8 -*-
"""重排跨请求攒批 + 多副本管线单测(不依赖模型)。

背景(20261009 重排吞吐改造):
  - RERANK_RECALL_K=96 时单请求 96 对,20 并发逐请求持全局锁串行会把队尾
    拖过 MCP 30s 超时(轮次13 实测);
  - _BatchedReranker 把窗口内并发请求合并成一次 compute_score(单飞批同款),
    副本数 >1 时多 leader 各占空闲副本并行。
"""
import threading
import time

import pytest

from embed import _BatchedReranker, _RerankSlot


class _FakeReplica:
    """模拟 gemma 重排副本:记录 score_pairs 调用,可选延迟/异常/阻塞。"""
    def __init__(self, delay=0.0, error=None, gate=None):
        self.delay = delay
        self.error = error
        self.gate = gate            # threading.Event:置位后才返回
        self.calls = []             # 每次 score_pairs 的 pairs 快照
        self.lock = threading.Lock()

    def score_pairs(self, pairs, batch_size=8):
        self.calls.append(list(pairs))
        if self.gate is not None:
            self.gate.wait(timeout=5.0)
        if self.delay:
            time.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return [float(i) for i, _ in enumerate(pairs)]


def _pipeline(replicas, **kw):
    d = dict(window_s=0.0, max_pairs=96)
    d.update(kw)
    return _BatchedReranker([(r, r.lock) for r in replicas], **d)


# ==================== 基本语义 ====================
def test_single_request_passthrough():
    r = _FakeReplica()
    p = _pipeline([r])
    out = p.rerank("q", ["d0", "d1", "d2"])
    assert out == [0.0, 1.0, 2.0]
    assert len(r.calls) == 1
    assert r.calls[0] == [["q", "d0"], ["q", "d1"], ["q", "d2"]]


def test_concurrent_requests_merged_and_split():
    r = _FakeReplica()
    p = _pipeline([r], window_s=0.05)          # 留窗口收拢
    outs = [None] * 3

    def go(i, n):
        outs[i] = p.rerank(f"q{i}", [f"q{i}-d{j}" for j in range(n)])

    ts = [threading.Thread(target=go, args=(i, 2)) for i in range(3)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    # 合并成一次调用(窗口内全到齐):6 对,按各请求文档数切分回填
    assert len(r.calls) == 1
    assert len(r.calls[0]) == 6
    assert outs[0] == [0.0, 1.0]
    assert outs[1] == [2.0, 3.0]
    assert outs[2] == [4.0, 5.0]


def test_max_pairs_chunking():
    r = _FakeReplica()
    p = _pipeline([r], window_s=0.05, max_pairs=4)
    outs = [None] * 3

    def go(i):
        outs[i] = p.rerank(f"q{i}", [f"d{j}" for j in range(3)])

    ts = [threading.Thread(target=go, args=(i,)) for i in range(3)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert all(o == [0.0, 1.0, 2.0] for o in outs)
    assert [len(c) for c in r.calls] == [3, 3, 3]   # 3+3>4:每请求独自成批


def test_failure_propagates_to_whole_batch():
    r = _FakeReplica(error=RuntimeError("gpu oom"))
    p = _pipeline([r], window_s=0.05)
    errs = [None] * 2

    def go(i):
        try:
            p.rerank("q", ["d0", "d1"])
        except Exception as e:  # noqa: BLE001
            errs[i] = e

    ts = [threading.Thread(target=go, args=(i,)) for i in range(2)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert all(isinstance(e, RuntimeError) for e in errs)   # 整批同享异常


def test_document_truncation_applied(monkeypatch):
    import config as C
    monkeypatch.setattr(C, "RERANK_MAX_CONTENT", 800, raising=False)
    r = _FakeReplica()
    p = _pipeline([r])
    p.rerank("q", ["x" * 2000, "short"])
    assert len(r.calls[0][0][1]) == 800
    assert r.calls[0][1][1] == "short"


def test_oversize_single_request_goes_alone():
    r = _FakeReplica()
    p = _pipeline([r], window_s=0.05, max_pairs=4)
    out = p.rerank("q", [f"d{j}" for j in range(10)])       # 单请求超限:独自成批不卡死
    assert out == [float(i) for i in range(10)]
    assert len(r.calls) == 1 and len(r.calls[0]) == 10


# ==================== 多副本并行 ====================
def test_two_replicas_run_concurrently():
    gate0 = threading.Event()
    r0, r1 = _FakeReplica(gate=gate0), _FakeReplica()
    p = _pipeline([r0, r1], window_s=0.05)
    done1 = threading.Event()
    holder = {}

    def go0():
        holder["r0"] = p.rerank("q0", ["a", "b"])   # 占住副本0(等 gate0)

    def go1():
        holder["r1"] = p.rerank("q1", ["c", "d"])
        done1.set()

    t0 = threading.Thread(target=go0)
    t0.start()
    time.sleep(0.15)                    # leader0 已进副本0(阻塞在 gate)
    t1 = threading.Thread(target=go1)
    t1.start()
    assert done1.wait(timeout=3.0)      # 副本1 空闲:请求1 不等请求0
    gate0.set()
    t0.join(timeout=3.0)
    t1.join(timeout=3.0)
    assert holder["r0"] == [0.0, 1.0]
    assert holder["r1"] == [0.0, 1.0]
    assert len(r0.calls) == 1 and len(r1.calls) == 1


def test_all_busy_replicas_fall_back_to_first_lock():
    """副本全忙时新 leader 等第一个副本(锁序一致,不死锁)。"""
    gate = threading.Event()
    r0, r1 = _FakeReplica(gate=gate), _FakeReplica(gate=gate)
    p = _pipeline([r0, r1], window_s=0.0)
    outs = [None] * 3
    ts = [threading.Thread(target=lambda i=i: outs.__setitem__(
        i, p.rerank(f"q{i}", ["d"]))) for i in range(3)]
    gate.set()                          # 三个 leader 两个副本:第三个等锁
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=5.0)
    assert all(not t.is_alive() for t in ts)
    assert all(o == [0.0] for o in outs)
    assert len(r0.calls) + len(r1.calls) == 3


# ==================== slot 语义 ====================
def test_slot_done_property():
    s = _RerankSlot("q", ["d"])
    assert not s.done
    s.result = [1.0]
    assert s.done
    s2 = _RerankSlot("q", ["d"])
    s2.error = RuntimeError("x")
    assert s2.done
