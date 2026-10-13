# -*- coding: utf-8 -*-
"""LLM 出口波次并发闸单测(_LLMGate)。

规格(2026-10-10 用户拍板):在途 ≤size 直通;超出的排队,当前波全部结束后
攒一批(≤size)齐发下一波,直到队列穷尽;排队超时报"稍后重试"。
"""
import threading
import time

import pytest

from agent_reasoning.ReAct.support.llm import _LLMGate


def test_low_load_passes_through():
    """低负载(<size 且无排队)直通,零等待。"""
    g = _LLMGate(3, wait_timeout=5)
    for _ in range(3):
        g.acquire()          # 全部立即返回
    assert g.stats == {"inflight": 3, "waiters": 0}
    for _ in range(3):
        g.release()
    assert g.stats == {"inflight": 0, "waiters": 0}


def test_wave_capped_and_drains_queue():
    """9 个请求、闸=3:任何时刻在途 ≤3;排队全部穷尽后都完成。"""
    g = _LLMGate(3, wait_timeout=10)
    lock = threading.Lock()
    state = {"cur": 0, "max": 0}

    def work():
        g.acquire()
        try:
            with lock:
                state["cur"] += 1
                state["max"] = max(state["max"], state["cur"])
            time.sleep(0.03)
        finally:
            with lock:
                state["cur"] -= 1
            g.release()

    threads = [threading.Thread(target=work) for _ in range(9)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15)
    assert state["max"] <= 3                    # 任何时刻在途不超闸值
    assert all(not t.is_alive() for t in threads)
    assert g.stats == {"inflight": 0, "waiters": 0}


def test_wave_release_is_batched():
    """闸=2,4 个请求:后 2 个必须等当前波【清零】才一起进(攒波齐发),
    而不是每释放一个名额放一个(semaphore 行为)。"""
    g = _LLMGate(2, wait_timeout=10)
    holders_in = threading.Barrier(3)   # 主线程 + 2 个占波者
    block = threading.Event()
    lock = threading.Lock()
    wave2 = {"n": 0, "both": threading.Event()}

    def holder():
        g.acquire()
        holders_in.wait(5)
        block.wait(10)
        g.release()

    def joiner():
        g.acquire()
        with lock:
            wave2["n"] += 1
            if wave2["n"] >= 2:
                wave2["both"].set()     # 两个同时进=攒批放行
        g.release()

    ths = [threading.Thread(target=holder) for _ in range(2)]
    for t in ths:
        t.start()
    holders_in.wait(5)
    assert g.stats["inflight"] == 2 and g.stats["waiters"] == 0

    ts = [threading.Thread(target=joiner) for _ in range(2)]
    for t in ts:
        t.start()
    time.sleep(0.2)                     # 排队者已在等,波仍被占
    assert g.stats == {"inflight": 2, "waiters": 2}

    block.set()                         # 当前波清零 -> 攒批放下一波
    for t in ths:
        t.join(timeout=10)
    assert wave2["both"].wait(10), "两个排队者未同时进第二波"
    for t in ts:
        t.join(timeout=10)
    assert g.stats == {"inflight": 0, "waiters": 0}


def test_queue_wait_timeout():
    """闸满且波不结束:排队超时抛 TimeoutError,且不占名额、不影响后续。"""
    g = _LLMGate(1, wait_timeout=0.3)
    g.acquire()
    with pytest.raises(TimeoutError):
        g.acquire()                     # 闸满,超时
    assert g.stats == {"inflight": 1, "waiters": 0}   # 超时者已退出队列
    g.release()
    g.acquire()                         # 释放后恢复正常
    g.release()


def test_late_arrival_queues_behind_waiters():
    """有排队者时,后到者即使当前波有空位也必须排队(保序,不插队);
    且波未清零(还有在途)不放行排队者。"""
    g = _LLMGate(2, wait_timeout=10)
    g.acquire()
    g.acquire()                         # 占满当前波

    def passthrough(name):
        g.acquire()
        g.release()

    tw = threading.Thread(target=passthrough, args=("w",))
    tw.start()
    time.sleep(0.15)
    assert g.stats["waiters"] == 1      # 波满,排队

    tl = threading.Thread(target=passthrough, args=("l",))
    tl.start()
    time.sleep(0.15)
    assert g.stats["waiters"] == 2      # 波未满(有空位)但仍排队,不插队

    g.release()                         # 只放一个名额:波未清零
    time.sleep(0.15)
    assert g.stats == {"inflight": 1, "waiters": 2}   # 不放行

    g.release()                         # 波清零 -> 两个排队者一起进
    tw.join(timeout=10)
    tl.join(timeout=10)
    assert g.stats == {"inflight": 0, "waiters": 0}
