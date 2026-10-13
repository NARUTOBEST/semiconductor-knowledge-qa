# -*- coding: utf-8 -*-
"""检索服务守卫测试:
  - 服务间鉴权:RETRIEVAL_INTERNAL_TOKEN 非空时所有端点(除 /health)要求
    X-Internal-Token(HTTP 依赖 + /mcp 挂载的 ASGI 中间件两路都生效);
  - 推理互斥:ENCODE_LOCK 下 encode 并发峰值恒为 1;
  - 单飞批合并:并发单条请求被攒成批量调用,结果按位切分回填、形状与单条一致。
全部用打桩编码器/打桩 engine_api,不加载真实模型、不连 Qdrant。
"""
import asyncio
import threading
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

import config as C
from mcp_servers.retrieval import engine_api
from mcp_servers.retrieval import service as svc
import embed  # noqa: E402  (RAG/embed.py)


# ==================== 服务间鉴权 ====================
class TestTokenAuth:
    def test_token_ok_semantics(self, monkeypatch):
        monkeypatch.setattr(C, "RETRIEVAL_INTERNAL_TOKEN", "")
        assert svc._token_ok("anything") is True           # 留空 = 鉴权关闭
        monkeypatch.setattr(C, "RETRIEVAL_INTERNAL_TOKEN", "s3cret")
        assert svc._token_ok("s3cret") is True
        assert svc._token_ok("wrong") is False
        assert svc._token_ok("") is False                  # 未携带

    def test_http_403_without_or_wrong_token(self, monkeypatch):
        monkeypatch.setattr(C, "RETRIEVAL_INTERNAL_TOKEN", "s3cret")
        client = TestClient(svc.app)   # 不用 with:跳过 lifespan(session_manager 仅可 run 一次)
        r1 = client.post("/get_chunk", json={"chunk_id": "abc__t00001"})
        r2 = client.post("/get_chunk", json={"chunk_id": "abc__t00001"},
                         headers={"X-Internal-Token": "wrong"})
        assert r1.status_code == 403 and r2.status_code == 403

    def test_http_passes_with_token(self, monkeypatch):
        monkeypatch.setattr(C, "RETRIEVAL_INTERNAL_TOKEN", "s3cret")
        monkeypatch.setattr(engine_api, "get_chunk",
                            lambda chunk_id: {"chunk_id": chunk_id})
        client = TestClient(svc.app)
        r = client.post("/get_chunk", json={"chunk_id": "abc__t00001"},
                        headers={"X-Internal-Token": "s3cret"})
        assert r.status_code == 200
        assert r.json()["chunk_id"] == "abc__t00001"

    def test_health_exempt(self, monkeypatch):
        monkeypatch.setattr(C, "RETRIEVAL_INTERNAL_TOKEN", "s3cret")
        client = TestClient(svc.app)
        assert client.get("/health").status_code == 200

    def test_asgi_guard_on_mcp_mount(self, monkeypatch):
        """/mcp 是挂载的 Starlette 子应用,守卫以 ASGI 中间件形态生效。"""
        monkeypatch.setattr(C, "RETRIEVAL_INTERNAL_TOKEN", "s3cret")
        called = []

        async def inner(scope, receive, send):
            called.append(scope)

        guard = svc._TokenGuardASGI(inner)

        async def run(headers):
            msgs = []
            scope = {"type": "http", "headers": headers}

            async def receive():
                return {"type": "http.request", "body": b"", "more_body": False}

            async def send(m):
                msgs.append(m)

            await guard(scope, receive, send)
            return msgs

        # 错 token:403 且不进内层
        msgs = asyncio.run(run([(b"x-internal-token", b"nope")]))
        assert msgs[0]["status"] == 403 and not called
        # 对 token:进入内层
        asyncio.run(run([(b"x-internal-token", b"s3cret")]))
        assert len(called) == 1
        # 鉴权关闭(留空):无头放行
        monkeypatch.setattr(C, "RETRIEVAL_INTERNAL_TOKEN", "")
        asyncio.run(run([]))
        assert len(called) == 2


# ==================== 推理互斥 + 单飞批合并 ====================
class _FakeEncoder:
    """打桩编码器:记录每次调用的批量与并发峰值;dense 行值 = text 的 float。"""

    def __init__(self, delay=0.0, fail=False):
        self.delay = delay
        self.fail = fail
        self.batches = []          # 每次 encode 收到的 text 列表
        self.cur = 0
        self.max_conc = 0
        self._lock = threading.Lock()

    def encode(self, texts):
        with self._lock:
            self.cur += 1
            self.max_conc = max(self.max_conc, self.cur)
            self.batches.append(list(texts))
        time.sleep(self.delay)
        with self._lock:
            self.cur -= 1
        if self.fail:
            raise RuntimeError("boom")
        dense = np.array([[float(t)] * 4 for t in texts], dtype=np.float32)
        return dense, [{"t": t} for t in texts]


class TestBatchedEncoder:
    def test_infer_lock_serializes(self):
        """ENCODE_LOCK 下,encode 的进程内并发峰值恒为 1(拆锁后同模型仍互斥)。"""
        fake = _FakeEncoder(delay=0.02)
        w = embed._BatchedEncoder(fake, window_s=0.01, max_batch=8)
        errors = []

        def go(i):
            try:
                w.encode([str(i)])
            except Exception as e:  # pragma: no cover
                errors.append(e)

        ts = [threading.Thread(target=go, args=(i,)) for i in range(6)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        assert errors == []
        assert fake.max_conc == 1
        assert sum(len(b) for b in fake.batches) == 6   # 无请求丢失

    def test_single_flight_merges(self):
        """窗口内到达的单条请求被合并进一次批量调用,结果按位切分回填。"""
        fake = _FakeEncoder(delay=0.08)
        w = embed._BatchedEncoder(fake, window_s=0.03, max_batch=16)
        out, errors = {}, []

        def go(i):
            try:
                dense, sparse = w.encode([str(i)])
                out[i] = (dense[0][0], sparse[0]["t"])
            except Exception as e:
                errors.append(e)

        ts = [threading.Thread(target=go, args=(i,)) for i in range(8)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        assert errors == []
        assert len(fake.batches) < 8                    # 发生了合并
        for i in range(8):
            assert out[i] == (float(i), str(i))         # 切分不串位

    def test_multi_text_direct_and_split(self):
        """多条(批量入库)不经窗口,直接持锁整批执行;超 max_batch 切分分批。"""
        fake = _FakeEncoder()
        w = embed._BatchedEncoder(fake, window_s=0.01, max_batch=2)
        dense, sparse = w.encode(["1", "2", "3", "4", "5"])
        assert dense.shape == (5, 4) and len(sparse) == 5
        assert [len(b) for b in fake.batches] == [2, 2, 1]
        assert fake.max_conc == 1
        assert sparse[3] == {"t": "4"}

    def test_error_propagates(self):
        """批量调用抛错:单条(攒批路径)与多条(直接路径)都收到同一异常。"""
        fake = _FakeEncoder(fail=True)
        w = embed._BatchedEncoder(fake, window_s=0.01, max_batch=4)
        with pytest.raises(RuntimeError, match="boom"):
            w.encode(["x"])
        with pytest.raises(RuntimeError, match="boom"):
            w.encode(["x", "y"])


# ==================== _dynamic_cut 断崖截断 ====================
class TestDynamicCutCliff:
    """τ 绝对阈值之后的断崖检测:金标块群分数扎堆,群后断崖即截断。"""

    def _cut(self, scores, monkeypatch, tau=0.3, max_k=8, min_k=3, cliff=0.6,
             fixed=0):
        monkeypatch.setattr(engine_api, "_reranker_failed", False)
        monkeypatch.setattr(C, "RERANK_TAU", tau)
        monkeypatch.setattr(C, "RERANK_MAX_K", max_k)
        monkeypatch.setattr(C, "RERANK_MIN_K", min_k)
        monkeypatch.setattr(C, "RERANK_CLIFF_RATIO", cliff)
        monkeypatch.setattr(C, "RERANK_FIXED_K", fixed)   # 默认旁路固定模式
        ranked = [(f"p{i}", s) for i, s in enumerate(scores)]
        return engine_api._dynamic_cut(ranked, k=3)

    def test_cliff_truncates_after_gap(self, monkeypatch):
        # 0.9/0.85/0.8 扎堆,0.4 < 0.6*0.8 断崖 -> 只留前 3
        out = self._cut([0.9, 0.85, 0.8, 0.4, 0.35, 0.3], monkeypatch)
        assert [p for p, _ in out] == ["p0", "p1", "p2"]

    def test_no_cliff_keeps_tau_passed(self, monkeypatch):
        out = self._cut([0.9, 0.85, 0.8, 0.75, 0.7, 0.65], monkeypatch)
        assert len(out) == 6            # 平缓下降无断崖,τ 过线全留

    def test_cliff_keeps_at_least_top1(self, monkeypatch):
        out = self._cut([0.9, 0.3], monkeypatch)   # 0.3 < 0.6*0.9 -> 保底 top-1
        assert [p for p, _ in out] == ["p0"]

    def test_cliff_disabled_by_zero(self, monkeypatch):
        out = self._cut([0.9, 0.85, 0.8, 0.4, 0.35, 0.3], monkeypatch, cliff=0)
        assert len(out) == 6

    def test_tau_fallback_still_cliffed(self, monkeypatch):
        # 全部低于 τ -> 保底 MIN_K=3(p0..p2),断崖在保底集上继续生效
        out = self._cut([0.25, 0.2, 0.1, 0.05], monkeypatch)
        assert [p for p, _ in out] == ["p0", "p1"]

    def test_reranker_failed_bypasses_cliff(self, monkeypatch):
        monkeypatch.setattr(engine_api, "_reranker_failed", True)
        monkeypatch.setattr(C, "RERANK_CLIFF_RATIO", 0.6)
        ranked = [(f"p{i}", s) for i, s in enumerate([5.0, 0.1, 0.05, 0.01])]
        assert len(engine_api._dynamic_cut(ranked, k=3)) == 3  # RRF 分不同标,不走断崖


# ==================== _dynamic_cut 固定 top-K ====================
class TestDynamicCutFixed:
    """RERANK_FIXED_K>0:重排后固定取 top-K,τ/断崖全旁路(评测对照基线)。"""

    def _cut(self, scores, monkeypatch, fixed=5, tau=0.3, cliff=0.6):
        monkeypatch.setattr(engine_api, "_reranker_failed", False)
        monkeypatch.setattr(C, "RERANK_FIXED_K", fixed)
        monkeypatch.setattr(C, "RERANK_TAU", tau)
        monkeypatch.setattr(C, "RERANK_CLIFF_RATIO", cliff)
        ranked = [(f"p{i}", s) for i, s in enumerate(scores)]
        return engine_api._dynamic_cut(ranked, k=3)

    def test_fixed_takes_top5_regardless_of_score(self, monkeypatch):
        # 低分块也照留——固定模式不看分数,τ/断崖均不生效
        out = self._cut([0.9, 0.8, 0.7, 0.1, 0.05, 0.01, 0.001], monkeypatch)
        assert [p for p, _ in out] == ["p0", "p1", "p2", "p3", "p4"]

    def test_fixed_pool_smaller_than_k(self, monkeypatch):
        out = self._cut([0.9, 0.8], monkeypatch, fixed=5)
        assert [p for p, _ in out] == ["p0", "p1"]   # 池不足 K 返回全部

    def test_fixed_bypasses_cliff(self, monkeypatch):
        # 有断崖(0.4<0.6*0.8)也照取 5 个
        out = self._cut([0.9, 0.8, 0.4, 0.35, 0.3, 0.2], monkeypatch)
        assert len(out) == 5

    def test_fixed_zero_falls_back_to_dynamic(self, monkeypatch):
        out = self._cut([0.9, 0.8, 0.4, 0.35, 0.3, 0.2], monkeypatch, fixed=0)
        assert [p for p, _ in out] == ["p0", "p1"]   # 回到 τ+断崖

    def test_reranker_failed_still_rrf_k(self, monkeypatch):
        monkeypatch.setattr(engine_api, "_reranker_failed", True)
        monkeypatch.setattr(C, "RERANK_FIXED_K", 5)
        ranked = [(f"p{i}", s) for i, s in enumerate([5.0, 0.1, 0.05, 0.01])]
        # 回退路径不受固定模式影响,按调用方 k 截断
        assert len(engine_api._dynamic_cut(ranked, k=3)) == 3
