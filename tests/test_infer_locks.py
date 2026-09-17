# -*- coding: utf-8 -*-
"""推理锁拆分与入库让路/异步化测试。

背景(吞吐瓶颈改造):
  - A: INFER_LOCK 拆为 ENCODE_LOCK/IMAGE_LOCK/RERANK_LOCK,不同模型并行前向;
  - B1: 批量(入库)编码在批间检查在线排队,有人等先退避让路;
  - B2: /ingest_document 支持 wait=False 后台执行 + /ingest_status 查进度。
"""
import threading
import time
import types

import numpy as np
import pytest
from fastapi.testclient import TestClient

import embed
from embed import _BatchedEncoder


class _SlowEncoder:
    """模拟 BGE-m3:每次前向睡固定时长,返回可按轴0切分的形状。"""
    def __init__(self, delay=0.2):
        self.delay = delay

    def encode(self, texts):
        time.sleep(self.delay)
        dense = np.zeros((len(texts), 4), dtype=np.float32)
        return dense, [{} for _ in texts]


class TestLockSplit:
    def test_encode_and_rerank_run_in_parallel(self):
        """encode(ENCODE_LOCK) 与 rerank(RERANK_LOCK) 不同锁:并发总耗时≈单次,而非相加。"""
        w = _BatchedEncoder(_SlowEncoder(0.2), window_s=0.0, max_batch=8,
                            lock=embed.ENCODE_LOCK)

        def do_encode():
            w.encode(["hello"])

        def do_rerank():
            with embed.RERANK_LOCK:
                time.sleep(0.2)

        t0 = time.monotonic()
        ts = [threading.Thread(target=do_encode),
              threading.Thread(target=do_rerank)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        elapsed = time.monotonic() - t0
        assert elapsed < 0.38   # 串行应 ~0.4s;并行应 ~0.2s

    def test_same_model_still_serialized(self):
        """同一 _BatchedEncoder 的并发前向峰值仍为 1(拆锁不放松同模型互斥)。"""
        class _Conc(_SlowEncoder):
            def __init__(self):
                super().__init__(0.03)
                self.cur = 0
                self.max_conc = 0
                self._lk = threading.Lock()

            def encode(self, texts):
                with self._lk:
                    self.cur += 1
                    self.max_conc = max(self.max_conc, self.cur)
                time.sleep(self.delay)
                with self._lk:
                    self.cur -= 1
                dense = np.zeros((len(texts), 4), dtype=np.float32)
                return dense, [{} for _ in texts]

        fake = _Conc()
        w = _BatchedEncoder(fake, window_s=0.0, max_batch=8,
                            lock=embed.ENCODE_LOCK)
        errs = []

        def go(i):
            try:
                w.encode([str(i)])
            except Exception as e:  # pragma: no cover
                errs.append(e)

        ts = [threading.Thread(target=go, args=(i,)) for i in range(6)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        assert errs == []
        assert fake.max_conc == 1


class TestIngestYield:
    def test_batch_yields_between_chunks_when_online_waiting(self, monkeypatch):
        """批量(入库)编码在批间发现在线排队 → 每批之间退避一拍。"""
        w = _BatchedEncoder(_SlowEncoder(0.0), window_s=0.0, max_batch=2,
                            lock=embed.ENCODE_LOCK)
        monkeypatch.setattr(embed, "_INGEST_YIELD_S", 0.03)
        sleeps = []
        monkeypatch.setattr(embed.time, "sleep",
                            lambda s: sleeps.append(s))
        # 在线队列非空:模拟有单条请求在等
        monkeypatch.setattr(w, "_yield_check", lambda: True)

        dense, sparse = w._run_batched(["a", "b", "c", "d"])
        assert dense.shape == (4, 4) and len(sparse) == 4   # 结果完整拼接
        yields = [s for s in sleeps if s > 0]               # 过滤 encoder 自身 sleep(0)
        assert len(yields) == 1                             # 2 批之间让路 1 次
        assert yields[0] == pytest.approx(0.03)

    def test_no_yield_when_nobody_waiting(self, monkeypatch):
        w = _BatchedEncoder(_SlowEncoder(0.0), window_s=0.0, max_batch=2,
                            lock=embed.ENCODE_LOCK)
        monkeypatch.setattr(embed, "_INGEST_YIELD_S", 0.03)
        sleeps = []
        monkeypatch.setattr(embed.time, "sleep", lambda s: sleeps.append(s))
        monkeypatch.setattr(w, "_yield_check", lambda: False)
        w._run_batched(["a", "b", "c", "d"])
        assert [s for s in sleeps if s > 0] == []


class TestDualEncoder:
    """双实例隔离:在线查询与批量入库各一份模型副本、各一把锁。"""

    @pytest.fixture()
    def fake_models(self, monkeypatch):
        """替换 TextEncoder/ImageEncoder 构造(不真加载模型),清空单例。"""
        monkeypatch.setattr(embed, "TextEncoder", lambda: object())
        monkeypatch.setattr(embed, "ImageEncoder", lambda: object())
        monkeypatch.setattr(embed, "_te", None)
        monkeypatch.setattr(embed, "_te_offline", None)
        monkeypatch.setattr(embed, "_ie", None)
        monkeypatch.setattr(embed, "_ie_offline", None)

    def test_offline_is_distinct_instance_with_own_lock(self, fake_models):
        online = embed.get_text_encoder()
        offline = embed.get_text_encoder(offline=True)
        assert online is not offline
        assert offline._infer_lock is embed.INGEST_ENCODE_LOCK
        assert online._infer_lock is embed.ENCODE_LOCK
        # CLIP 同理
        assert embed.get_image_encoder() is not embed.get_image_encoder(offline=True)
        assert embed.get_image_encoder(offline=True)._infer_lock \
            is embed.INGEST_IMAGE_LOCK

    def test_offline_yield_watches_online_queue(self, fake_models):
        """离线实例的让路判断看【在线实例】排队(双实例无锁竞争,但共享 CPU 核)。"""
        online = embed.get_text_encoder()
        offline = embed.get_text_encoder(offline=True)
        assert offline._yield_check() is False      # 在线空闲
        slot = embed._Slot("q")
        with online._cond:
            online._pending.append(slot)             # 在线有请求在排队
        assert offline._yield_check() is True

    def test_dual_disabled_falls_back_to_single(self, fake_models, monkeypatch):
        """RETRIEVAL_DUAL_ENCODER=0:离线请求退回在线单实例(B1 让路兜底)。"""
        monkeypatch.setattr(embed, "_DUAL", False)
        assert embed.get_text_encoder(offline=True) is embed.get_text_encoder()
        assert embed.get_image_encoder(offline=True) is embed.get_image_encoder()


# ---------------- B2: /ingest_document 异步模式 ----------------
class TestIngestAsync:
    @pytest.fixture()
    def client(self):
        from mcp_servers.retrieval import service as svc
        # 鉴权:env 未设 RETRIEVAL_INTERNAL_TOKEN 时依赖直接放行(测试默认)
        return TestClient(svc.app), svc

    def test_wait_false_returns_task_id_and_status(self, client, monkeypatch):
        tc, svc = client
        monkeypatch.setattr(svc, "_run_ingest",
                            lambda req, task_id=None: (
                                svc._set_ingest_task(task_id, status="done",
                                                     text_count=3, image_count=1)
                                if task_id else None) or {"ok": True})
        r = tc.post("/ingest_document", json={
            "stem": "s", "auto_dir": "d", "source_path": "p", "wait": False})
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] and body["status"] == "running" and body["task_id"]

        # 后台线程把状态置为 done → /ingest_status 可查
        tid = body["task_id"]
        deadline = time.time() + 5
        while time.time() < deadline:
            st = tc.get(f"/ingest_status/{tid}").json()
            if st.get("status") == "done":
                break
            time.sleep(0.05)
        assert st["status"] == "done"
        assert st["text_count"] == 3

    def test_ingest_status_unknown_task(self, client):
        tc, _ = client
        assert tc.get("/ingest_status/nope").json()["ok"] is False

    def test_wait_true_sync_backward_compatible(self, client, monkeypatch):
        """wait=True(默认):同步返回最终计数,与旧行为一致。"""
        tc, svc = client
        monkeypatch.setattr(svc, "_run_ingest",
                            lambda req, task_id=None:
                            {"ok": True, "text_count": 5, "image_count": 2})
        r = tc.post("/ingest_document", json={
            "stem": "s", "auto_dir": "d", "source_path": "p"})
        assert r.json() == {"ok": True, "text_count": 5, "image_count": 2}
