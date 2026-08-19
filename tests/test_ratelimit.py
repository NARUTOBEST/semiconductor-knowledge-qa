# -*- coding: utf-8 -*-
"""ratelimit tests: per-user lock + global semaphore + queue timeout."""
import pytest, threading, time
from fastapi import HTTPException

class TestPerUserLimit:
    def test_acquire_and_release(self, isolated_ratelimit):
        rl = isolated_ratelimit
        rl.acquire_user_slot("u1")
        rl.release_user_slot("u1")
        rl.acquire_user_slot("u1")
        rl.release_user_slot("u1")

    def test_concurrent_same_user_blocked(self, isolated_ratelimit):
        rl = isolated_ratelimit
        rl.acquire_user_slot("u1")
        with pytest.raises(HTTPException) as exc:
            rl.acquire_user_slot("u1")
        assert exc.value.status_code == 429
        rl.release_user_slot("u1")

    def test_different_users_independent(self, isolated_ratelimit):
        rl = isolated_ratelimit
        rl.acquire_user_slot("u1")
        rl.acquire_user_slot("u2")
        rl.release_user_slot("u1")
        rl.release_user_slot("u2")

    def test_release_then_reacquire(self, isolated_ratelimit):
        rl = isolated_ratelimit
        rl.acquire_user_slot("u1")
        rl.release_user_slot("u1")
        rl.acquire_user_slot("u1")
        rl.release_user_slot("u1")

class TestGlobalLimit:
    def test_within_limit(self, isolated_ratelimit):
        rl = isolated_ratelimit
        rl.acquire_global_slot("u1")
        rl.acquire_global_slot("u2")
        rl.release_global_slot()
        rl.release_global_slot()

    def test_exceed_limit_times_out(self, isolated_ratelimit):
        rl = isolated_ratelimit
        rl.acquire_global_slot("u1")
        rl.acquire_global_slot("u2")
        t0 = time.time()
        with pytest.raises(HTTPException) as exc:
            rl.acquire_global_slot("u3")
        elapsed = time.time() - t0
        assert exc.value.status_code == 429
        assert elapsed >= 0.3
        rl.release_global_slot()
        rl.release_global_slot()

    def test_release_allows_new_acquire(self, isolated_ratelimit):
        rl = isolated_ratelimit
        rl.acquire_global_slot("u1")
        rl.acquire_global_slot("u2")
        rl.release_global_slot()
        rl.acquire_global_slot("u3")
        rl.release_global_slot()
        rl.release_global_slot()

    def test_release_all(self, isolated_ratelimit):
        rl = isolated_ratelimit
        rl.acquire_user_slot("u1")
        rl.acquire_global_slot("u1")
        rl.release_all("u1")
        rl.acquire_user_slot("u1")
        rl.acquire_global_slot("u1")
        rl.release_all("u1")

    def test_global_exhausted_rejects_multiple(self, isolated_ratelimit):
        """When global slots are exhausted, multiple users are rejected."""
        rl = isolated_ratelimit
        rl.acquire_global_slot("u1")
        rl.acquire_global_slot("u2")
        with pytest.raises(HTTPException) as exc1:
            rl.acquire_global_slot("u3")
        assert exc1.value.status_code == 429
        with pytest.raises(HTTPException) as exc2:
            rl.acquire_global_slot("u4")
        assert exc2.value.status_code == 429
        rl.release_global_slot()
        rl.acquire_global_slot("u3")
        rl.release_global_slot()
        rl.release_global_slot()

class TestConcurrencySimulation:
    def test_two_users_concurrent_ok(self, isolated_ratelimit):
        rl = isolated_ratelimit
        results = []
        def worker(name):
            try:
                rl.acquire_user_slot(name)
                rl.acquire_global_slot(name)
                results.append("ok")
                time.sleep(0.05)
                rl.release_all(name)
            except HTTPException:
                results.append("rejected")
        t1 = threading.Thread(target=worker, args=("u1",))
        t2 = threading.Thread(target=worker, args=("u2",))
        t1.start(); t2.start()
        t1.join(); t2.join()
        assert results.count("ok") == 2
