# -*- coding: utf-8 -*-
"""Pytest global config: sys.path injection, env isolation, shared fixtures."""
import os
import sys
import tempfile

# ---- 1. sys.path injection (must happen before any project import) ----
_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in [
    _PROJECT,
    os.path.join(_PROJECT, "config"),
    os.path.join(_PROJECT, "RAG"),
    os.path.join(_PROJECT, "server"),
    os.path.join(_PROJECT, "context management"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import pytest

# ---- 2. Redirect AUTH_DB_PATH before auth package is imported ----
import config as C
_REDIRECTED_AUTH_DB = os.path.join(tempfile.gettempdir(), "semi_agent_pytest_auth.db")
C.AUTH_DB_PATH = _REDIRECTED_AUTH_DB


# ---- 3. Shared fixtures ----

@pytest.fixture
def tmp_auth_db(tmp_path):
    """Each test gets a fresh SQLite auth DB."""
    db_path = str(tmp_path / "auth.db")
    old = C.AUTH_DB_PATH
    C.AUTH_DB_PATH = db_path
    from auth.db import init_db
    init_db()
    yield db_path
    C.AUTH_DB_PATH = old
    if os.path.exists(db_path):
        os.remove(db_path)



@pytest.fixture(autouse=True)
def _reset_login_rate_limit():
    """每个测试前清空登录/注册 IP 限流状态(autouse)。"""
    try:
        from auth.router import _login_attempts
        _login_attempts.clear()
    except Exception:
        pass
    yield


@pytest.fixture
def isolated_ratelimit(monkeypatch):
    """Reset rate-limit state: fresh semaphore(2) + empty user locks + 0.5s timeout."""
    import threading
    from support import ratelimit

    monkeypatch.setattr(ratelimit, "_global_sem", threading.Semaphore(2))
    monkeypatch.setattr(ratelimit, "_user_locks", {})
    monkeypatch.setattr(C, "RATE_LIMIT_QUEUE_TIMEOUT", 0.5)
    yield ratelimit


def make_llm_response(content: str):
    """Build a mock OpenAI ChatCompletion response object."""
    from unittest.mock import MagicMock
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    return resp
