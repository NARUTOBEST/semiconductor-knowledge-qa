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
    os.path.join(_PROJECT, "RAG", "pdf"),
    os.path.join(_PROJECT, "server"),
    os.path.join(_PROJECT, "context management"),
]:
    # 注:mcp_servers/retrieval 不入 sys.path——它含 tools.py,提前插入会让
    # ``import tools`` 解析到检索服务的工具模块而非 agent 工具包(子模块
    # 自带 sys.path 兜底,裸 import 无需此路径)。
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# MCP 工具桥默认走 memory 传输在进程内直连检索工具声明(不触网、不加载模型):
# import tools 即注册三件套进 registry,与旧本地子包 import 期注册行为等价。
os.environ.setdefault(
    "MCP_SERVERS",
    '[{"name": "retrieval", "module": "mcp_servers.retrieval.tools"}]',
)

# 测试默认跑【全功能模式】:经济模式(ECONOMY_MODE)在无网关时会自动关闭路由/质检/
# 升迁等旁路 LLM,而单测用 mock LLM 验证这些完整路径,故这里显式关闭经济模式。
# 生产环境不受影响(未配置网关时仍自动进入经济模式)。
os.environ.setdefault("ECONOMY_MODE", "0")

# 长期记忆(PG 偏好)默认在单测里关闭:它会直连真实 PostgreSQL 并同步调用检索微服务
# /embed_text(不走 dispatch_fn/LLM 的 mock),属外部依赖。专项测试
# (test_long_*.py)用 monkeypatch 显式打开并打桩;其余图/流式测试不应触网。
os.environ.setdefault("LONG_MEM_ENABLED", "0")

# 运行态(限流/指标/admin任务态)外置 Redis;单测强制内存后端保证确定性
# (不连真实 Redis、计数互不污染)。Redis 路径由 test_state_redis.py 用
# fakeredis 注入专项覆盖。setdefault 允许个别用例自行切回。
os.environ.setdefault("RUNTIME_STATE_BACKEND", "memory")

# 服务间鉴权默认关闭:本地 env/env.env 现已配置 RETRIEVAL_INTERNAL_TOKEN(生产),
# 而多数 HTTP 端点测试依赖"未设 token 直接放行"。专项鉴权用例
# (test_retrieval_guards)自行 monkeypatch 置非空验证 403 路径。
os.environ.setdefault("RETRIEVAL_INTERNAL_TOKEN", "")

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
def isolated_registry():
    """测试用例注册的临时工具在用例结束后清理,保留 import 期注册的检索工具。"""
    from tools import registry
    saved = dict(registry._specs)
    yield registry
    registry._specs = saved


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
