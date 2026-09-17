# -*- coding: utf-8 -*-
"""连接工厂:Redis(工作记忆 checkpoint + 短期记忆流水共用同一实例)。

- 工作记忆走 langgraph-checkpoint-redis 的 RedisSaver,用 RedisSaver.from_conn_string
  直接吃 config.REDIS_URL(见 working.py),不在此建客户端。
- 短期记忆流水(mem:* 键)用 get_redis() 返回的 decode_responses=True 单例。
- Redis 连接信息全部来自 config(REDIS_HOST/PORT/PASSWORD/DB / REDIS_URL),禁止硬编码。
- redis-py 客户端在每条命令失败后会自动重连,无需永久"死亡"标记;调用方(短期流水)
  对写入异常做吞掉处理,记忆是旁路但 Redis 为主存储,故超时给足而非缓存级 1s。
"""
import logging
import os
import sys

# 本文件位于 memories/storage/;项目根在上两级
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "config"))
import config as C  # noqa: E402

logger = logging.getLogger("agent")

_CONNECT_TIMEOUT = float(os.getenv("REDIS_CONNECT_TIMEOUT", "3"))
_SOCKET_TIMEOUT = float(os.getenv("REDIS_SOCKET_TIMEOUT", "5"))

_redis_client = None


def _build_redis(*, decode_responses: bool):
    """构建 Redis 客户端;不验证连通性(交给首次命令/ping)。失败返回 None。"""
    try:
        import redis  # redis-py
    except Exception:
        return None
    try:
        return redis.Redis(
            host=C.REDIS_HOST,
            port=C.REDIS_PORT,
            password=C.REDIS_PASSWORD or None,
            db=C.REDIS_DB,
            decode_responses=decode_responses,
            socket_connect_timeout=_CONNECT_TIMEOUT,
            socket_timeout=_SOCKET_TIMEOUT,
            # 健康检查:借出连接前若距上次活动超过 idle 则先发 PING,避免用到坏连接
            health_check_interval=30,
        )
    except Exception:
        return None


def get_redis():
    """返回 decode_responses=True 的 Redis 单例(短期流水用);未装 redis 包时返回 None。

    不做一次性"永久不可用"标记:redis-py 在命令失败后自动重连,Redis 晚于进程启动
    也能在下一条命令恢复。连通性用 ping_redis() 显式探测。
    """
    global _redis_client
    if _redis_client is None:
        _redis_client = _build_redis(decode_responses=True)
    return _redis_client


def ping_redis() -> bool:
    """显式探测 Redis 连通性。"""
    client = get_redis()
    if client is None:
        return False
    try:
        return bool(client.ping())
    except Exception:
        return False


# ---- 热路径快探(供"每轮都要跑"的旁路读取,如短期近期对话注入)----
# 共享客户端带 3s/5s 超时 + 重试,Redis 半挂(端口在但不应答)时一次命令可能阻塞十余秒。
# 热路径旁路不能这么慢:这里用【短超时 + 无重试】的一次性客户端探活,失败后进入冷却,
# 冷却期内直接判定不可用(不再发探测),把对主流程的影响压到亚秒级且只付一次。
import time as _time  # noqa: E402

_probe_fail_until = 0.0
_PROBE_TIMEOUT = float(os.getenv("REDIS_FAST_PROBE_TIMEOUT", "0.2"))
_PROBE_COOLDOWN = float(os.getenv("REDIS_FAST_PROBE_COOLDOWN", "30"))


def redis_ready_fast(cooldown: float = None) -> bool:
    """热路径用的 Redis 快探:亚秒级超时、无重试、失败冷却。

    与 ping_redis() 的区别:ping_redis 用共享客户端(超时/重试较宽松,适合启动探测);
    本函数用于每次请求都会执行的旁路读取,必须在 Redis 不可用时【快速】返回 False。
    成功不缓存(本地 Redis ping 亚毫秒,代价可忽略);失败冷却默认 _PROBE_COOLDOWN 秒,
    可用 cooldown 参数覆盖(如熔断器半开探测传 0 强制真实连接,绕过冷却缓存)。
    """
    global _probe_fail_until
    _cd = _PROBE_COOLDOWN if cooldown is None else cooldown
    if _time.monotonic() < _probe_fail_until:
        return False
    try:
        import redis  # redis-py
        try:
            from redis.retry import Retry
            from redis.backoff import NoBackoff
            retry = Retry(NoBackoff(), 0)  # 0 次重试 = 仅单次尝试
        except Exception:
            retry = None
        probe = redis.Redis(
            host=C.REDIS_HOST, port=C.REDIS_PORT,
            password=C.REDIS_PASSWORD or None, db=C.REDIS_DB,
            socket_connect_timeout=_PROBE_TIMEOUT,
            socket_timeout=_PROBE_TIMEOUT,
            retry=retry,
        )
        try:
            return bool(probe.ping())
        finally:
            try:
                probe.close()  # 立即释放一次性连接池/socket
            except Exception:
                pass
    except Exception:
        _probe_fail_until = _time.monotonic() + _cd
        return False


def memory_ttl_seconds() -> int:
    """短期记忆流水键的 TTL(秒),取 config.MEMORY_TTL_DAYS;<=0 表示不设过期。"""
    try:
        days = int(getattr(C, "MEMORY_TTL_DAYS", 30))
    except Exception:
        days = 30
    return max(0, days) * 86400
