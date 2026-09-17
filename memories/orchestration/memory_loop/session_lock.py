# -*- coding: utf-8 -*-
"""跨进程会话锁:多 worker 部署下,同一会话的记忆任务全局互斥。

单 worker 时代管道"进程内 FIFO 即锁"只保证本进程内串行;WORKERS>1 时同一
会话的请求可能落在不同 worker,两条管道会并发改同一会话的游标/指纹/摘要文件。
本模块用 Redis SET NX PX 实现按会话的全局互斥(键 ``memlock:{username}|{tid}``):
  - 锁值 = 持有者 token,释放前 WATCH 比对删除(防误删他人新锁,与
    server/support/ratelimit 的槽位同套路;不 import server 层,保持依赖单向);
  - PX TTL 兜底:持有者崩溃未释放时到期自动回收,不死锁;
  - Redis 不可用 fail-open(不加锁直接执行):退化为单 worker 时代的进程内
    FIFO 串行语义,与外置前行为一致。
"""
import logging
import uuid

import config as C

from ...storage import connections

logger = logging.getLogger("agent")

# 锁 TTL(秒):须大于单个记忆任务最长耗时(含升迁门退避重试);持有者崩溃后
# 该会话的记忆最长被阻塞 TTL 秒,期间任务在管道内重试等待,超时前不并发执行。
def _lock_ttl_ms() -> int:
    return int(max(30.0, float(getattr(C, "MEM_LOOP_LOCK_TTL", 300.0))) * 1000)

# 热路径快探(与短期流水同策略):Redis 不可用时亚秒级判退,不拖慢管道/等待门。
# 经 connections 模块属性引用,便于测试 monkeypatch 注入。
def _ready_fast() -> bool:
    return connections.redis_ready_fast()


def _lock_key(key: tuple[str, str]) -> str:
    return f"memlock:{key[0] or ''}|{key[1] or ''}"


def acquire(key: tuple[str, str]) -> str | None:
    """尝试获取会话锁。成功返回 token(凭此释放);被占/Redis 不可用返回 None。

    返回 None 不区分"被他人持有"与"Redis 不可用":调用方对前者重试等待,
    对后者(冷卻期内快探失败)直接无锁执行——两者都指向"本轮先不执行"或
    "退化为无锁",由调用方统一按 None 处理,语义安全(fail-open)。
    """
    if not _ready_fast():
        return None
    r = connections.get_redis()
    if r is None:
        return None
    token = uuid.uuid4().hex
    try:
        if not r.set(_lock_key(key), token, nx=True, px=_lock_ttl_ms()):
            return None
    except Exception:  # noqa: BLE001  Redis 当场失败 → 无锁执行(退化)
        return None
    return token


def release(key: tuple[str, str], token: str | None) -> None:
    """释放会话锁:GET==token 才 DEL(WATCH 事务,防误删他人新锁)。

    失败静默(TTL 兜底回收);token 为 None 表示从未持有,直接忽略。
    """
    if not token:
        return
    r = connections.get_redis()
    if r is None:
        return
    try:
        from redis.exceptions import WatchError
        k = _lock_key(key)
        with r.pipeline() as p:
            while True:
                try:
                    p.watch(k)
                    if p.get(k) != token:
                        p.unwatch()
                        return
                    p.multi()
                    p.delete(k)
                    p.execute()
                    return
                except WatchError:
                    continue
    except Exception:  # noqa: BLE001
        logger.info("memory session lock release failed key=%s", _lock_key(key))


def held(key: tuple[str, str]) -> bool:
    """该会话锁当前是否被持有(供入口等待门跨进程感知)。

    Redis 不可用(快探失败)返回 False——与 fail-open 一致,等待门退化为
    只看进程内队列。
    """
    if not _ready_fast():
        return False
    r = connections.get_redis()
    if r is None:
        return False
    try:
        return bool(r.exists(_lock_key(key)))
    except Exception:  # noqa: BLE001
        return False
