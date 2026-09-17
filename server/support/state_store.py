# -*- coding: utf-8 -*-
"""运行态外置存储门控:限流/指标/admin任务态 共用的 Redis 熔断器。

背景(审计项6):这些运行态原本都是单进程内存态,只能单 worker 部署。
现统一外置到 Redis(与短期记忆/工作记忆同一实例),使多 worker 共享一致状态、
应用重启状态不丢。本模块是 Redis ↔ 内存回退之间的熔断器,状态机:

  CLOSED(正常)
    每次状态操作走 Redis;某次 Redis 命令失败 → 调用方当场用内存完成本请求
    (数据不丢),并 note_fail() 记一次连续失败。
    连续失败 < 阈值:仅进入【快速重试窗口】RETRY_COOLDOWN 秒,窗口内走内存,
    窗口一过下个请求自动重试 Redis(= CLOSED 内先重试)。
    连续失败 >= TRIP_THRESHOLD → 跳闸 OPEN。
    任一次探测/操作成功 → 连续失败清零,回到纯 CLOSED。

  OPEN(降级)
    OPEN_COOLDOWN 秒内所有状态操作直接走内存,不再触碰 Redis(不拖慢请求)。

  HALF_OPEN(试探)
    OPEN 冷却结束后,下个请求单飞发起一次真实探测(0.2s 超时,绕过模块级
    探测冷却):成功 → CLOSED(自愈);失败 → 回到 OPEN 再冷却。
    探测进行中其余并发请求走内存,不放大故障流量。

重试语义说明:【不做】同一次操作内的盲目重试——限流 SET NX / 指标 INCR 均非
幂等,重放可能造成误 429(锁已建立却判失败)或计数翻倍。重试由"请求间快速
重试窗口"承担:失败请求当场内存兜底,后续请求自动重试 Redis。

内存回退语义 = 外置前的原行为(单进程),因此 Redis 不可用时系统行为与
旧版完全一致,只是退回单 worker 语义。
"""
import os
import threading
import time

# ---- 可调参数(env 覆盖)----
_TRIP_THRESHOLD = int(os.getenv("RUNTIME_STATE_TRIP_THRESHOLD", "3"))
_RETRY_COOLDOWN = float(os.getenv("RUNTIME_STATE_RETRY_COOLDOWN", "2"))
_OPEN_COOLDOWN = float(
    os.getenv("RUNTIME_STATE_OPEN_COOLDOWN",
              os.getenv("RUNTIME_STATE_FAIL_COOLDOWN", "30")))
_STREAK_WINDOW = float(os.getenv("RUNTIME_STATE_STREAK_WINDOW", "60"))
_OK_CACHE_S = 2.0            # 快探成功结果短缓存:避免每次状态操作都新建探测连接

# ---- 熔断状态(线程安全:_lk 保护)----
_lk = threading.Lock()
_open_until = 0.0            # >now = OPEN;冷却结束进入 HALF_OPEN
_probing = False             # HALF_OPEN 单飞探测标志
_retry_until = 0.0           # CLOSED 内快速重试窗口
_fail_streak = 0             # 连续失败计数(成功清零;超 _STREAK_WINDOW 衰减)
_last_fail = 0.0
_ok_until = 0.0              # CLOSED 下快探成功短缓存

# 测试注入点:monkeypatch 本函数返回 fakeredis 客户端即可测 Redis 路径
_client_fn = None


def _backend() -> str:
    return os.getenv("RUNTIME_STATE_BACKEND", "redis").strip().lower()


def _probe_real() -> bool:
    """强制真实连接的快探(绕过探测模块自身冷却),供 HALF_OPEN 用。"""
    try:
        from memories.storage.connections import redis_ready_fast
        return bool(redis_ready_fast(cooldown=0))
    except Exception:
        return False


def redis_mode() -> bool:
    """本次操作是否应走 Redis(熔断器当前状态判定)。"""
    if _backend() != "redis":
        return False
    global _probing, _ok_until, _fail_streak, _open_until
    now = time.monotonic()

    # OPEN:降级中;冷却结束 → HALF_OPEN 单飞探测
    if _open_until:
        if now < _open_until:
            return False
        with _lk:
            if _probing or time.monotonic() < _open_until:
                return False          # 已有探测在飞 / 恰好被并发方复活
            _probing = True
        try:
            ok = _probe_real()
        finally:
            with _lk:
                _probing = False
        if ok:
            with _lk:                 # 自愈 → CLOSED
                _open_until = 0.0
                _fail_streak = 0
                _ok_until = time.monotonic() + _OK_CACHE_S
        else:
            with _lk:                 # 试探失败 → 回到 OPEN 再冷却
                _open_until = time.monotonic() + _OPEN_COOLDOWN
        return ok

    # CLOSED 但处于快速重试窗口:先内存,窗口一过自动重试
    if now < _retry_until:
        return False
    # CLOSED 正常路径:快探(成功结果短缓存)
    if now < _ok_until:
        return True
    ok = _probe_real()
    if ok:
        with _lk:
            _fail_streak = 0
            _ok_until = time.monotonic() + _OK_CACHE_S
    else:
        # 快探失败同样计入熔断:达阈值跳闸,否则进快速重试窗口(避免每请求都探)
        note_fail()
    return ok


def note_fail():
    """Redis 命令失败后调用:记连续失败;达阈值跳闸 OPEN,否则进快速重试窗口。"""
    global _fail_streak, _last_fail, _open_until, _retry_until
    with _lk:
        now = time.monotonic()
        if now - _last_fail > _STREAK_WINDOW:
            _fail_streak = 0          # 失败间隔拉长:视为已恢复,重新计数
        _fail_streak += 1
        _last_fail = now
        _ok_until = 0.0
        if _fail_streak >= _TRIP_THRESHOLD:
            _open_until = now + _OPEN_COOLDOWN      # OPEN:持续降级
        else:
            _retry_until = now + _RETRY_COOLDOWN    # CLOSED 内先快速重试


def get_state_redis():
    """返回共享 Redis 客户端(decode_responses);不可用返回 None(调用方走内存)。"""
    if not redis_mode():
        return None
    if _client_fn is not None:
        return _client_fn()
    try:
        from memories.storage.connections import get_redis
        return get_redis()
    except Exception:
        note_fail()
        return None
