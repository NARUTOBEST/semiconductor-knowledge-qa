# -*- coding: utf-8 -*-
"""memory-loop 节点一【记忆沉淀】。

流程(全部旁路,绝不阻塞/冒泡到应答):
  1. 门控:匿名 / MEM_LOOP_ENABLED=0 / final_reason != "answer" / 空回复 -> skip。
  2. 事实表先按规范化指纹查重:重复问答只刷新 freq/last_access(不调升迁门 LLM)。
  3. 未命中且升迁开启(MEM_PROMOTE_ENABLED,经济模式关)-> 升迁门
     extract.consolidate_turn(LLM 抽取长期偏好,importance≥阈值才写 PG):
       - 成功:事实落短期表,升迁条目标记 promoted;
       - LLM 失败:fail_count+1,未达 MEM_RETRY_MAX -> 路由 retry(指数退避后重跑);
         达阈值 / 熔断开 -> degrade:无条件裸写 Q+A 到短期表(跳过去重与升迁)。
  4. breaker:模块级 monotonic 冷却(连续失败达 MEM_BREAKER_FAIL_THRESHOLD 打开,
     冷却 MEM_BREAKER_COOLDOWN 秒),冷却期直接 degrade,不调 LLM。

事实表权威存 Redis(本模块不持有状态);retry 计数经 state.mem_llm_fail_count
在节点间传递;breaker 为模块级(不进 state)。

欠账暂存(storage/work_spool,重放=本模块 replay_turn):
  - Redis 不可达:事实表写不了 -> Q+A 记入 spool,恢复后补跑完整沉淀;
  - 升迁门 LLM 失败降级:裸写只保证内容短期可见,判定欠账记入 spool,
    LLM 恢复后重放补判(命中已裸写事实则经指纹去重原地升级 promoted)。
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Optional

import config as C

from ...storage.connections import redis_ready_fast
from ...storage.short.facts import fact_table, fingerprint
from ..long import extract

logger = logging.getLogger("agent")

# 路由结果
ROUTE_SKIP = "skip"        # 门控不通过,不写任何记忆
ROUTE_OK = "ok"            # 沉淀完成(含重复/无升迁),进节点二
ROUTE_RETRY = "retry"      # 升迁门 LLM 失败,可重试
ROUTE_DEGRADE = "degrade"  # 熔断/耗尽重试,裸写降级

# ---- 模块级熔断(monotonic 冷却,仿 storage/long/embed.py)----
# 单进程假设(有意接受):计数不外置 Redis——升迁门熔断是韧性逻辑,不应依赖
# 可能正在故障的 Redis。多 worker 下各 worker 独立计数,仅保守性变差(升迁门
# LLM 要挂更久才全集群熔断),不产生错误行为;降级路径(裸写+spool 欠账)本身
# 是安全的。
_breaker_lock = threading.Lock()
_breaker_fail = 0
_breaker_open_until = 0.0

# 重试退避 sleep(Req7:可注入——测试替换为 no-op,不真实等待;生产用 time.sleep)。
_sleep = time.sleep


def _now() -> float:
    return time.monotonic()


def breaker_is_open() -> bool:
    return _now() < _breaker_open_until


def _record_fail() -> None:
    """升迁门失败计数;连续达阈值打开熔断。"""
    global _breaker_fail, _breaker_open_until
    with _breaker_lock:
        _breaker_fail += 1
        if _breaker_fail >= int(getattr(C, "MEM_BREAKER_FAIL_THRESHOLD", 3)):
            _breaker_open_until = _now() + float(getattr(C, "MEM_BREAKER_COOLDOWN", 300))
            logger.warning("memory-loop promotion breaker OPEN for %.0fs",
                           getattr(C, "MEM_BREAKER_COOLDOWN", 300))
            _breaker_fail = 0


def _record_success() -> None:
    global _breaker_fail
    with _breaker_lock:
        _breaker_fail = 0


def reset_breaker() -> None:
    """测试用:复位熔断。"""
    global _breaker_fail, _breaker_open_until
    with _breaker_lock:
        _breaker_fail = 0
        _breaker_open_until = 0.0


# ---------------------------------------------------------------- 门控
def _gated(username: Optional[str], final_reason: Optional[str],
           question: str, answer: str) -> bool:
    if not getattr(C, "MEM_LOOP_ENABLED", True):
        return False
    if not username:  # 匿名不写记忆
        return False
    if (final_reason or "answer") != "answer":
        return False
    if not (question or "").strip() or not (answer or "").strip():
        return False
    return True


# ---------------------------------------------------------------- 主逻辑
def run_consolidation(username: Optional[str], thread_id: Optional[str],
                      question: str, answer: str, final_reason: Optional[str],
                      *, fail_count: int = 0,
                      deadline: Optional[float] = None) -> dict[str, Any]:
    """节点一同步逻辑(供图节点与 simple 函数式入口共用)。

    deadline:整链时间预算(monotonic 时间戳,由调用方/节点创建)。升迁门 LLM
    调用前检查,预算耗尽直接降级裸写(兜底不丢);None 表示不限时(旧调用方兼容)。
    返回 {"route": ROUTE_*, "fid": str|None, "fail_count": int,
          "breaker_open": bool, "degraded": bool, "slept": float}。
    最外层不抛:任何存储/LLM 异常都落到 degrade(裸写)或 ok(重复)。
    """
    if not _gated(username, final_reason, question, answer):
        return {"route": ROUTE_SKIP, "fid": None, "fail_count": 0,
                "breaker_open": breaker_is_open(), "degraded": False, "slept": 0.0}

    # Redis 快探:不可用则事实表/升迁都做不了 —— 本轮 Q+A 记入欠账 spool,Redis
    # 恢复后由兜底维护补跑完整沉淀(去重/升迁/落库);绝不因连接超时阻塞应答。
    # 返回 OK 以继续节点二(文件)。
    if not redis_ready_fast():
        logger.info("memory-loop consolidate skipped: redis not ready")
        try:
            from ...storage import work_spool
            work_spool.append_record(username, thread_id, question, answer)
        except Exception as e:  # noqa: BLE001  本地盘也不可写:只告警
            logger.info("memory-loop work spool append failed: %s: %s",
                        type(e).__name__, str(e)[:120])
        return {"route": ROUTE_OK, "fid": None, "fail_count": 0,
                "breaker_open": breaker_is_open(), "degraded": False, "slept": 0.0}

    fp = fingerprint(question, answer)

    # 1) 先查重:重复问答只刷新,不调升迁门
    try:
        existing = fact_table.touch_if_exists(thread_id, fp)
    except Exception as e:  # noqa: BLE001  Redis 不可用等
        existing = None
        logger.info("memory-loop fact touch failed: %s: %s",
                    type(e).__name__, str(e)[:120])
    if existing:
        _record_success()
        return {"route": ROUTE_OK, "fid": existing, "fail_count": 0,
                "breaker_open": False, "degraded": False, "slept": 0.0}

    # 2) 熔断快检 -> 直接降级裸写
    if breaker_is_open():
        return _degrade_write(username, thread_id, question, answer,
                              fail_count, breaker_open=True)

    # 3) 升迁门(未开启则只落事实、不调 LLM)
    if not getattr(C, "MEM_PROMOTE_ENABLED", True):
        fid = _safe_add(thread_id, question, answer, degraded=False)
        return {"route": ROUTE_OK, "fid": fid, "fail_count": 0,
                "breaker_open": False, "degraded": False, "slept": 0.0}

    # 整链预算耗尽:不再发起 LLM 调用,直接降级裸写(事实不丢,长期下轮再判)
    if deadline is not None and time.monotonic() >= deadline:
        logger.info("memory-loop promotion skipped: chain budget exhausted")
        return _degrade_write(username, thread_id, question, answer,
                              fail_count, breaker_open=breaker_is_open())

    try:
        res = extract.consolidate_turn(
            username, thread_id,
            user_message=question, assistant_message=answer,
            deadline=deadline)
    except Exception as e:  # noqa: BLE001  升迁门 LLM 失败(consolidate_turn 抛 RuntimeError)
        logger.info("memory-loop promotion gate failed: %s: %s",
                    type(e).__name__, str(e)[:120])
        _record_fail()
        nf = fail_count + 1
        # MEM_RETRY_MAX=2 -> 失败后最多再试 2 次(nf=1、2 走 retry,nf=3 degrade)
        if nf <= int(getattr(C, "MEM_RETRY_MAX", 2)) and not breaker_is_open():
            return {"route": ROUTE_RETRY, "fid": None, "fail_count": nf,
                    "breaker_open": False, "degraded": False, "slept": 0.0}
        return _degrade_write(username, thread_id, question, answer,
                              nf, breaker_open=breaker_is_open())

    # 4) 升迁门成功:落事实(未命中),有升迁则标记 promoted
    _record_success()
    fid = _safe_add(thread_id, question, answer, degraded=False)
    promoted = int(res.get("promoted") or 0) if isinstance(res, dict) else 0
    if fid and promoted:
        try:
            fact_table.mark_promoted(thread_id, fid)
        except Exception as e:  # noqa: BLE001
            logger.info("memory-loop mark_promoted failed: %s: %s",
                        type(e).__name__, str(e)[:120])
    return {"route": ROUTE_OK, "fid": fid, "fail_count": 0,
            "breaker_open": False, "degraded": False, "slept": 0.0}


def run_retry(username, thread_id, question, answer, final_reason,
              *, fail_count: int, deadline: Optional[float] = None) -> dict[str, Any]:
    """重试节点:指数退避 sleep 后重跑升迁门。

    sleep = min(2**fail_count, MEM_RETRY_MAX_SLEEP);异常不外抛。重跑仍走
    run_consolidation(含查重/熔断):失败且未达阈值继续返回 retry(由图条件边
    回到本节点,下一轮 fail_count 更大),达阈值/熔断则在 run_consolidation 内
    直接 degrade 裸写。fail_count 单调递增保证图循环必然终止。
    deadline:整链预算——已耗尽则不退避不重试,直接裸写降级;剩余不足时把
    sleep 压到剩余预算内,避免退避本身吃穿预算。
    """
    delay = min(2.0 ** max(1, fail_count),
                float(getattr(C, "MEM_RETRY_MAX_SLEEP", 2.0)))
    rem = None if deadline is None else deadline - time.monotonic()
    if rem is not None and rem <= 0:
        logger.info("memory-loop retry skipped: chain budget exhausted")
        return _degrade_write(username, thread_id, question, answer,
                              fail_count, breaker_open=breaker_is_open())
    if rem is not None:
        delay = min(delay, max(0.0, rem))
    try:
        _sleep(delay)  # 可注入(测试置 no-op);生产为 time.sleep
    except Exception:  # noqa: BLE001
        delay = 0.0
    out = run_consolidation(username, thread_id, question, answer, final_reason,
                            fail_count=fail_count, deadline=deadline)
    out["slept"] = delay
    return out


def _degrade_write(username, thread_id, question, answer, fail_count,
                   *, breaker_open: bool) -> dict[str, Any]:
    """降级:无条件裸写 Q+A 到短期事实表(跳过去重/升迁),同时把"升迁判定欠账"
    记入 work_spool —— LLM 恢复后由兜底维护重放补判(命中已裸写事实则经指纹
    去重原地升级 promoted,不重复插入)。"""
    fid = _safe_add(thread_id, question, answer, degraded=True)
    try:
        from ...storage import work_spool
        work_spool.append_record(username, thread_id, question, answer)
    except Exception as e:  # noqa: BLE001  本地盘也不可写:只告警
        logger.info("memory-loop work spool append failed: %s: %s",
                    type(e).__name__, str(e)[:120])
    return {"route": ROUTE_DEGRADE, "fid": fid, "fail_count": fail_count,
            "breaker_open": breaker_open, "degraded": True, "slept": 0.0}


def replay_turn(rec: dict) -> bool:
    """补做一条欠账(work_spool 重放回调):升迁门 + 事实落库。

    与 run_consolidation 的差异:不先做指纹去重短路 —— 已裸写的 degraded 事实
    也必须补跑升迁门,成功后经 add_or_touch 的指纹去重 touch 原有事实并升级
    promoted(不产生重复条目)。返回 True=已完成(含门控不再通过:丢弃);
    False=本轮失败(LLM 挂/熔断中/事实没写成),保留到下轮。
    调用方(resilience)已保证 Redis 快探通过;异常不外抛。
    """
    username = rec.get("username") or ""
    thread_id = rec.get("thread_id") or ""
    question = rec.get("q") or ""
    answer = rec.get("a") or ""
    if not _gated(username, "answer", question, answer):
        return True  # 记忆已关/内容已空:视为处理完,丢弃
    if breaker_is_open():
        return False  # 熔断冷却期:保留,下轮再试
    if not getattr(C, "MEM_PROMOTE_ENABLED", True):
        _safe_add(thread_id, question, answer, degraded=False)
        return True
    try:
        res = extract.consolidate_turn(
            username, thread_id, user_message=question, assistant_message=answer)
    except Exception as e:  # noqa: BLE001  升迁门 LLM 仍挂:计熔断、保留重试
        logger.info("memory-loop replay gate failed: %s: %s",
                    type(e).__name__, str(e)[:120])
        _record_fail()
        return False
    _record_success()
    fid = _safe_add(thread_id, question, answer, degraded=False)
    if fid is None:
        return False  # 事实没写成(Redis 闪断):保留下轮补写(PG 侧语义去重防重复升迁)
    promoted = int(res.get("promoted") or 0) if isinstance(res, dict) else 0
    if promoted:
        try:
            fact_table.mark_promoted(thread_id, fid)
        except Exception as e:  # noqa: BLE001
            logger.info("memory-loop replay mark_promoted failed: %s: %s",
                        type(e).__name__, str(e)[:120])
    return True


def _safe_add(thread_id, question, answer, *, degraded: bool) -> Optional[str]:
    """写事实表,异常不外抛(返回 None)。"""
    try:
        res = fact_table.add_or_touch(
            thread_id, q=question, a=answer, degraded=degraded)
        return res.get("fid")
    except Exception as e:  # noqa: BLE001  Redis 不可用:记忆旁路,静默
        logger.info("memory-loop fact write skipped: %s: %s",
                    type(e).__name__, str(e)[:120])
        return None
