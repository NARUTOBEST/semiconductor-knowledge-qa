# -*- coding: utf-8 -*-
"""会话级联清理与工作记忆滚动 TTL。

属于记忆编排层(memories/orchestration),被会话删除路由与 main 启动钩子调用。
第二种语义(thread_id = conversation.id)下,工作记忆(checkpoint)是会话级状态。
会话删除或长期不活跃时必须清理,否则 checkpoint + 短期流水无限累积:

  - delete_thread_artifacts(thread_id):会话被删除时调用,清 working 库 checkpoint
    三表 + short 库该 thread 流水 + summarize 预生成缓存。
  - prune_inactive(older_than_days):滚动 TTL,清理最后活动早于 N 天的线程。
    checkpoint 表无时间列,以 short 库 max(created_at) 作为线程最后活动时间。

清理失败一律记录日志、不抛异常,避免阻断会话删除等主流程。
"""
from __future__ import annotations

import json
import logging
import threading
import time

from memories.storage import working_saver
from memories.storage.short import short_term
from memories.storage.working import summarize

logger = logging.getLogger("agent")

DEFAULT_TTL_DAYS = 30
_PRUNE_INTERVAL_SECONDS = 24 * 3600  # 每日一次


def delete_thread_artifacts(thread_id: str) -> None:
    """会话删除时级联清理该 thread 的 checkpoint 与短期流水。"""
    # 0) 清掉可能残留的摘要预生成缓存(避免内存泄漏)
    try:
        with summarize._pregen_lock:
            summarize._pregen.pop(thread_id, None)
    except Exception:
        pass

    # 1) 工作记忆 checkpoint(checkpoints / blobs / writes 三表)
    try:
        with working_saver() as cp:
            cp.delete_thread(thread_id)
    except Exception as e:
        logger.warning(json.dumps({
            "event": "cleanup_checkpoint_fail", "thread_id": thread_id,
            "error": f"{type(e).__name__}: {e}",
        }, ensure_ascii=False))

    # 2) 短期会话流水
    try:
        n = short_term.delete_thread(thread_id)
        logger.info(json.dumps({
            "event": "cleanup_thread", "thread_id": thread_id,
            "short_deleted": n,
        }, ensure_ascii=False))
    except Exception as e:
        logger.warning(json.dumps({
            "event": "cleanup_short_fail", "thread_id": thread_id,
            "error": f"{type(e).__name__}: {e}",
        }, ensure_ascii=False))

    # 3) 短期记忆事实表(memf:*)+ 会话摘要文件(session-memory.md)
    try:
        from memories.storage.short.facts import fact_table
        fact_table.delete_thread(thread_id)
    except Exception as e:
        logger.warning(json.dumps({
            "event": "cleanup_facts_fail", "thread_id": thread_id,
            "error": f"{type(e).__name__}: {e}",
        }, ensure_ascii=False))

    # 4) 工作流追踪记录(trace:* 独立键空间;tool_call/tool_result/error/
    #    error_trace/done)。旁路清理,失败只记日志不阻断会话删除。
    try:
        from trace import delete_trace_thread
        n = delete_trace_thread(thread_id)
        logger.info(json.dumps({
            "event": "cleanup_trace_thread", "thread_id": thread_id,
            "trace_deleted": n,
        }, ensure_ascii=False))
    except Exception as e:
        logger.warning(json.dumps({
            "event": "cleanup_trace_fail", "thread_id": thread_id,
            "error": f"{type(e).__name__}: {e}",
        }, ensure_ascii=False))
    try:
        from memories.storage.working.session_file import SessionFile
        uname = thread_id.split("|", 1)[0] if "|" in thread_id else None
        if uname:
            SessionFile(uname, thread_id).delete()
    except Exception as e:
        logger.warning(json.dumps({
            "event": "cleanup_session_file_fail", "thread_id": thread_id,
            "error": f"{type(e).__name__}: {e}",
        }, ensure_ascii=False))


def delete_user_artifacts(username: str) -> dict:
    """账号注销:级联清理该用户在三层记忆库的全部数据。

      - working: 该用户命名空间(``username|%``)下所有 thread 的 checkpoint 三表;
      - short : 该用户命名空间下的 session_events 流水;
      - long  : 该用户的 PG 长期偏好(画像 + 全部分片条目)。

    顺序:先从短期流水枚举该用户的 thread 键(删 checkpoint 需要),再删
    checkpoint / short。各层失败只计数记日志、不阻断其余层清理——
    账号注销应尽可能多地删除,残留由运维据日志补偿。
    返回 {"checkpoints_deleted", "short_events_deleted"}。
    """
    from memories.storage.thread_scope import scoped_thread_id

    result = {"checkpoints_deleted": 0, "short_events_deleted": 0}

    # 0) 先枚举该用户的 thread 键(删 checkpoint 用);须在删 short 之前做
    try:
        thread_keys = short_term.list_user_threads(username)
    except Exception as e:
        logger.warning(json.dumps({
            "event": "account_purge_list_threads_fail", "user": username,
            "error": f"{type(e).__name__}: {e}",
        }, ensure_ascii=False))
        thread_keys = []

    # 1) 工作记忆 checkpoint:按枚举出的 thread 键逐个删。
    #    会话表(auth.db conversations)里的 id 也兜底补一遍,覆盖短期流水
    #    已被 TTL 清掉、但 checkpoint 仍残留的情况。
    conv_ids = []
    try:
        from chat.conversation.db import get_conversations
        conv_ids = [c["id"] for c in get_conversations(username)]
    except Exception:
        pass
    keys_to_delete = set(thread_keys)
    for cid in conv_ids:
        keys_to_delete.add(scoped_thread_id(cid, username))
    try:
        with working_saver() as cp:
            for key in keys_to_delete:
                try:
                    cp.delete_thread(key)
                    result["checkpoints_deleted"] += 1
                except Exception as e:
                    logger.warning(json.dumps({
                        "event": "account_purge_cp_fail", "thread_id": key,
                        "error": f"{type(e).__name__}: {e}",
                    }, ensure_ascii=False))
    except Exception as e:
        logger.warning(json.dumps({
            "event": "account_purge_saver_fail", "user": username,
            "error": f"{type(e).__name__}: {e}",
        }, ensure_ascii=False))

    # 2) 短期流水(最后删:它是上面枚举 thread 的数据源)
    try:
        r = short_term.delete_user(username)
        result["short_events_deleted"] = r.get("events", 0)
    except Exception as e:
        logger.warning(json.dumps({
            "event": "account_purge_short_fail", "user": username,
            "error": f"{type(e).__name__}: {e}",
        }, ensure_ascii=False))

    # 3) 长期偏好(PG):按 user 分片删除画像 + 全部偏好条目
    try:
        from ..long import delete_user_long_term
        result["long_memories_deleted"] = int(delete_user_long_term(username) or 0)
    except Exception as e:
        logger.warning(json.dumps({
            "event": "account_purge_long_fail", "user": username,
            "error": f"{type(e).__name__}: {e}",
        }, ensure_ascii=False))

    # 4) 短期记忆事实表(memf:*)+ 会话摘要文件目录(整目录 rmtree)
    try:
        from memories.storage.short.facts import fact_table
        r = fact_table.delete_user(username)
        result["facts_deleted"] = int(r.get("facts", 0))
    except Exception as e:
        logger.warning(json.dumps({
            "event": "account_purge_facts_fail", "user": username,
            "error": f"{type(e).__name__}: {e}",
        }, ensure_ascii=False))
    try:
        from memories.storage.working.session_file import delete_user_files
        delete_user_files(username)
    except Exception as e:
        logger.warning(json.dumps({
            "event": "account_purge_session_files_fail", "user": username,
            "error": f"{type(e).__name__}: {e}",
        }, ensure_ascii=False))

    # 5) 工作流追踪记录(trace:* 独立键空间):按用户索引枚举并整批删除。
    try:
        from trace import delete_trace_user
        result["trace_events_deleted"] = int(delete_trace_user(username) or 0)
    except Exception as e:
        logger.warning(json.dumps({
            "event": "account_purge_trace_fail", "user": username,
            "error": f"{type(e).__name__}: {e}",
        }, ensure_ascii=False))

    logger.info(json.dumps({
        "event": "account_purge_done", "user": username, **result,
    }, ensure_ascii=False))
    return result


def prune_inactive(older_than_days: int = DEFAULT_TTL_DAYS) -> dict:
    """滚动清理最后活动早于 N 天的线程的 checkpoint + 短期流水。

    返回 {candidates, checkpoints_deleted, short_deleted}。
    """
    try:
        stale = short_term.stale_threads(older_than_days)
    except Exception as e:
        logger.warning(json.dumps({
            "event": "prune_list_fail",
            "error": f"{type(e).__name__}: {e}",
        }, ensure_ascii=False))
        return {"candidates": 0, "checkpoints_deleted": 0, "short_deleted": 0}

    cp_deleted = 0
    short_deleted = 0
    try:
        with working_saver() as cp:
            for thread_id, _last_active in stale:
                try:
                    cp.delete_thread(thread_id)
                    cp_deleted += 1
                except Exception as e:
                    logger.warning(json.dumps({
                        "event": "prune_cp_fail", "thread_id": thread_id,
                        "error": f"{type(e).__name__}: {e}",
                    }, ensure_ascii=False))
    except Exception as e:
        logger.warning(json.dumps({
            "event": "prune_saver_fail",
            "error": f"{type(e).__name__}: {e}",
        }, ensure_ascii=False))

    try:
        short_deleted = short_term.delete_threads_before(older_than_days)
    except Exception as e:
        logger.warning(json.dumps({
            "event": "prune_short_fail",
            "error": f"{type(e).__name__}: {e}",
        }, ensure_ascii=False))

    logger.info(json.dumps({
        "event": "prune_done", "older_than_days": older_than_days,
        "candidates": len(stale), "checkpoints_deleted": cp_deleted,
        "short_deleted": short_deleted,
    }, ensure_ascii=False))
    return {
        "candidates": len(stale),
        "checkpoints_deleted": cp_deleted,
        "short_deleted": short_deleted,
    }


_LEADER_KEY = "mem:prune:leader"   # 值班锁:多 worker 下仅一个进程执行清理


def _i_lead_prune(ttl_ms: int, token: str) -> bool:
    """抢主/续期值班锁(SET NX PX,持有者用 GET==token 后 PEXPIRE 续期)。

    Redis 不可用 → 返回 False(此时本就无 Redis checkpoint 可清,跳过是正确语义)。
    GET 与 PEXPIRE 非原子:极端竞态下可能给他人新锁短暂续期 → 两个进程各清理一次,
    prune_inactive 幂等,无害(故不再引入 WATCH 事务加重复杂度)。
    """
    try:
        from ...storage.connections import get_redis
        r = get_redis()
        if r is None:
            return False
        if r.set(_LEADER_KEY, token, nx=True, px=ttl_ms):
            return True
        if r.get(_LEADER_KEY) == token:
            r.pexpire(_LEADER_KEY, ttl_ms)
            return True
        return False
    except Exception:
        return False


def prune_loop(older_than_days: int = DEFAULT_TTL_DAYS,
               interval_seconds: int = _PRUNE_INTERVAL_SECONDS,
               tick_seconds: int = 30) -> None:
    """守护线程循环:每隔 interval 跑一次 prune_inactive(多 worker 值班选主)。

    按 tick(默认 30s)小步轮询值班锁:锁 TTL = 2×tick,持有者每 tick 续期,
    值班进程崩溃后锁最迟 2×tick 过期、其他 worker 自动接管。到 interval 才真正
    执行清理,单次异常被捕获,不影响下一轮。
    """
    import uuid
    token = uuid.uuid4().hex
    elapsed = 0
    while True:
        time.sleep(tick_seconds)
        if not _i_lead_prune(tick_seconds * 2000, token):
            elapsed = 0          # 失去/未获得值班权:重新计满 interval 再清
            continue
        elapsed += tick_seconds
        if elapsed < interval_seconds:
            continue
        elapsed = 0
        try:
            prune_inactive(older_than_days)
        except Exception:
            logger.exception("prune_loop iteration failed")


def start_prune_daemon(older_than_days: int = DEFAULT_TTL_DAYS) -> threading.Thread:
    """启动滚动 TTL 守护线程(daemon,随进程退出)。"""
    t = threading.Thread(
        target=prune_loop, args=(older_than_days,),
        daemon=True, name="memory-prune",
    )
    t.start()
    return t
