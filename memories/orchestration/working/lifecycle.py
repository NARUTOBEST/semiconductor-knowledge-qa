# -*- coding: utf-8 -*-
"""会话级联清理与工作记忆滚动 TTL。

属于记忆编排层(memories/orchestration),被会话删除路由与 main 启动钩子调用。
第二种语义(thread_id = conversation.id)下,工作记忆(checkpoint)是会话级状态。
会话删除或长期不活跃时必须清理,否则 checkpoint + 短期流水无限累积:

  - delete_thread_artifacts(thread_id):会话被删除时调用,清 working 库 checkpoint
    三表 + short 库该 thread 流水 + summarize 预生成缓存。**不动 long 库**
    (长期记忆是跨会话萃取的事实,不按单会话删除)。
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


def prune_inactive(older_than_days: int = DEFAULT_TTL_DAYS) -> dict:
    """滚动清理最后活动早于 N 天的线程的 checkpoint + 短期流水。

    返回 {candidates, checkpoints_deleted, short_deleted}。long 库不动。
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


def prune_loop(older_than_days: int = DEFAULT_TTL_DAYS,
               interval_seconds: int = _PRUNE_INTERVAL_SECONDS) -> None:
    """守护线程循环:每隔 interval 跑一次 prune_inactive。

    启动时先等一个 interval 再首次执行(避免与启动预热抢资源),之后按天清理。
    单次异常被捕获,不影响下一轮。
    """
    while True:
        time.sleep(interval_seconds)
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
