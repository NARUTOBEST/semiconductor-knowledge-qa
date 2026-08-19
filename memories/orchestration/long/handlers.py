# -*- coding: utf-8 -*-
"""长期记忆升迁的触发入口。

属于记忆编排层(memories/orchestration),被 react/runner 在流结束/异常后调用:
后台 daemon 线程把该 thread 的短期流水萃取成事实写入长期库。
- 用 daemon 线程,不阻塞 SSE 响应收尾,也不依赖 asyncio 事件循环(同步 runner)。
- 升迁失败只打印,不影响主流程。
"""
from __future__ import annotations

import logging
import threading
from typing import Optional

from memories.storage.long import promote_thread
from memories.storage.short import short_term

logger = logging.getLogger("agent")

# 启动补偿扫描最多补处理的 thread 数,避免重启后一次性起太多升迁线程
_RECOVERY_BATCH = 50


def after_stream(thread_id: str,
                 user_id: Optional[str],
                 session_id: Optional[str] = None) -> None:
    """在后台线程触发一次升迁。user_id 为空则跳过(长期记忆按用户隔离)。"""
    if not user_id:
        return

    def _run():
        try:
            promote_thread(thread_id, user_id, session_id=session_id)
        except Exception as e:
            logger.warning("after_stream promote_thread failed (ignored): %s: %s",
                           type(e).__name__, e)

    threading.Thread(
        target=_run, name=f"promote-{thread_id}", daemon=True,
    ).start()


def recover_pending_promotions(limit: int = _RECOVERY_BATCH) -> int:
    """启动补偿:把有未升迁事件但进程退出期间没处理的 thread 补跑一次升迁。

    daemon 升迁线程随进程退出会丢失未完成批次,水位不推进;若该 thread 之后
    再也没有新对话,就永远不会升迁。启动时扫一遍补齐。每个待补 thread 各起
    一个 daemon 线程(与正常 after_stream 同路径),失败由 promote_thread 自身
    的熔断/重试逻辑兜底。返回补跑的 thread 数。
    """
    try:
        pending = short_term.pending_promotion_threads(limit=limit)
    except Exception as e:
        logger.warning("promotion recovery list failed: %s: %s",
                       type(e).__name__, e)
        return 0

    for thread_id, user_id in pending:
        after_stream(thread_id, user_id)
    if pending:
        logger.info("promotion recovery scheduled %d thread(s)", len(pending))
    return len(pending)


def start_recovery_daemon(limit: int = _RECOVERY_BATCH) -> None:
    """在后台线程执行一次启动补偿扫描,不阻塞应用启动。"""
    threading.Thread(
        target=recover_pending_promotions, args=(limit,),
        daemon=True, name="promote-recover",
    ).start()
