# -*- coding: utf-8 -*-
"""memory-loop(记忆工作闭环)——独立于主链路的后台记忆编排。

记忆编排已从 ReAct 主图迁出:主图 finalize → emit_done → END(done 立即发、
流即完即关);每轮答案定稿后由主链路向本模块的管道提交记忆任务,daemon worker
串行执行独立记忆图(memories/orchestration/memory_loop/graph.py):
  ① 存储节点 consolidate : 短期事实表去重写入 + 长期升迁门(retry/熔断/裸写降级;
                           PG 宕机走本地 spool,恢复后重放);
  ② 摘要压缩节点 summary : 阈值触发(不达标零成本早退)——session-memory.md
                           9 章节摘要 + Auto-Compact(RemoveMessage 经
                           compact_applier 落会话 checkpoint);
  ③ 兜底维护 resilience  : WAL 回填 + NULL 向量 backfill + PG spool 重放
                           + 记忆欠账补做(work_spool:事实表欠账/升迁门欠账)。

- 同一会话的任务由管道 FIFO 天然串行(游标/compact 无并发写竞态);
- 下一轮请求入口经 wait_idle 等待上一轮处理完成(有界超时,超时放行);
- 全程不占请求流:done 立即发、限流槽随 done 释放、断连杀不到记忆线程。

对外入口(供 agent_reasoning 接线层使用):
  submit_turn_memory() / wait_previous_turn()
"""
from __future__ import annotations

import logging
from typing import Optional

from . import consolidate as CSL
from . import session_summary as SS
from .pipeline import MemoryJob, MemoryPipeline, get_pipeline, reset_pipeline

logger = logging.getLogger("agent")


def submit_turn_memory(*, username: Optional[str], thread_id: Optional[str],
                       question: str = "", answer: str = "",
                       messages: Optional[list] = None,
                       final_reason: str = "answer",
                       compact_applier=None) -> None:
    """把一轮的记忆维护任务非阻塞提交给后台管道;异常不外抛。"""
    try:
        get_pipeline().submit(MemoryJob(
            username=username, thread_id=thread_id,
            question=question, answer=answer,
            messages=messages, final_reason=final_reason,
            compact_applier=compact_applier))
    except Exception as e:  # noqa: BLE001  记忆旁路,绝不影响应答
        logger.info("memory submit skipped: %s: %s",
                    type(e).__name__, str(e)[:120])


def wait_previous_turn(username: Optional[str], thread_id: Optional[str],
                       timeout: Optional[float] = None) -> bool:
    """入口等待门:等该会话上一轮记忆链处理完成(有界超时,超时放行兜底)。

    匿名(无记忆工作)直接放行。
    """
    if not username or not thread_id:
        return True
    try:
        return get_pipeline().wait_idle(username, thread_id, timeout=timeout)
    except Exception as e:  # noqa: BLE001
        logger.info("memory wait_idle skipped: %s: %s",
                    type(e).__name__, str(e)[:120])
        return True


__all__ = [
    "submit_turn_memory",
    "wait_previous_turn",
    "get_pipeline",
    "reset_pipeline",
    "MemoryJob",
    "MemoryPipeline",
    "CSL",
    "SS",
]
