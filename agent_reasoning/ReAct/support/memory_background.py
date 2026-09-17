# -*- coding: utf-8 -*-
"""主链路与后台记忆管道的接线层。

职责边界:memories 包不反向依赖 agent_reasoning(依赖方向保持单向),故
checkpoint 压缩(RemoveMessage → 会话 checkpoint)这类需要 ReAct 图 schema
的动作在本层实现,以 compact_applier 回调形式注入管道。

- submit_turn_memory():答案定稿后调用,非阻塞提交本轮记忆任务;
- wait_previous_turn():下一轮请求入口调用,等上一轮记忆处理完成(有界);
- reset_react_memory():质检重做前调用,清空 ReAct 工作记忆(不换会话键)。
"""
from __future__ import annotations

import logging
from typing import Optional

import config as C

logger = logging.getLogger("agent")


def _apply_compact(store_thread_id: str, remove_messages: list) -> None:
    """把摘要节点产出的 RemoveMessage 应用到会话 checkpoint。

    在后台 worker(两轮之间)执行:自开 working_saver + 最小压缩图 update_state。
    与运行中图的 checkpoint 写并发时 update_state 可能被覆盖——入口等待门
    (wait_previous_turn)保证正常情况下不会发生;万一发生,压缩幂等,下一轮重做。
    """
    if not remove_messages:
        return
    try:
        from memories.storage.working import working_saver
        from agent_reasoning.ReAct.core.graph import build_compaction_graph
        with working_saver() as cp:
            g = build_compaction_graph(checkpointer=cp)
            g.update_state(
                {"configurable": {"thread_id": store_thread_id}},
                {"messages": list(remove_messages)})
    except Exception as e:  # noqa: BLE001  Redis 不可用等:压缩下轮重做
        logger.info("memory compact apply skipped: %s: %s",
                    type(e).__name__, str(e)[:160])


def submit_turn_memory(*, username: Optional[str], store_thread_id: Optional[str],
                       question: str = "", answer: str = "",
                       messages: Optional[list] = None,
                       final_reason: str = "answer") -> None:
    """非阻塞提交本轮记忆任务(compact_applier 注入 checkpoint 压缩动作)。"""
    from memories.orchestration.memory_loop import submit_turn_memory as _submit
    _submit(username=username, thread_id=store_thread_id,
            question=question, answer=answer, messages=messages,
            final_reason=final_reason,
            compact_applier=_apply_compact if username else None)


def wait_previous_turn(username: Optional[str], thread_id: Optional[str]) -> bool:
    """入口等待门:等该会话上一轮记忆链处理完成(MEM_WAIT_IDLE_TIMEOUT 兜底放行)。

    等待键与提交键一致:都使用 scoped_thread_id(与 runner 提交、checkpoint/流水
    同一命名空间),否则门会永远空放。
    """
    from memories.orchestration.memory_loop import wait_previous_turn as _wait
    from memories.storage.thread_scope import scoped_thread_id
    store_key = scoped_thread_id(thread_id, username)
    return _wait(username, store_key,
                 timeout=float(getattr(C, "MEM_WAIT_IDLE_TIMEOUT", 10.0)))


def reset_react_memory(username: Optional[str], thread_id: Optional[str]) -> None:
    """清空会话 checkpoint 的 messages(质检重做前调用,等价旧版"换新 thread_id")。

    但【不换会话键】:短期流水/事实表/摘要/入口等待门仍归原 thread——重做轮的
    问答对下一轮召回可见(修复"换 thread 导致多轮失忆")。清空后重做轮为冷启动,
    build_messages 以 Redis 短期流水为权威重建完整上下文(含上一版失败答案,
    配合 qc_feedback 显式注入质检反馈)。

    失败向上抛,由调用方兜底(带旧上下文重做,不阻断)。
    """
    from langchain_core.messages import RemoveMessage
    from memories.storage.working import working_saver
    from memories.storage.thread_scope import scoped_thread_id
    from agent_reasoning.ReAct.core.graph import build_compaction_graph
    store_thread_id = scoped_thread_id(thread_id, username)
    with working_saver() as cp:
        g = build_compaction_graph(checkpointer=cp)
        cfg = {"configurable": {"thread_id": store_thread_id}}
        values = g.get_state(cfg).values or {}
        removes = [RemoveMessage(id=m.id)
                   for m in (values.get("messages") or [])
                   if getattr(m, "id", None)]
        if removes:
            g.update_state(cfg, {"messages": removes})
