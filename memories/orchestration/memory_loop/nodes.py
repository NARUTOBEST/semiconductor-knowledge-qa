# -*- coding: utf-8 -*-
"""独立记忆图的薄图节点 + 条件路由。

记忆编排已从 ReAct 主图整体迁出(主图 finalize → emit_done → END),由
后台管道(pipeline.py)在答案定稿后以 daemon 线程 invoke 本图。节点只做
"取 state -> 调纯逻辑 -> 回写 state 路由字段",不发 SSE 心跳(不在请求流内),
最外层不抛(逻辑内部已兜底)。

路由靠 state 字段:
  mem_consolidate_done=True 且 mem_degraded=False -> summary
  mem_consolidate_done=False(仍可重试)          -> consolidate_retry
  mem_degraded=True(熔断/耗尽,已裸写)          -> consolidate_degrade

后台执行不占用户等待时间,故不再做整链预算/deadline(旧 MEM_CHAIN_BUDGET 机制
随同步链一并退役);重试退避/熔断/降级逻辑全部保留在纯逻辑层(consolidate.py)。
"""
from __future__ import annotations

import logging
from typing import Any

from . import consolidate as CSL
from . import session_summary as SS
from . import resilience as RS

logger = logging.getLogger("agent")


def _patch(r: dict[str, Any]) -> dict[str, Any]:
    return {
        "mem_llm_fail_count": int(r.get("fail_count") or 0),
        "mem_breaker_open": bool(r.get("breaker_open")),
        "mem_degraded": bool(r.get("degraded")),
        "mem_fact_id": r.get("fid") or "",
        "mem_consolidate_done": r.get("route") != CSL.ROUTE_RETRY,
    }


def _user(state) -> tuple[str, str]:
    return state.get("username") or "", state.get("thread_id") or ""


# ---- 节点一:记忆沉淀(含降级兜底)----
def consolidate_node(state) -> dict[str, Any]:
    username, thread_id = _user(state)
    # 兜底先行(原独立 mem_resilience 节点收拢于此):WAL 回填 + PG spool 重放 +
    # NULL 向量补嵌 + 记忆欠账补做,有界、best-effort。不挑门控——匿名/关闭记忆
    # 轮也要推进故障欠账;Redis/PG 刚恢复时先回填旧账,再做本轮沉淀(直接写恢复后的后端)。
    try:
        RS.run_resilience_maintenance()
    except Exception as e:  # noqa: BLE001  绝不冒泡
        logger.info("memory-graph resilience drain failed: %s: %s",
                    type(e).__name__, str(e)[:120])
    try:
        r = CSL.run_consolidation(
            username, thread_id,
            state.get("question") or "", state.get("full_reply") or "",
            state.get("final_reason") or "answer",
            fail_count=int(state.get("mem_llm_fail_count") or 0))
    except Exception as e:  # noqa: BLE001  绝不冒泡
        logger.info("memory-graph consolidate node failed: %s: %s",
                    type(e).__name__, str(e)[:120])
        return {"mem_consolidate_done": True, "mem_degraded": True}
    return _patch(r)


def consolidate_retry_node(state) -> dict[str, Any]:
    username, thread_id = _user(state)
    try:
        r = CSL.run_retry(
            username, thread_id,
            state.get("question") or "", state.get("full_reply") or "",
            state.get("final_reason") or "answer",
            fail_count=int(state.get("mem_llm_fail_count") or 0))
    except Exception as e:  # noqa: BLE001
        logger.info("memory-graph retry node failed: %s: %s",
                    type(e).__name__, str(e)[:120])
        return {"mem_consolidate_done": True, "mem_degraded": True}
    return _patch(r)


def consolidate_degrade_node(state) -> dict[str, Any]:
    """降级汇合节点:裸写已在逻辑层(consolidate._degrade_write)完成,这里仅标记。"""
    return {"mem_consolidate_done": True, "mem_degraded": True}


# ---- 节点二:会话摘要与 Auto-Compact(阈值不满足时内部零成本早退)----
def summary_node(state) -> dict[str, Any]:
    username, thread_id = _user(state)
    try:
        r = SS.run_session_maintenance(
            username, thread_id, state.get("messages") or [],
            final_reason=state.get("final_reason") or "answer")
    except Exception as e:  # noqa: BLE001  绝不冒泡
        logger.info("memory-graph summary node failed: %s: %s",
                    type(e).__name__, str(e)[:120])
        return {}
    out: dict[str, Any] = {
        "summary_level": r.get("level") or "",
        "summarized": bool(r.get("summarized")),
        "compacted": bool(r.get("compacted")),
    }
    if r.get("remove"):
        # compact 不再经图 state 生效:RemoveMessage 由管道交给调用方注入的
        # compact_applier 直接落到会话 checkpoint(见 pipeline._execute)。
        out["remove_messages"] = r["remove"]
    return out


# ---- 条件路由 ----
def route_consolidate(state) -> str:
    """节点一(含重试)后的三向路由。"""
    if state.get("mem_degraded"):
        return "degrade"
    if not state.get("mem_consolidate_done"):
        return "retry"
    return "summary"
