# -*- coding: utf-8 -*-
"""知识助手 -- 聊天服务入口(薄封装)。

本文件只做一件事:提供 react_stream(),把路由层请求委托给新版 LangGraph ReAct
实现(agent_reasoning.ReAct.support.runner.run_agent_graph)。具体的 ReAct 主循环、LLM 调用、grounding
检测等全部在 chat/react/ 包内,此处不再保留实现。

阶段 3/4:react_stream 先做复杂度路由(classify_complexity),按 tier 分发到
simple / medium(ReAct)/ complex(暂走 ReAct,阶段 5 接 P&E),并在答案产出后、
``done`` 事件前过共享质检门(quality_gate),按 verdict 放行 / 重做一次 / 升级
tier(最多升级一次)。

为兼容现有调用方(health/service.py、context management/query_rewrite.py 及
若干测试以 chat.service.get_client 等路径导入/打桩),这里重导出相关符号。
"""
from __future__ import annotations

import uuid
from typing import Callable, Optional

# ---- 新版 ReAct 实现(已独立到顶层包 agent_reasoning.ReAct)----
from agent_reasoning.ReAct.support.runner import run_agent_graph, run_simple
from agent_reasoning.ReAct.support.llm import get_client, llm_create_with_retry, LLM_RETRIES
from agent_reasoning.ReAct.support.answer_grounding import (
    verify_citations,
    check_faithfulness,
    grounding_check,
    yield_grounding_warnings,
)
from agent_reasoning.router import classify_complexity
from agent_reasoning.quality_gate import check as quality_check

__all__ = [
    "react_stream",
    "get_client",
    "llm_create_with_retry",
    "LLM_RETRIES",
    "verify_citations",
    "check_faithfulness",
    "grounding_check",
    "yield_grounding_warnings",
]

# 升级链:每 tier 至多向上一级;complex 无可升级目标
_NEXT_TIER = {"simple": "medium", "medium": "complex", "complex": None}
_MAX_ESCALATIONS = 1   # 整条请求最多升级一次(4.4)
_MAX_REDOS = 1         # 同一 tier 质检 failed 最多重做一次(4.3)


def _run_tier(tier: str,
              message: str,
              history: list[dict],
              **kwargs):
    """按 tier 调用对应路径,返回事件生成器。

    complex 在阶段 5 接入 Plan-and-Execute 前先走 run_agent_graph(medium ReAct)。
    """
    if tier == "simple":
        return run_simple(message, history, **kwargs)
    return run_agent_graph(message, history, **kwargs)


def _last(seq):
    return seq[-1] if seq else None


def react_stream(message,
                 history,
                 max_total_seconds: int = 60,
                 on_event: Optional[Callable[[dict], None]] = None,
                 *,
                 thread_id: Optional[str] = None,
                 username: Optional[str] = None,
                 session_id: Optional[str] = None,
                 max_steps: int = 6):
    """生成器:yield SSE 事件 dict。

    流程(阶段 3 路由 + 阶段 4 质检/升级):
      1. classify_complexity 判 tier -> 先发 ``tier`` 事件。
      2. 跑该 tier 路径,实时转发 token 等事件,但**扣留**终端 ``done``。
      3. 拿到答案 + trace 后过 quality_gate:
         - passed           -> 补发警示(若有)再放行 held ``done``;
         - failed           -> 同 tier 重做一次(带 feedback);
         - needs_escalation -> 升一 tier 重跑(发 ``escalation`` 事件让前端重置),
                              整条请求最多升一次;complex 失败带警示放行。
    """
    if thread_id is None:
        thread_id = str(uuid.uuid4())

    # 1. 复杂度分类(失败兜底 medium,不阻断)
    decision = classify_complexity(message, history or [])
    tier = decision["tier"]

    # 2. 告知前端所选 tier(前端可据此展示不同 UI/进度)
    yield {"type": "tier", "tier": tier,
           "confidence": decision.get("confidence", 0.0),
           "source": decision.get("source", "fallback")}

    run_history = list(history or [])
    escalations = 0
    redos = 0
    held_done: Optional[dict] = None
    answer = ""

    while True:
        held_done = None
        answer = ""
        path_errored = False

        gen = _run_tier(
            tier, message, run_history,
            thread_id=thread_id, username=username, session_id=session_id,
            max_steps=max_steps, max_total_seconds=max_total_seconds,
            on_event=on_event,
        )
        for ev in gen:
            etype = ev.get("type")
            if etype == "assistant_message":
                answer = ev.get("content", "") or answer
            if etype == "error":
                path_errored = True
            if etype == "done":
                # 扣留 done:质检通过/放行后再发;升级/重做时丢弃
                held_done = ev
                continue
            yield ev

        # 路径异常且未产出 done:直接收尾,不再质检(错误事件已由 run_path 发出)
        if held_done is None:
            return

        # 路径自身报错(final_reason=error)但仍发了 done:不做升级/重做,放行
        if path_errored:
            yield held_done
            return

        # 3. 质检门
        trace = held_done.get("trace") or {}
        context = {
            "question": message,
            "history": run_history,
            "grounding": trace.get("grounding"),   # medium 已算过,复用避免重复 LLM
            "coverage": trace.get("coverage"),
        }
        result = quality_check(answer, context, tier=tier)
        verdict = result.get("verdict", "passed")
        warnings = list(result.get("warnings") or [])
        feedback = result.get("feedback", "") or ""

        if verdict == "needs_escalation":
            nxt = _NEXT_TIER.get(tier)
            if nxt is not None and escalations < _MAX_ESCALATIONS:
                escalations += 1
                redos = 0
                # 带上已有历史 + 失败答案 + 反馈,供升级后路径参考
                if answer:
                    run_history = run_history + [
                        {"role": "assistant", "content": answer},
                        {"role": "user", "content": feedback or "请更严谨地回答"},
                    ]
                yield {"type": "escalation", "from_tier": tier,
                       "to_tier": nxt, "reason": feedback}
                tier = nxt
                continue  # 丢弃当前 held_done,重跑升级路径

        if verdict == "failed" and redos < _MAX_REDOS:
            redos += 1
            yield {"type": "status",
                   "message": "质检未过,正在重新生成…"}
            if answer:
                run_history = run_history + [
                    {"role": "assistant", "content": answer},
                    {"role": "user", "content": feedback or "请重新生成回答"},
                ]
            continue  # 同 tier 重做,丢弃 held_done

        # passed,或已无升级/重做预算 -> 放行(有警示则以可见 status 提示,fail-open)
        for w in warnings:
            yield {"type": "status", "message": f"⚠️ {w}"}
        yield held_done
        return
