# -*- coding: utf-8 -*-
"""知识助手 -- 聊天服务入口(薄封装)。

本文件只做一件事:提供 react_stream(),先做复杂度路由(classify_complexity),
按 tier 分发到 simple(单轮直答)/ react(ReAct 检索循环),并在答案产出后、
``done`` 事件前过共享质检门(quality_gate),按 verdict 放行 / 重做一次 /
simple→react 升级一次。

具体的 ReAct 主循环、LLM 调用、工具调用全部在 agent_reasoning 包内。

为兼容现有调用方(health/service.py 及若干测试以 chat.service.get_client 等
路径导入/打桩),这里重导出 LLM 相关符号。
"""
from __future__ import annotations

import time
import uuid
import logging
from typing import Callable, Optional

from agent_reasoning.ReAct.support.runner import run_agent_graph
from agent_reasoning.simple import run_simple
from agent_reasoning.raglite import run_raglite
from agent_reasoning.ReAct.support.llm import get_client, llm_create_with_retry, LLM_RETRIES
from agent_reasoning.router import classify_complexity, classify_and_clarify, _obviously_simple
from agent_reasoning.clarify import check_clarify
from agent_reasoning.quality_gate import check as quality_check
import config as C

logger = logging.getLogger("agent")

__all__ = [
    "react_stream",
    "get_client",
    "llm_create_with_retry",
    "LLM_RETRIES",
]

# 升级链:simple/raglite(快路径) -> react(检索问答);react 已为最高 tier。
_NEXT_TIER = {"simple": "react", "raglite": "react", "react": None}
_MAX_ESCALATIONS = 1   # 整条请求最多升级一次
_MAX_REDOS = 1         # 同一 tier 质检 failed 最多重做一次

# run_simple 不认这些 ReAct 预算/重做参数,分发前剔除。
_NON_SIMPLE_KW = ("max_steps", "max_total_seconds", "hard_deadline",
                  "qc_feedback", "search_future")


def _speculative_search(message: str):
    """投机检索:在后台工具池预发一次 search_text,与路由/分类重叠执行。
    返回 future(.result() -> (result, err),与 call_with_resilience 同构);
    无人消费时结果自然丢弃(daemon 池,不阻塞收尾)。"""
    from tools import dispatch, registry
    from agent_reasoning.ReAct.support.tool_resilience import call_with_resilience
    from agent_reasoning.ReAct.support.tool_pool import get_pool
    spec = registry.get("search_text")
    args = {"query": message, "k": C.RAGLITE_SEARCH_K,
            "score_ratio": C.RAGLITE_SEARCH_SCORE_RATIO}
    return get_pool().submit(
        call_with_resilience, "search_text", args, spec, invoke=dispatch)


def _qc_feedback_msg(feedback: str) -> dict:
    """质检/升级反馈包装成 user 槽位的消息(维持 user/assistant 交替结构),
    但明确标注这是系统内部自动质检结果、并非用户发言。
    否则模型会误以为用户在纠错,回出"您说得对/感谢指出"这类对不存在的用户发言的回应。"""
    fb = (feedback or "").strip() or "请更严谨地重新生成回答"
    return {"role": "user", "content": (
        "[系统内部质检提示 —— 这不是用户消息,用户并未发言或指出问题。]"
        f"上一版回答未通过自动核查:{fb}。"
        "请据此自检并重新给出完整、准确、表述自然的最终回答;"
        "不要向用户致谢或表示认同(如“您说得对/感谢指出”),直接呈现修正后的答案。"
    )}


def _run_tier(tier: str,
              message: str,
              history: list[dict],
              **kwargs):
    """按 tier 调用对应路径,返回事件生成器。

    simple  -> run_simple(单轮直答,忽略步数/时长/硬截止);
    raglite -> run_raglite(1 次检索 + 1 次作答快路径);
    react   -> run_agent_graph(ReAct 检索循环)。
    """
    if tier == "simple":
        simple_kw = {k: v for k, v in kwargs.items() if k not in _NON_SIMPLE_KW}
        return run_simple(message, history, **simple_kw)
    if tier == "raglite":
        return run_raglite(message, history, **kwargs)
    # react 消费投机检索 future:首轮作为"预检索"注入(见 nodes.agent_node),
    # 模型可直接引用原始问题的检索结果,省一步检索;simple 不消费。
    return run_agent_graph(message, history, **kwargs)


def react_stream(message,
                 history,
                 max_total_seconds: Optional[int] = None,
                 on_event: Optional[Callable[[dict], None]] = None,
                 *,
                 thread_id: Optional[str] = None,
                 username: Optional[str] = None,
                 session_id: Optional[str] = None,
                 max_steps: Optional[int] = None):
    """生成器:yield SSE 事件 dict。

    流程:
      1. classify_complexity 判 tier -> 先发 ``tier`` 事件。
      2. 跑该 tier 路径,实时转发 token 等事件,但**扣留**终端 ``done``。
      3. 拿到答案后过 quality_gate:
         - passed           -> 补发警示(若有)再放行 held ``done``;
         - failed           -> 同 tier 重做一次(带 feedback);
         - needs_escalation -> simple→react 升级重跑(发 ``escalation`` 事件),
                              整条请求最多升一次。
    """
    if thread_id is None:
        thread_id = str(uuid.uuid4())

    # 入口等待门:上一轮的后台记忆链(独立记忆图)没跑完时先等它,保证本轮
    # 构建上下文时长期偏好/摘要/compact 已就位。有界超时(MEM_WAIT_IDLE_TIMEOUT,
    # 默认 10s)兜底放行——主链路永不被记忆链拖死;未完成的幂等重做留给下一轮。
    try:
        from agent_reasoning.ReAct.support.memory_background import wait_previous_turn
        wait_previous_turn(username, thread_id)
    except Exception:
        pass

    # 0+1. 入口判定(澄清 + 路由)。ROUTER_FUSED=1 时合并为单次判定
    #      (规则可判时零 LLM),替代旧的 clarify + router 两次串行调用。
    search_future = None
    if getattr(C, "ROUTER_FUSED", True):
        # 投机检索:在融合分类 LLM 调用前预发 search_text,两路重叠省 1.5~2.5s;
        # 规则短路/rule-routed 场景该 future 就是 raglite 本来要做的检索(零浪费),
        # react 则作为首轮预检索注入。仅在明显闲聊(问候/超短)时跳过——领域
        # 关键词表无法穷举(实测"重掺砷硅单晶/离子发生器"均漏),宁滥勿缺:
        # 一次后台检索成本远低于 react 漏检索导致的零来源作答。
        if (getattr(C, "RAGLITE_SPECULATIVE_SEARCH", True)
                and getattr(C, "RAGLITE_ENABLED", True)
                and not _obviously_simple(message)):
            try:
                search_future = _speculative_search(message)
            except Exception:  # noqa: BLE001  投机失败不影响主链路
                search_future = None
        try:
            entry = classify_and_clarify(message, history or [])
        except Exception:
            logger.warning("classify_and_clarify 异常,兜底 react", exc_info=True)
            entry = {"need_clarify": False, "tier": "react",
                     "confidence": 0.0, "source": "fallback"}
    else:
        try:
            clarify = check_clarify(message, history or [])
        except Exception:
            logger.warning("check_clarify 异常,跳过澄清", exc_info=True)
            clarify = {"need_clarify": False}
        if clarify.get("need_clarify"):
            yield {"type": "clarify",
                   "question": clarify.get("question", ""),
                   "options": clarify.get("options") or [],
                   "source": clarify.get("source", "llm")}
            yield {"type": "done", "trace": {}}
            return
        decision = classify_complexity(message, history or [])
        entry = {"need_clarify": False, "tier": decision["tier"],
                 "confidence": decision.get("confidence", 0.0),
                 "source": decision.get("source", "fallback")}
    if entry.get("need_clarify"):
        yield {"type": "clarify",
               "question": entry.get("question", ""),
               "options": entry.get("options") or [],
               "source": entry.get("source", "llm")}
        yield {"type": "done", "trace": {}}
        return

    tier = entry["tier"] or "react"

    # 2. 告知前端所选 tier。raglite 对外仍报 "react"(eval tier_ok / 前端零改动),
    #    真实路径放 path 字段供观测。
    yield {"type": "tier",
           "tier": "react" if tier == "raglite" else tier,
           "path": tier,
           "confidence": entry.get("confidence", 0.0),
           "source": entry.get("source", "fallback")}

    run_history = list(history or [])
    escalations = 0
    redos = 0
    held_done: Optional[dict] = None
    answer = ""
    qc_feedback: Optional[str] = None

    # 端到端硬预算:整条请求(含升级/重做)共享一个绝对截止墙钟。
    # 初始 tier 预算 + 预留一次升级的 react 预算(raglite 25s + react 25s),
    # 各 tier 自身的 max_total_seconds 仍约束单次执行。
    deadline_cap = (max_total_seconds if max_total_seconds is not None
                    else int(C.TIER_CONFIG.get(tier, C.TIER_CONFIG["react"])
                             ["max_total_seconds"])
                    + (int(C.TIER_CONFIG["react"]["max_total_seconds"])
                       if tier != "react" else 0))
    hard_deadline = time.time() + deadline_cap

    while True:
        held_done = None
        answer = ""
        path_errored = False
        # qc_feedback 不在循环顶复位:升级/重做分支设置后,下一轮迭代消费
        # (普通首轮为 None;重做分支总会覆盖,无残留路径)。

        # 硬预算将尽:不再开启新路径/重做,直接收尾。
        if hard_deadline - time.time() < 3.0:
            yield {"type": "status",
                   "message": "已达整体响应时限,请稍后重试或简化问题。"}
            yield {"type": "done", "trace": {}}
            return

        # 每个 tier 用自己的步数/时长预算;显式入参覆盖配置(测试/调用方可用)。
        tcfg = C.TIER_CONFIG.get(tier, C.TIER_CONFIG["react"])
        tier_max_steps = max_steps if max_steps is not None else tcfg["max_steps"]
        tier_max_total = (max_total_seconds if max_total_seconds is not None
                          else tcfg["max_total_seconds"])

        gen = _run_tier(
            tier, message, run_history,
            thread_id=thread_id, username=username, session_id=session_id,
            max_steps=tier_max_steps, max_total_seconds=tier_max_total,
            hard_deadline=hard_deadline,
            qc_feedback=qc_feedback,
            on_event=on_event,
            # 投机检索 future:raglite 直接消费,react 作为预检索注入;simple 不消费
            search_future=(search_future if tier in ("raglite", "react") else None),
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

        # 路径异常且未产出 done:不再质检/升级(答案为空),但必须补发终态 done
        # (错误事件已由 run_path 发出)。done 是 SSE 契约的唯一终端事件,
        # 无条件补齐,前端不能依赖"连接关闭"这类传输层副作用收尾。
        if held_done is None:
            yield {"type": "done", "trace": {}, "final_reason": "error"}
            return

        # 路径自身报错(final_reason=error)但仍发了 done:不做升级/重做,放行
        if path_errored:
            yield held_done
            return

        # 3. 质检门
        context = {
            "question": message,
            "history": run_history,
            # react 路径在 done 事件带回本轮最高检索相关分/检索次数,供低置信判定;
            # simple 直答不带这两个字段(默认 0,不触发 react 低置信规则)。
            "retrieval_max_score": float(held_done.get("retrieval_max_score") or 0.0),
            "search_count": int(held_done.get("search_count") or 0),
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
                        _qc_feedback_msg(feedback or "请更严谨地回答"),
                    ]
                yield {"type": "escalation", "from_tier": tier,
                       "to_tier": nxt, "reason": feedback}
                qc_feedback = _qc_feedback_msg(feedback or "请更严谨地回答")["content"]
                tier = nxt
                continue  # 丢弃当前 held_done,重跑升级路径

        if verdict == "failed" and redos < _MAX_REDOS:
            redos += 1
            # 先发 reflect 让前端清空已上屏的旧答案,否则重做的 token 会追加在旧答案后。
            yield {"type": "reflect", "feedback": feedback or "质检未过,重新生成"}
            yield {"type": "status",
                   "message": "质检未过,正在换关键词重新检索作答…"}
            if answer:
                run_history = run_history + [
                    {"role": "assistant", "content": answer},
                    _qc_feedback_msg(feedback or "请重新生成回答"),
                ]
            # 同一 thread_id 重做,只清空 ReAct 工作记忆(checkpoint messages):
            # 等价旧版"换新 thread_id = 全新 checkpoint"的清理效果,但短期流水/
            # 事实表/摘要/入口等待门仍归原会话键——重做轮问答对下一轮召回可见
            # (修复换 thread 导致的下一轮多轮失忆)。清空后重做轮为冷启动,由
            # build_messages 以 Redis 短期流水为权威重建完整上下文;质检反馈经
            # qc_feedback 显式注入(不再依赖 history 夹带)。
            # 仅 react 有 checkpoint;simple/raglite 无工作记忆可清。
            if tier == "react":
                try:
                    from agent_reasoning.ReAct.support.memory_background import reset_react_memory
                    reset_react_memory(username, thread_id)
                except Exception:
                    logger.warning("质检重做清空工作记忆失败,将带旧上下文重做",
                                   exc_info=True)
            qc_feedback = _qc_feedback_msg(feedback or "请重新生成回答")["content"]
            continue  # 同 tier 干净重做,丢弃 held_done

        # passed,或已无升级/重做预算 -> 放行(有警示则以可见 status 提示,fail-open)
        for w in warnings:
            yield {"type": "status", "message": f"⚠️ {w}"}
        yield held_done
        return
