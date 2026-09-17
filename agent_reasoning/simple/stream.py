# -*- coding: utf-8 -*-
"""simple 路径:单轮直答,不绑定工具,lite 模型。

适用:闲聊、寒暄、元问题等明确不需要半导体领域检索的问题。
事件序列(2.1): ``status → token* → assistant_message``(质检/升级在阶段 4 接入,
trace/done 与流水包装在 2.4 由 run_path 统一处理)。

设计:
- 模型用 ``config.TIER_MODEL_SIMPLE``(doubao-seed-2.0-lite)。
- 不传 ``tools``/``tool_choice``,模型只生成文本,不会发 tool_calls。
- LLM 失败 fail-open:发 error 事件(由 run_path 兜底),不伪造答案。
"""
from __future__ import annotations

import logging
import time
from typing import Any, Optional

import config as C  # noqa: E402
from support.metrics import metrics  # noqa: E402
from system_prompt import SIMPLE_SYSTEM_PROMPT  # noqa: E402
from agent_reasoning.ReAct.support.llm import (  # noqa: E402
    get_client, llm_create_with_retry, STREAM_TIMEOUT, arm_stream_watchdog,
)
from agent_reasoning.ReAct.trace import TraceRecorder  # noqa: E402
from agent_reasoning.ReAct.utils.events import meta_event  # noqa: E402

logger = logging.getLogger("agent")


def _extract_usage(chunk):
    u = getattr(chunk, "usage", None)
    if not u:
        return None
    return {
        "prompt_tokens": getattr(u, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(u, "completion_tokens", 0) or 0,
        "total_tokens": getattr(u, "total_tokens", 0) or 0,
    }


def simple_answer_stream(message: str,
                         history: Optional[list[dict]] = None,
                         *,
                         recorder: TraceRecorder,
                         trace_id: str,
                         t0: float,
                         username: Optional[str] = None,
                         thread_id: Optional[str] = None):
    """生成器:执行 simple 单轮直答,yield SSE 事件 dict。

    yield: ``status`` → ``token*`` → ``assistant_message``。
    LLM 创建/流读取失败时 yield ``error`` 事件(不抛异常,交由 run_path 收尾)。

    :param recorder: TraceRecorder,记录 LLM 调用与 token。
    :param t0: 请求开始时间戳。
    :param username: 登录用户名(用于长期偏好召回/抽取;匿名 None 则跳过)。
    :param thread_id: 按用户隔离后的存储键(偏好来源标记)。
    """
    history = history or []

    # 首轮多轮上下文以服务端 Redis 短期流水为权威来源(不依赖前端重发 history);
    # Redis 不可用/无流水时退回前端 history 兜底。simple 只带最近少量历史,控 lite 成本。
    prior_turns: list[dict] = []
    try:
        from memories.orchestration.short.recall import recent_dialogue_messages
        prior_turns = recent_dialogue_messages(thread_id, message, limit=6)
    except Exception:  # noqa: BLE001  旁路:取短期流水失败不阻断
        prior_turns = []
    if not prior_turns:
        prior_turns = [
            {"role": t.get("role"), "content": t.get("content", "")}
            for t in history[-6:]
            if t.get("role") in ("user", "assistant")
        ]

    # ---- 构建消息:simple 裁剪版 system prompt(无工具说明/引用规则)----
    # 注:长期记忆召回在 react 路径里是模型 auto 决策触发的【专用图节点 recall_memory】;
    # simple 路径不绑定工具(闲聊直答),故不做记忆召回。答完后的后台偏好抽取(写入)仍保留。
    system_content = SIMPLE_SYSTEM_PROMPT
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_content}]
    for turn in prior_turns:
        messages.append({"role": turn["role"], "content": turn.get("content", "")})
    messages.append({"role": "user", "content": message})

    yield {"type": "status", "message": "思考中…", "trace_id": trace_id, "step": 1}

    step_doc = recorder.new_step(1)
    t_llm = time.time()
    stream, err = llm_create_with_retry(
        get_client(), trace_id=trace_id,
        model=C.TIER_MODEL_SIMPLE,
        messages=messages,
        # 不传 tools/tool_choice:lite 模型只生成文本
        stream=True, stream_options={"include_usage": True},
        temperature=0.3, timeout=STREAM_TIMEOUT,
    )
    if err is not None:
        err_doc = recorder.record_error(1, "llm_create", err)
        yield {"type": "error_trace", "trace_id": trace_id, **err_doc}
        yield {"type": "error", "trace_id": trace_id, "step": 1,
               "message": f"模型请求失败: {str(err)[:160]}"}
        recorder.finish_step(step_doc, "error")
        recorder.final_reason = "error"
        return

    content_buf = ""
    usage = None
    finish_reason = None
    ttft_ms = None
    # 总时长看门狗:思考模型流式 reasoning 持续到达会绕过 STREAM_TIMEOUT 空闲超时
    _wd_cancel, _wd_killed = arm_stream_watchdog(
        stream, float(getattr(C, "REACT_STREAM_DEADLINE_S", 45)))
    try:
        for chunk in stream:
            choice = chunk.choices[0] if getattr(chunk, "choices", None) else None
            if choice is not None:
                if getattr(choice, "finish_reason", None):
                    finish_reason = choice.finish_reason
                delta = choice.delta
                if getattr(delta, "content", None):
                    if ttft_ms is None:
                        ttft_ms = int((time.time() - t_llm) * 1000)
                    content_buf += delta.content
                    yield {"type": "token", "delta": delta.content,
                           "trace_id": trace_id, "step": 1}
            chunk_usage = _extract_usage(chunk)
            if chunk_usage:
                usage = chunk_usage
    except Exception as e:
        if not _wd_killed.is_set():
            err_doc = recorder.record_error(1, "llm_stream", e)
            yield {"type": "error_trace", "trace_id": trace_id, **err_doc}
            yield {"type": "error", "trace_id": trace_id, "step": 1,
                   "message": f"流读取中断: {str(e)[:160]}"}
            recorder.finish_step(step_doc, "error")
            recorder.final_reason = "error"
            return
        # 看门狗截断:已有部分正文流给客户端,按正常结束收尾
    finally:
        _wd_cancel()

    recorder.record_llm(
        step_doc, finish_reason=finish_reason, usage=usage,
        thought=content_buf, tool_calls=[],
        stream_duration_ms=int((time.time() - t_llm) * 1000),
        time_to_first_token_ms=ttft_ms,
    )
    if usage:
        metrics.record_tokens(usage["prompt_tokens"], usage["completion_tokens"])
    recorder.finish_step(step_doc, "answer")
    recorder.final_reason = "answer"

    yield {"type": "assistant_message", "trace_id": trace_id,
           "content": content_buf}

    # 记忆维护闭环:定稿后非阻塞提交给后台记忆管道(与 react 路径同一管道/独立记忆图;
    # 按会话串行,不占请求流)。simple 无 checkpointer,compact_applier 不注入
    # (仅事实/长期/摘要文件落盘)。失败静默,绝不影响应答。
    try:
        from langchain_core.messages import AIMessage, HumanMessage
        # 整理上下文同样以 Redis 短期流水为准(prior_turns),前端 history 不再作为来源。
        lc_messages = []
        for turn in prior_turns:
            if turn.get("role") == "user":
                lc_messages.append(HumanMessage(content=turn.get("content", "")))
            elif turn.get("role") == "assistant":
                lc_messages.append(AIMessage(content=turn.get("content", "")))
        lc_messages.append(HumanMessage(content=message))
        lc_messages.append(AIMessage(content=content_buf))
        from agent_reasoning.ReAct.support.memory_background import submit_turn_memory
        submit_turn_memory(
            username=username, store_thread_id=thread_id,
            question=message, answer=content_buf,
            messages=lc_messages, final_reason="answer")
    except Exception:
        pass

    # 与 medium 路径 finalize_node 对齐:发 meta(成本/性能)+ done(含 trace)收尾,
    # 让前端 SSE 处理逻辑三条路径一致。simple 无工具/检索,sources_count=0。
    yield meta_event(
        trace_id, t0, 1, {},
        tokens=recorder.total_tokens, tools_count=0,
    )
    yield {"type": "done", "trace_id": trace_id, "trace": recorder.to_dict(),
           # Req5:done 强制带终态原因;simple 不产生检索,分/次数为 0。
           "final_reason": recorder.final_reason or "answer",
           "retrieval_max_score": 0.0, "search_count": 0}
