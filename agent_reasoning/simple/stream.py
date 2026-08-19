# -*- coding: utf-8 -*-
"""simple 路径:单轮直答,不绑定工具,lite 模型。

适用:闲聊、寒暄、元问题等明确不需要半导体领域检索的问题。
事件序列(2.1): ``status → token* → assistant_message``(质检/升级在阶段 4 接入,
trace/done 与流水包装在 2.4 由 run_path 统一处理)。

设计:
- 模型用 ``config.TIER_MODEL_SIMPLE``(doubao-seed-2.0-lite)。
- 不传 ``tools``/``tool_choice``,模型只生成文本,不会发 tool_calls。
- 长期记忆召回在调用方(runner 装配)完成后通过 ``recalled_memories`` 注入,
  本模块只负责把它拼进 system prompt 并做一次 LLM 调用。
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
    get_client, llm_create_with_retry, STREAM_TIMEOUT,
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
                         recalled_memories: Optional[list[dict]] = None,
                         *,
                         recorder: TraceRecorder,
                         trace_id: str,
                         t0: float):
    """生成器:执行 simple 单轮直答,yield SSE 事件 dict。

    yield: ``status`` → ``token*`` → ``assistant_message``。
    LLM 创建/流读取失败时 yield ``error`` 事件(不抛异常,交由 run_path 收尾)。

    :param recalled_memories: 已召回的长期记忆(由调用方在 runner 层完成,2.3/2.4)。
    :param recorder: TraceRecorder,记录 LLM 调用与 token。
    :param t0: 请求开始时间戳。
    """
    history = history or []
    recalled_memories = recalled_memories or []

    # ---- 构建消息:simple 裁剪版 system prompt(无工具说明/引用规则)----
    system_parts = [SIMPLE_SYSTEM_PROMPT]
    if recalled_memories:
        from memories.storage.long import format_memories_for_prompt
        block = format_memories_for_prompt(recalled_memories)
        if block:
            system_parts.append(block)
    system_content = "\n\n".join(system_parts)

    messages: list[dict[str, Any]] = [{"role": "system", "content": system_content}]
    for turn in history[-6:]:  # simple 只带最近少量历史,控制 lite 模型成本
        role = turn.get("role")
        if role in ("user", "assistant"):
            messages.append({"role": role, "content": turn.get("content", "")})
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
        err_doc = recorder.record_error(1, "llm_stream", e)
        yield {"type": "error_trace", "trace_id": trace_id, **err_doc}
        yield {"type": "error", "trace_id": trace_id, "step": 1,
               "message": f"流读取中断: {str(e)[:160]}"}
        recorder.finish_step(step_doc, "error")
        recorder.final_reason = "error"
        return

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

    # 与 medium 路径 finalize_node 对齐:发 meta(成本/性能)+ done(含 trace)收尾,
    # 让前端 SSE 处理逻辑三条路径一致。simple 无工具/检索,sources_count=0。
    yield meta_event(
        trace_id, t0, 1, {},
        tokens=recorder.total_tokens, tools_count=0, grounding_passed=None,
    )
    yield {"type": "done", "trace_id": trace_id, "trace": recorder.to_dict()}
