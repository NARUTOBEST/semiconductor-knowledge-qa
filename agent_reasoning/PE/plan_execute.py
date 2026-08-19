# -*- coding: utf-8 -*-
"""complex 路径:Plan-and-Execute(阶段 5)。

流程:
  1. ``generate_plan(force=True)`` 必经规划,发 ``plan`` 事件;planner 失败由调用方
     (runner.run_plan_execute)降级为普通 ReAct(5.8)。
  2. 对每个 step 调阶段 1.1 的 ``react_loop(step_instruction=step)``——每步独立
     messages / state / collected_sources(步骤间隔离,Q3 决策);转发子循环事件。
  3. 步骤失败处理:某步搜不到结果 -> 换词重试 1 次 -> 仍失败则标记 missing 跳过,
     不中断整体(5.3)。
  4. 跨步骤总预算:整体受 ``max_total_seconds`` 约束,按剩余时间给每步分配时间片,
     预算不足时跳过剩余步骤直接 synthesizer(5.5)。
  5. synthesizer:一次 LLM 调用整合原问题 + 各步结果(含缺失标记),流式输出
     ``synthesis_start → token* → assistant_message``(5.4)。
  6. grounding:synthesizer 之后做引用 + 忠实度,写入 recorder 供质检门 complex
     深度判定(5.6)。

5.7 决策:P&E **不挂 CoverageTracker**——每步都被实际执行,覆盖度由步骤完成度
(``missing`` 列表)结构性保证;未取到结果的步骤作为 ``uncovered_steps`` 写入
``trace["coverage"]``,由共享质检门 complex 深度检查 synthesizer 是否漏用。
"""
from __future__ import annotations

import logging
import time
from typing import Any, Optional

from langchain_core.messages import SystemMessage

import config as C  # noqa: E402
from support.metrics import metrics  # noqa: E402
from system_prompt import SYSTEM_PROMPT  # noqa: E402
from agent_reasoning.ReAct.core.loop import react_loop  # noqa: E402
from agent_reasoning.ReAct.support.llm import (  # noqa: E402
    get_client, llm_create_with_retry, STREAM_TIMEOUT,
)
from agent_reasoning.ReAct.support.answer_grounding import grounding_check  # noqa: E402
from agent_reasoning.ReAct.trace import TraceRecorder  # noqa: E402
from agent_reasoning.ReAct.utils.events import meta_event  # noqa: E402

logger = logging.getLogger("agent")

# 每步分配的时间片上限(s);实际取 min(此值, 剩余时间-合成预留)
_STEP_BUDGET = 20
# synthesizer + grounding 预留时间(s),低于此值不再开新步
_SYNTH_RESERVE = 15
# 子循环每步最多内部 ReAct 轮次
_STEP_MAX_ROUNDS = 3


def _extract_usage(chunk):
    u = getattr(chunk, "usage", None)
    if not u:
        return None
    return {
        "prompt_tokens": getattr(u, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(u, "completion_tokens", 0) or 0,
        "total_tokens": getattr(u, "total_tokens", 0) or 0,
    }


def _run_one_step(step_instruction: str,
                  *,
                  parent_recorder: TraceRecorder,
                  trace_id: str,
                  configurable_base: dict[str, Any],
                  step_budget: float,
                  started_at: float):
    """跑一个步骤的 ReAct 子循环,返回 (events, result, child_recorder)。

    result: {"answer", "sources"(list), "search_count", "final_reason"}。
    步骤间隔离(Q3):每次新建独立 state/messages,且用独立子 TraceRecorder 记录该步
    的 tools/llm,完成后由调用方折叠进父 recorder 的嵌套结构,不污染父 steps 列表。
    """
    initial_messages = [SystemMessage(content=SYSTEM_PROMPT)]
    events: list[dict] = []
    final = {}

    child_recorder = TraceRecorder(trace_id, started_at, step_instruction)
    configurable = dict(configurable_base)
    configurable["trace_recorder"] = child_recorder

    gen = react_loop(
        initial_messages,
        step_instruction=step_instruction,
        max_steps=_STEP_MAX_ROUNDS,
        max_total_seconds=int(max(step_budget, 5)),
        started_at=started_at,
        configurable=configurable,
        bind_tools=True,
        skip_rewrite=False,
        skip_recall=True,   # 每步是子任务,不重复做长期记忆召回(外层已做)
        question=step_instruction,
        trace_id=trace_id,
    )
    try:
        while True:
            ev = next(gen)
            events.append(ev)
    except StopIteration as stop:
        final = stop.value or {}

    sources_map = final.get("collected_sources") or {}
    # __reset__ 哨兵不是真实来源,过滤掉
    sources = [s for k, s in sources_map.items()
               if k != "__reset__" and isinstance(s, dict)]
    return events, {
        "answer": (final.get("full_reply") or "").strip(),
        "sources": sources,
        "search_count": int(final.get("search_count") or 0),
        "final_reason": final.get("final_reason"),
    }, child_recorder


def _step_found_nothing(result: dict) -> bool:
    """该步是否未取得有效结果(用于触发换词重试)。"""
    return not result["answer"] or not result["sources"]


def _synthesize(question: str,
                step_results: list[dict],
                *,
                parent_recorder: TraceRecorder,
                trace_id: str,
                t0: float):
    """一次 LLM 调用整合各步结果,生成器:yield synthesis_start/token*/assistant_message。

    返回 (answer, usage, error, synth_recorder)。synthesizer 用独立子 recorder 记录
    (不污染父 steps),token 用量由调用方折叠;失败时由调用方拼接各步结果降级(5.4/5.8)。
    """
    blocks = []
    for i, r in enumerate(step_results, 1):
        if r.get("missing"):
            blocks.append(f"【步骤{i}】{r['instruction']}\n(未检索到资料,缺失)")
        else:
            blocks.append(f"【步骤{i}】{r['instruction']}\n{r['answer']}")
    steps_text = "\n\n".join(blocks)

    prompt = (
        "你是半导体领域知识助手。下面是针对一个复杂问题分步骤检索/分析得到的结果,"
        "请据此整合成一份结构清晰、忠于资料的最终回答。\n"
        "若某步骤标注为缺失,如实说明该部分资料不足,不要编造;引用资料时保留"
        "[文档名 p页码] 标注。\n\n"
        f"原问题:{question}\n\n各步结果:\n{steps_text}\n\n最终回答:"
    )
    yield {"type": "synthesis_start", "trace_id": trace_id}

    synth_recorder = TraceRecorder(trace_id, t0, "synthesis")
    step_doc = synth_recorder.new_step(1)
    t_llm = time.time()
    stream, err = llm_create_with_retry(
        get_client(), trace_id=trace_id,
        model=C.OPENAI_TEXT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True, stream_options={"include_usage": True},
        temperature=0.3, timeout=STREAM_TIMEOUT,
    )
    if err is not None:
        err_doc = synth_recorder.record_error(1, "synthesis_create", err)
        yield {"type": "error_trace", "trace_id": trace_id, **err_doc}
        synth_recorder.finish_step(step_doc, "error")
        return "", None, err, synth_recorder

    content_buf = ""
    usage = None
    finish_reason = None
    try:
        for chunk in stream:
            choice = chunk.choices[0] if getattr(chunk, "choices", None) else None
            if choice is not None:
                if getattr(choice, "finish_reason", None):
                    finish_reason = choice.finish_reason
                delta = choice.delta
                if getattr(delta, "content", None):
                    content_buf += delta.content
                    yield {"type": "token", "delta": delta.content,
                           "trace_id": trace_id, "step": 1}
            cu = _extract_usage(chunk)
            if cu:
                usage = cu
    except Exception as e:
        err_doc = synth_recorder.record_error(1, "synthesis_stream", e)
        yield {"type": "error_trace", "trace_id": trace_id, **err_doc}
        synth_recorder.finish_step(step_doc, "error")
        return content_buf, usage, e, synth_recorder

    synth_recorder.record_llm(
        step_doc, finish_reason=finish_reason, usage=usage,
        thought=content_buf, tool_calls=[],
        stream_duration_ms=int((time.time() - t_llm) * 1000),
    )
    if usage:
        metrics.record_tokens(usage["prompt_tokens"], usage["completion_tokens"])
    synth_recorder.finish_step(step_doc, "answer")

    yield {"type": "assistant_message", "trace_id": trace_id,
           "content": content_buf}
    return content_buf, usage, None, synth_recorder


def plan_execute_stream(question: str,
                        history: Optional[list[dict]],
                        steps: list[str],
                        *,
                        recorder: TraceRecorder,
                        trace_id: str,
                        t0: float,
                        thread_id: Optional[str] = None,
                        username: Optional[str] = None,
                        max_total_seconds: int = 60,
                        planner_error: Optional[str] = None,
                        planner_duration_ms: Optional[int] = None):
    """生成器:执行已规划好的 P&E 步骤并综合作答,yield SSE 事件 dict。

    ``steps`` 由调用方(5.1 generate_plan)生成并已判定非空;planner 失败的降级
    在 runner 层处理。末尾 yield meta + done(含 trace),与其它路径收尾一致。
    """
    history = history or []
    max_total = int(max_total_seconds) if max_total_seconds is not None else 60
    # 每步用独立子 TraceRecorder(在 _run_one_step 内注入),base 不带父 recorder;
    # 5.7:P&E 不挂 CoverageTracker;覆盖度由步骤 missing 列表结构性保证
    configurable_base: dict[str, Any] = {
        "thread_id": thread_id,
        "user_id": username,
    }

    # 计划开始事件(planner 已在 runner 层跑过,这里补发 plan 事件给前端)
    recorder.record_plan(steps, question)
    recorder.begin_plan_execute(question, steps)
    recorder.record_pe_planner(steps=steps, error=planner_error,
                               duration_ms=planner_duration_ms)
    yield {"type": "plan", "trace_id": trace_id, "steps": steps, "question": question}

    step_results: list[dict] = []
    all_sources: list[dict] = []

    for idx, instruction in enumerate(steps, 1):
        remaining = max_total - (time.time() - t0)
        # 5.5 预算不足(且已至少执行一步):剩余步骤全部标记缺失,直接进 synthesizer
        if idx > 1 and remaining < _SYNTH_RESERVE:
            yield {"type": "status", "trace_id": trace_id,
                   "message": f"响应时间预算不足,跳过剩余 {len(steps) - idx + 1} 步直接综合…"}
            for later in steps[idx - 1:]:
                step_results.append({"instruction": later, "answer": "",
                                     "sources": [], "missing": True})
                recorder.record_pe_step(
                    idx, later, None, missing=True, retried=False,
                    elapsed_ms=0,
                )
            break

        step_budget = min(_STEP_BUDGET, remaining - _SYNTH_RESERVE)
        yield {"type": "status", "trace_id": trace_id,
               "message": f"执行计划第 {idx}/{len(steps)} 步:{instruction}"}

        step_start = time.time()
        events, result, child = _run_one_step(
            instruction, parent_recorder=recorder, trace_id=trace_id,
            configurable_base=configurable_base, step_budget=step_budget,
            started_at=step_start,
        )
        for ev in events:
            yield ev

        missing = False
        retried = False
        # 5.3 搜不到结果 -> 换词重试 1 次
        if _step_found_nothing(result):
            retried = True
            yield {"type": "status", "trace_id": trace_id,
                   "message": f"第 {idx} 步未检索到结果,更换关键词重试…"}
            retry_instruction = (f"{instruction}(上一次未找到资料,"
                                 "请更换关键词或角度重试)")
            retry_start = time.time()
            retry_budget = min(_STEP_BUDGET,
                               max_total - (time.time() - t0) - _SYNTH_RESERVE)
            _, result, child = _run_one_step(
                retry_instruction, parent_recorder=recorder, trace_id=trace_id,
                configurable_base=configurable_base, step_budget=max(retry_budget, 5),
                started_at=retry_start,
            )
            if _step_found_nothing(result):
                missing = True
                yield {"type": "status", "trace_id": trace_id,
                       "message": f"第 {idx} 步仍无结果,标记缺失并继续。"}

        result["instruction"] = instruction
        result["missing"] = missing
        step_results.append(result)
        all_sources.extend(result["sources"])
        # 折叠该步子 recorder 进父 trace 的嵌套结构(含 token 累加)
        recorder.record_pe_step(
            idx, instruction, child,
            answer=result["answer"],
            sources_count=len(result["sources"]),
            search_count=result["search_count"],
            final_reason=result["final_reason"],
            missing=missing, retried=retried,
            elapsed_ms=int((time.time() - step_start) * 1000),
        )

    # ---- synthesizer(5.4)----
    synth_gen = _synthesize(question, step_results,
                            parent_recorder=recorder, trace_id=trace_id, t0=t0)
    answer = ""
    usage = None
    synth_err = None
    synth_recorder = None
    synth_start = time.time()
    try:
        while True:
            ev = next(synth_gen)
            if ev.get("type") == "assistant_message":
                answer = ev.get("content", "")
            yield ev
    except StopIteration as stop:
        answer, usage, synth_err, synth_recorder = stop.value  # type: ignore[misc]

    # 折叠 synthesizer 子 recorder 的 token 到父,并记录嵌套 synthesizer 段
    if synth_recorder is not None:
        recorder.total_tokens["prompt"] += synth_recorder.total_tokens["prompt"]
        recorder.total_tokens["completion"] += synth_recorder.total_tokens["completion"]
        recorder.total_tokens["total"] += synth_recorder.total_tokens["total"]
    recorder.record_pe_synthesizer(
        answer=answer, usage=usage, error=synth_err,
        duration_ms=int((time.time() - synth_start) * 1000),
    )

    # 5.8 synthesizer 失败 -> 拼接各步结果降级返回
    if synth_err is not None or not answer.strip():
        fallback_parts = []
        for i, r in enumerate(step_results, 1):
            if not r.get("missing") and r.get("answer"):
                fallback_parts.append(f"【{r['instruction']}】\n{r['answer']}")
        answer = "\n\n".join(fallback_parts) or "各步骤均未能生成结果,请稍后重试。"
        recorder.final_reason = "answer"
        yield {"type": "status", "trace_id": trace_id,
               "message": "整合服务异常,已直接拼接各步结果。"}
        yield {"type": "assistant_message", "trace_id": trace_id, "content": answer}

    # ---- grounding(5.6):synthesizer 之后做引用+忠实度 ----
    grounding = None
    if answer.strip():
        try:
            grounding = grounding_check(answer, all_sources)
            recorder.record_grounding(grounding["passed"], grounding["warnings"])
            yield {"type": "grounding", "trace_id": trace_id, **grounding}
        except Exception as e:
            logger.warning("P&E grounding error (fail-open): %s", e)
            grounding = {"passed": True,
                         "warnings": ["答案质检因临时异常暂不可用,请自行核实答案"]}
            recorder.record_grounding(True, grounding["warnings"])

    # 5.7:把缺失步骤作为覆盖度结果写 trace,供质检门 complex 深度检查
    recorder.coverage = {
        "covered": [r["instruction"] for r in step_results if not r.get("missing")],
        "uncovered_steps": [r["instruction"] for r in step_results
                            if r.get("missing")],
        "reason": "plan_execute",
    }

    recorder.final_reason = "answer"
    yield meta_event(
        trace_id, t0, len(step_results), {str(i): s for i, s in enumerate(all_sources)},
        tokens=recorder.total_tokens,
        tools_count=sum(1 for r in step_results if r["sources"]),
        grounding_passed=(grounding or {}).get("passed"),
    )
    yield {"type": "done", "trace_id": trace_id, "trace": recorder.to_dict()}
