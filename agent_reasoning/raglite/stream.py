# -*- coding: utf-8 -*-
"""raglite 快路径:1 次检索 + 1 次主模型流式作答。

事件序列:sources → status → token* → assistant_message(sources 在生成前发出,
即使生成失败来源也已透出,评测 R@3 依赖该事件)。无工具绑定,不会出现
tool_calls,token 即时下发,无 REACT_STREAM_BUFFER 缓冲。

设计:
- 检索经韧性中间件 call_with_resilience(熔断/重试与 react 同一套);
  服务层投机检索的 future 传入则直接取结果(与路由分类重叠)。
- 模型用 C.TIER_MODEL_REACT(主模型,保忠实度),max_tokens 封顶。
- prompt 只喂前 RAGLITE_PROMPT_CHUNKS 块(控 prefill),sources 事件发
  RAGLITE_SEARCH_K 条(评测来源池不被 prompt 截断稀释)。
"""
from __future__ import annotations

import logging
import time
from typing import Any, Optional

import config as C  # noqa: E402
from support.metrics import metrics  # noqa: E402
from system_prompt import RAGLITE_SYSTEM_PROMPT  # noqa: E402
from tools import dispatch, registry  # noqa: E402
from agent_reasoning.ReAct.support.llm import (  # noqa: E402
    get_client, llm_create_with_retry, STREAM_TIMEOUT, no_think_extra,
    arm_stream_watchdog,
)
from agent_reasoning.ReAct.support.tool_resilience import call_with_resilience  # noqa: E402
from agent_reasoning.ReAct.trace import TraceRecorder  # noqa: E402
from agent_reasoning.ReAct.utils.events import meta_event  # noqa: E402
from agent_reasoning.ReAct.utils.sources import sources_from_result  # noqa: E402

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


def _do_search(message, hard_deadline, search_future):
    """执行 search_text:优先取投机检索 future,失败回退同步调用。
    返回 (result, err)。"""
    spec = registry.get("search_text")
    args = {"query": message, "k": C.RAGLITE_SEARCH_K,
            "score_ratio": C.RAGLITE_SEARCH_SCORE_RATIO}
    if search_future is not None:
        budget = C.RAGLITE_MAX_TOTAL_SECONDS
        if hard_deadline:
            budget = max(1.0, hard_deadline - time.time())
        try:
            out = search_future.result(timeout=budget)
            # 投机 future 经 call_with_resilience,返回 (result, err) 元组
            if isinstance(out, tuple) and len(out) == 2:
                return out
            return out, None
        except Exception:  # noqa: BLE001  投机结果不可用则同步重查
            pass
    return call_with_resilience(
        "search_text", args, spec, deadline=hard_deadline, invoke=dispatch)


def _prompt_chunks(result):
    """从 search_text 结果取 (chunk_id, source_stem, page, content, score) 列表。"""
    items = result if isinstance(result, list) else (
        (result or {}).get("results") or (result or {}).get("items") or [])
    out = []
    for it in items:
        if not isinstance(it, dict):
            continue
        out.append({
            "chunk_id": it.get("chunk_id", ""),
            "source_stem": it.get("source_stem", ""),
            "page": it.get("page_num", it.get("page", "")),
            "content": it.get("content", "") or "",
            "score": float(it.get("score") or 0.0),
        })
    return out


def _result_items(result):
    """search_text 原始结果 → 原始条目列表(保留全部字段供 sources_from_result 用)。"""
    return result if isinstance(result, list) else (
        (result or {}).get("results") or (result or {}).get("items") or [])


def _merge_results(result_a, result_b):
    """两次检索原始结果按 chunk_id 去重合并(同块取高分),按分数降序。"""
    best: dict[str, dict] = {}
    for it in list(_result_items(result_a)) + list(_result_items(result_b)):
        if not isinstance(it, dict):
            continue
        cid = str(it.get("chunk_id") or "")
        if not cid:
            continue
        if cid not in best or (float(it.get("score") or 0.0)
                               > float(best[cid].get("score") or 0.0)):
            best[cid] = it
    return sorted(best.values(),
                  key=lambda x: -float(x.get("score") or 0.0))


def _rewrite_query(message: str, prior_turns: list[dict], trace_id: str) -> str:
    """轻模型把口语问题改写成更贴手册原文的检索查询;失败返回空串。"""
    context = "".join(
        f"{t.get('role')}: {str(t.get('content', ''))[:80]}\n"
        for t in prior_turns[-2:] if t.get("role") == "user")
    try:
        resp, err = llm_create_with_retry(
            get_client(), trace_id=trace_id,
            model=C.MODEL_LIGHT,
            messages=[
                {"role": "system", "content":
                    "你是检索查询改写器。把用户问题改写成一句更适合在设备手册中"
                    "全文/语义检索的查询:保留设备型号与专业术语,补齐可能的手册"
                    "用词(如口语'垫脚'→'调整垫脚/调平螺栓'),去掉口语与寒暄。"
                    "只输出改写后的查询本身。"},
                {"role": "user", "content": (context + "问题:" + message)},
            ],
            temperature=0.0, timeout=4.0, max_tokens=60,
            **no_think_extra(),
        )
        if err is not None:
            return ""
        return (resp.choices[0].message.content or "").strip().strip('"“”')[:80]
    except Exception:  # noqa: BLE001
        return ""


def raglite_answer_stream(message: str,
                          history: Optional[list[dict]] = None,
                          *,
                          recorder: TraceRecorder,
                          trace_id: str,
                          t0: float,
                          username: Optional[str] = None,
                          thread_id: Optional[str] = None,
                          hard_deadline: Optional[float] = None,
                          qc_feedback: Optional[str] = None,
                          search_future=None):
    """生成器:执行 raglite 单检索单作答,yield SSE 事件 dict。"""
    history = history or []

    # 首轮多轮上下文以服务端 Redis 短期流水为权威来源(与 simple/react 同口径)。
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

    # ---- 1 次检索(投机 future 或同步)----
    t_search = time.time()
    result, err = _do_search(message, hard_deadline, search_future)
    chunks = _prompt_chunks(result) if err is None else []
    if err is None:
        search_count = 1
        max_score = max((c["score"] for c in chunks), default=0.0)
    else:
        search_count = 0
        max_score = 0.0

    # ---- 自适应重查:首轮顶分低于置信线时,改写查询再检索一次(至多 1 次)----
    # 单步路径没有 react 循环内的换词重查,检索缺失只能兜底拒答;这里以
    # RETRIEVAL_CONFIDENT_SCORE 为触发线,命中失败面(如口语词与手册用词错位)
    # 的题多花 ~1s 换一次检索机会,高置信题零开销。
    if (getattr(C, "RAGLITE_ADAPTIVE_REQUERY", True) and err is None and chunks
            and max_score < C.RETRIEVAL_CONFIDENT_SCORE
            and (hard_deadline is None or time.time() < hard_deadline - 2.0)):
        yield {"type": "status", "message": "首轮命中不理想,换个关键词再检索…",
               "trace_id": trace_id, "step": 1}
        rq = _rewrite_query(message, prior_turns, trace_id)
        if rq and rq != message:
            result2, err2 = _do_search(rq, hard_deadline, None)
            if err2 is None:
                merged = _merge_results(result, result2)
                new_top = max((c["score"] for c in _prompt_chunks(merged)),
                              default=0.0)
                if new_top > max_score + 1e-9:  # 确有更优命中才采纳,防噪声倒退
                    old_top = max_score
                    result = {"results": merged}
                    chunks = _prompt_chunks(merged)
                    max_score = new_top
                    search_count = 2
                    logger.info("raglite requery: %.40r -> %.40r score %.2f->%.2f",
                                message, rq, old_top, new_top)

    # sources 事件在生成前发出:生成失败来源也已透出(评测/前端来源面板依赖)。
    if chunks:
        try:
            items = sources_from_result(result, spec=registry.get("search_text"))
            yield {"type": "sources", "items": items[:C.RAGLITE_SEARCH_K],
                   "trace_id": trace_id, "step": 1}
        except Exception:  # noqa: BLE001  来源提取失败不阻断作答
            pass
    else:
        yield {"type": "status", "message": "未检索到相关资料,尝试直接作答…",
               "trace_id": trace_id, "step": 1}

    # ---- 构建消息 ----
    messages: list[dict[str, Any]] = [{"role": "system",
                                       "content": RAGLITE_SYSTEM_PROMPT}]
    for turn in prior_turns:
        messages.append({"role": turn["role"],
                         "content": turn.get("content", "")})
    user_content = message
    if chunks:
        blocks = []
        for c in chunks[:C.RAGLITE_PROMPT_CHUNKS]:
            head = c["source_stem"] + (f" p{c['page']}" if c["page"] else "")
            blocks.append(f"【{head}】\n{c['content'][:C.RAGLITE_CHUNK_CHARS]}")
        user_content = ("以下是检索到的内部资料:\n\n" + "\n\n".join(blocks)
                        + "\n\n---\n问题:" + message)
    if qc_feedback:
        messages.append({"role": "system",
                         "content": f"【上一轮质检反馈】{qc_feedback}"})
    messages.append({"role": "user", "content": user_content})

    yield {"type": "status", "message": "整理资料作答中…",
           "trace_id": trace_id, "step": 1}

    # ---- 1 次主模型流式作答(无工具绑定,token 即时下发)----
    step_doc = recorder.new_step(1)
    recorder.record_tool(step_doc, tool_call_id="raglite-search-1",
                         name="search_text", args={"query": message},
                         duration_ms=int((time.time() - t_search) * 1000),
                         ok=err is None, category="retrieval")
    t_llm = time.time()
    stream, lerr = llm_create_with_retry(
        get_client(), trace_id=trace_id,
        model=C.TIER_MODEL_REACT,
        messages=messages,
        stream=True, stream_options={"include_usage": True},
        temperature=0.3, timeout=STREAM_TIMEOUT,
        max_tokens=C.RAGLITE_ANSWER_MAX_TOKENS,
        **no_think_extra(),
    )
    if lerr is not None:
        err_doc = recorder.record_error(1, "llm_create", lerr)
        yield {"type": "error_trace", "trace_id": trace_id, **err_doc}
        yield {"type": "error", "trace_id": trace_id, "step": 1,
               "message": f"模型请求失败: {str(lerr)[:160]}"}
        recorder.finish_step(step_doc, "error")
        recorder.final_reason = "error"
        return

    # 总时长看门狗:思考模型流式 reasoning 持续到达会绕过 STREAM_TIMEOUT 空闲超时,
    # 到点强制断流(有部分正文按正常流末走,空正文走下游空答案降级)。
    _wd_cancel, _wd_killed = arm_stream_watchdog(
        stream, float(getattr(C, "REACT_STREAM_DEADLINE_S", 45)))

    content_buf = ""
    usage = None
    finish_reason = None
    ttft_ms = None
    # grounding 门控开启时 token 先缓冲不直发:整段生成完校验置信度,
    # 达标才下发原文,不达标整段替换为人工引导(低置信答案不出后端)。
    do_grounding = (getattr(C, "RAGLITE_GROUNDING_CHECK", True)
                    and getattr(C, "GROUNDING_CHECK", True) and bool(chunks))
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
                    if not do_grounding:
                        yield {"type": "token", "delta": delta.content,
                               "trace_id": trace_id, "step": 1}
            chunk_usage = _extract_usage(chunk)
            if chunk_usage:
                usage = chunk_usage
    except Exception as e:  # noqa: BLE001
        if not _wd_killed.is_set():
            err_doc = recorder.record_error(1, "llm_stream", e)
            yield {"type": "error_trace", "trace_id": trace_id, **err_doc}
            yield {"type": "error", "trace_id": trace_id, "step": 1,
                   "message": f"流读取中断: {str(e)[:160]}"}
            recorder.finish_step(step_doc, "error")
            recorder.final_reason = "error"
            return
        # 看门狗截断:按正常流末处理(部分正文继续走 grounding/下发)
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

    # ---- grounding 置信度门控(与 react 同一实现):达标才下发,否则人工引导 ----
    if do_grounding and content_buf.strip():
        yield {"type": "status", "message": "正在校验答案…",
               "trace_id": trace_id, "step": 1}
        srcs = [{"source_stem": c["source_stem"], "page": c["page"],
                 "content": c["content"]}
                for c in chunks[:C.RAGLITE_PROMPT_CHUNKS]]
        try:
            from agent_reasoning.ReAct.support.grounding import grounding_filter
            content_buf, ginfo = grounding_filter(
                content_buf, srcs, trace_id=trace_id)
        except Exception as e:  # noqa: BLE001  校验崩溃≠低置信,放行原文
            logger.warning("raglite grounding exception, passthrough: %s", e)
        else:
            if ginfo.get("error") == "LLM调用失败":
                yield {"type": "status",
                       "message": "grounding 校验暂不可用(LLM调用失败),已降级放行原文",
                       "trace_id": trace_id, "step": 1}
            elif ginfo.get("action") == "guidance":
                yield {"type": "status",
                       "message": f"答案置信度不足({ginfo.get('confidence')}),已替换为人工核查指引",
                       "trace_id": trace_id, "step": 1}
            elif ginfo.get("removed"):
                yield {"type": "status",
                       "message": f"已按检索资料过滤 {ginfo['removed']} 句无支撑内容",
                       "trace_id": trace_id, "step": 1}
    if content_buf:
        yield {"type": "token", "delta": content_buf,
               "trace_id": trace_id, "step": 1}
    elif not content_buf.strip():
        # 零正文两种成因:①看门狗截断(思考拖过总时限);②正常流末但
        # reasoning 烧光 max_tokens(finish=length,正文 0 token——GLM 对抽象
        # 问题思考可达 2000+ token,实测连非流式重试都会再烧光一次)。救援
        # 必须直连轻模型 doubao(thinking 由 llm 层策略关闭,2-3s 必出答案),
        # 不能再走 GLM——否则救援本身也被思考拖死,空答案收尾。
        yield {"type": "status", "message": "正在重试生成答案…",
               "trace_id": trace_id, "step": 1}
        try:
            resp, rerr = llm_create_with_retry(
                get_client(), trace_id=trace_id, retries=1,
                model=str(getattr(C, "MODEL_LIGHT", "")
                          or "doubao-seed-2.0-lite"),
                messages=messages,
                stream=False, temperature=0.3,
                max_tokens=600,
            )
            if rerr is None:
                content_buf = (resp.choices[0].message.content or "").strip()
        except Exception:  # noqa: BLE001  救援失败按空答案收尾
            content_buf = ""
        if content_buf:
            yield {"type": "token", "delta": content_buf,
                   "trace_id": trace_id, "step": 1}

    recorder.finish_step(step_doc, "answer")
    recorder.final_reason = "answer"

    yield {"type": "assistant_message", "trace_id": trace_id,
           "content": content_buf}

    # 记忆维护闭环:与 simple/react 同一后台管道(按会话串行,不占请求流)。
    # raglite 无 checkpointer,compact_applier 不注入。失败静默,绝不影响应答。
    try:
        from langchain_core.messages import AIMessage, HumanMessage
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
    except Exception:  # noqa: BLE001
        pass

    yield meta_event(
        trace_id, t0, 1, {},
        tokens=recorder.total_tokens, tools_count=search_count,
    )
    yield {"type": "done", "trace_id": trace_id, "trace": recorder.to_dict(),
           "final_reason": recorder.final_reason or "answer",
           "retrieval_max_score": max_score, "search_count": search_count}
