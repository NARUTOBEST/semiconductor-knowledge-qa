# -*- coding: utf-8 -*-
"""LangGraph Agent 的运行入口(同步生成器)。

本模块提供两类东西:
1. 通用包装器 ``run_path()``:把任意一条 SSE 事件流(simple / react)
   包上横切逻辑——短期流水落库、on_event 回调、异常兜底发 error。各路径共用,
   保证流水/异常行为一致。
2. ``run_agent_graph()``:react 路径的具体实现——初始化 TraceRecorder、
   checkpointer,编译图,再把 ``graph.stream`` 交给 run_path。

simple 范式入口在 ``agent_reasoning.simple.run_simple``。
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Callable, Iterable, Optional

from ..trace import TraceRecorder
from memories.storage import working_saver
from memories.storage.thread_scope import scoped_thread_id

from ..core.graph import build_graph
from memories.orchestration import persist_event

logger = logging.getLogger("agent")

# 追踪记录(独立 Redis 键空间,供测试/运维复查工作流与故障)。
# 全程旁路:导入或运行失败都不得影响聊天流,故包在 try 里并提供空实现兜底。
try:
    from trace import record_trace_event as _record_trace_event
except Exception:  # noqa: BLE001
    def _record_trace_event(*_a, **_k):  # type: ignore[no-redef]
        return False


def _trace(ev: dict, *, thread_id: str, username: Optional[str],
           session_id: Optional[str]) -> None:
    """软失败地把事件写入追踪存储;任何异常都吞掉,绝不影响 yield 给前端。"""
    try:
        _record_trace_event(ev, thread_id=thread_id,
                            user_id=username, session_id=session_id)
    except Exception:  # noqa: BLE001
        logger.debug("trace record skipped (ignored)", exc_info=True)


def run_path(path_name: str,
             event_iterable: Iterable[dict],
             *,
             thread_id: str,
             username: Optional[str] = None,
             session_id: Optional[str] = None,
             trace_id: Optional[str] = None,
             on_event: Optional[Callable[[dict], None]] = None,
             on_error: Optional[Callable[[Exception], None]] = None):
    """通用事件流包装器:包裹 simple / react 任意一条事件流。

    :param path_name: 路径名(仅日志/审计用,如 "simple"/"react")。
    :param event_iterable: 被包装的 SSE 事件流(生成器)。
    :param thread_id/username/session_id: 短期流水主键。
    :param trace_id: 异常兜底 error 事件使用的 trace 关联 id。
    :param on_event: 每个事件的额外回调(审计/指标),异常不影响主流。
    :param on_error: 流抛异常时的回调(路径用于置 recorder.final_reason 等)。

    透传事件;异常时补发一条 ``error`` 事件并保证前端能收尾。
    """
    trace_id = trace_id or ""

    try:
        for ev in event_iterable:
            persist_event(ev, thread_id=thread_id,
                          user_id=username, session_id=session_id)
            # 旁路写入追踪存储(tool_call/tool_result/error/error_trace/done);
            # 与 yield 解耦,失败不影响推送浏览器。
            _trace(ev, thread_id=thread_id, username=username,
                   session_id=session_id)
            if on_event is not None:
                try:
                    on_event(ev)
                except Exception:
                    pass
            yield ev
    except Exception as e:
        # 图/路径执行异常:发 error 事件,保证前端能收尾。
        # 安全:不把异常类型/文本外泄(可能含内部路径、SQL、下游地址等),
        # 只回通用提示 + trace_id;详细堆栈仅记服务端日志,凭 trace_id 可查。
        logger.exception("path %s error (trace_id=%s)", path_name, trace_id)
        err = {"type": "error", "trace_id": trace_id,
               "message": "服务暂时不可用,请稍后重试。"
                          + (f"(追踪号 {trace_id})" if trace_id else "")}
        persist_event(err, thread_id=thread_id,
                      user_id=username, session_id=session_id)
        # 兜底 error 同样落追踪存储,保证故障也能复查到。
        _trace(err, thread_id=thread_id, username=username,
               session_id=session_id)
        if on_event is not None:
            try:
                on_event(err)
            except Exception:
                pass
        if on_error is not None:
            try:
                on_error(e)
            except Exception:
                pass
        yield err


def _resolve_pre_search(search_future):
    """把服务层投机检索 future 解析为可序列化的块列表;失败返回 None。"""
    if search_future is None:
        return None
    try:
        out = search_future.result(timeout=15)
        if isinstance(out, tuple) and len(out) == 2:
            result, err = out
            if err is not None:
                return None
            out = result
    except Exception:  # noqa: BLE001  投机结果不可用,react 正常自行检索
        return None
    return out if isinstance(out, list) and out else None


def run_agent_graph(message: str,
                    history: Optional[list[dict]] = None,
                    *,
                    thread_id: str,
                    username: Optional[str] = None,
                    session_id: Optional[str] = None,
                    max_steps: int = 6,
                    max_total_seconds: int = 60,
                    hard_deadline: Optional[float] = None,
                    on_event: Optional[Callable[[dict], None]] = None,
                    qc_feedback: Optional[str] = None,
                    search_future=None):
    """生成器:yield SSE 事件 dict(react ReAct 路径)。

    thread_id: 会话/任务标识(对应 LangGraph checkpoint thread_id)。
    username:  当前用户名(=user_id,记忆按此隔离)。
    qc_feedback: 质检/升级重做轮的系统反馈(注入 user 槽位;普通轮为 None)。
    on_event:  每个事件的额外回调(供上层审计/指标用),异常不影响主流。
    """
    t0 = time.time()
    trace_id = str(uuid.uuid4())[:8]
    recorder = TraceRecorder(trace_id, t0, message)

    # 工作记忆 checkpoint + 短期流水按用户隔离的存储键(防 thread_id IDOR 越权):
    # 不同用户即使传入相同 thread_id 也落到各自命名空间。
    store_thread_id = scoped_thread_id(thread_id, username)

    config: dict[str, Any] = {
        "configurable": {
            "thread_id": store_thread_id,
            "user_id": username,
            "session_id": session_id,
            "trace_recorder": recorder,
            # 后台记忆管道原料快照 holder:emit_done_node 写入,流结束后提交管道
            "mem_snapshot": {},
        }
    }
    inputs: dict[str, Any] = {
        "question": message,
        "history": history or [],
        "started_at": t0,
        "max_steps": max_steps,
        "max_total_seconds": max_total_seconds,
        # 端到端硬截止(跨 service 层升级/重做共享);None 表示仅用 tier 相对预算。
        "hard_deadline": hard_deadline,
        # 质检/升级反馈(重做/升级轮显式注入;普通轮 None)
        "qc_feedback": qc_feedback,
        # 投机检索结果(future 就地解析为可序列化块列表,checkpoint 安全)
        "pre_search": _resolve_pre_search(search_future),
        # 每轮传入新 trace_id,覆盖 checkpoint 残留的旧值,
        # 保证 setup_node 与 recorder/异常兜底事件用同一个 id
        "trace_id": trace_id,
    }

    # 流开始前:落一条用户消息到短期流水(用按用户隔离的存储键)
    try:
        from memories.storage.short import short_term
        short_term.append_event(
            store_thread_id, "user_message", {"content": message},
            user_id=username, session_id=session_id,
        )
    except Exception:
        pass

    with working_saver() as cp:
        graph = build_graph(checkpointer=cp)
        event_iterable = graph.stream(
            inputs, config=config, stream_mode="custom",
        )
        yield from run_path(
            "react", event_iterable,
            thread_id=store_thread_id, username=username, session_id=session_id,
            trace_id=trace_id, on_event=on_event,
            on_error=lambda e: setattr(recorder, "final_reason", "error"),
        )

    # 流结束(done 已发):把本轮记忆原料交给后台记忆管道(非阻塞,不占请求流;
    # 记忆链在 daemon worker 里跑独立记忆图,与主链路完全解耦)。异常静默。
    try:
        from .memory_background import submit_turn_memory
        snap = config["configurable"].get("mem_snapshot") or {}
        submit_turn_memory(
            username=username, store_thread_id=store_thread_id,
            question=message, answer=snap.get("full_reply") or "",
            messages=snap.get("messages") or [],
            final_reason=snap.get("final_reason") or "answer")
    except Exception:  # noqa: BLE001
        pass
