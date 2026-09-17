# -*- coding: utf-8 -*-
"""ReAct 流程结构化轨迹记录器。

负责累积一次请求中的全部中间数据:
  - 每轮 LLM 响应(finish_reason / usage / thought 预览 / 解析后的 tool_calls / 计时)
  - 每次工具调用(参数 / 成功与否 / 耗时 / 结果大小与预览 / 错误)
  - 错误与异常(含 traceback 预览)

内容粒度:元数据 + 截断预览(默认 500 字),避免 trace 体积爆炸。
记录器与生成器机制解耦,仅做纯数据累积,便于单测。
"""
import json
import sys
import time
import traceback

PREVIEW_LIMIT = 500          # 文本/结果预览上限
TRACEBACK_PREVIEW_LIMIT = 1000  # traceback 预览上限


def preview(value, limit=PREVIEW_LIMIT):
    """把任意值转成截断字符串预览。

    dict/list -> JSON 序列化后截断;str 直接截断;其余 repr 后截断。
    超长尾部追加省略号。
    """
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    elif isinstance(value, (dict, list, tuple)):
        try:
            text = json.dumps(value, ensure_ascii=False, default=str)
        except Exception:
            text = repr(value)
    else:
        text = repr(value)
    if len(text) > limit:
        return text[:limit] + "…"
    return text


class TraceRecorder:
    """单次请求的轨迹累积器(simple / react 共用)。"""

    def __init__(self, trace_id, t0, message):
        self.trace_id = trace_id
        self.t0 = t0
        self.message_preview = preview(message, 200)
        self.steps = []
        self.total_tokens = {"prompt": 0, "completion": 0, "total": 0}
        self.errors = []               # 过程中的非致命/致命错误
        self.final_reason = None       # answer / timeout / max_steps / error

    def _elapsed_ms(self):
        return int((time.time() - self.t0) * 1000)

    # ---- step ----
    def new_step(self, step):
        """开新一轮,返回该轮 dict(同时 append 到 steps)。"""
        doc = {
            "step": step,
            "started_at_offset_ms": self._elapsed_ms(),
            "llm": None,
            "tools": [],
            "decision": None,
            "new_sources_count": 0,
        }
        self.steps.append(doc)
        return doc

    def finish_step(self, step_doc, decision, new_sources_count=0):
        step_doc["decision"] = decision
        step_doc["elapsed_ms"] = self._elapsed_ms() - step_doc["started_at_offset_ms"]
        step_doc["new_sources_count"] = new_sources_count

    # ---- llm ----
    def record_llm(self, step_doc, *, finish_reason=None, usage=None,
                   thought="", tool_calls=None, stream_duration_ms=None,
                   time_to_first_token_ms=None):
        """填充一轮的 LLM 响应元数据。

        tool_calls: 已解析的 [{"id","name","args"(对象/None),"args_len",
                     "args_parse_error"}]
        """
        tool_calls = tool_calls or []
        step_doc["llm"] = {
            "finish_reason": finish_reason,
            "usage": dict(usage) if usage else None,
            "thought_preview": preview(thought),
            "thought_len": len(thought or ""),
            "has_tool_calls": bool(tool_calls),
            "tool_calls": tool_calls,
            "stream_duration_ms": stream_duration_ms,
            "time_to_first_token_ms": time_to_first_token_ms,
        }
        if usage:
            self.total_tokens["prompt"] += int(usage.get("prompt_tokens", 0) or 0)
            self.total_tokens["completion"] += int(usage.get("completion_tokens", 0) or 0)
            self.total_tokens["total"] += int(usage.get("total_tokens", 0) or 0)

    # ---- tool ----
    def record_tool(self, step_doc, *, tool_call_id, name, args, args_parse_error=None,
                    ok=True, duration_ms=None, result=None, error=None,
                    category=None, error_type=None, cache_hit=False):
        """追加一次工具调用记录。

          category   - 工具类别(检索三件套均为 "retrieval")
          error_type - 失败分类(timeout/retryable/fatal/circuit_open/...)
          cache_hit  - 是否命中结果缓存
        """
        try:
            result_size = len(json.dumps(result, ensure_ascii=False, default=str))
        except Exception:
            result_size = len(repr(result))
        entry = {
            "tool_call_id": tool_call_id,
            "name": name,
            "args": args if isinstance(args, (dict, list)) else None,
            "args_preview": preview(args, 60),
            "args_parse_error": args_parse_error,
            "ok": ok,
            "duration_ms": duration_ms,
            "result_size": result_size,
            "result_preview": preview(result),
            "error": error,
            "category": category,
            "error_type": error_type,
            "cache_hit": bool(cache_hit),
        }
        step_doc["tools"].append(entry)
        return entry

    def record_error(self, step, phase, exc):
        """记录异常(带 traceback 预览)。phase 见 service.py。"""
        # 仅当当前处于 except 上下文中时才取 traceback,
        # 否则(如 LLM create 把异常作为返回值回传)format_exc() 只会得到 "NoneType"。
        tb_preview = ""
        if exc and sys.exc_info()[0] is not None:
            tb_preview = traceback.format_exc()[-TRACEBACK_PREVIEW_LIMIT:]
        entry = {
            "step": step,
            "phase": phase,
            "error_type": type(exc).__name__ if exc else None,
            "message": str(exc)[:300] if exc else "",
            "traceback_preview": tb_preview,
        }
        self.errors.append(entry)
        return entry

    # ---- 输出 ----
    def to_dict(self):
        return {
            "trace_id": self.trace_id,
            "message_preview": self.message_preview,
            "started_at_offset_ms": 0,
            "total_elapsed_ms": self._elapsed_ms(),
            "steps_count": len(self.steps),
            "final_reason": self.final_reason,
            "total_tokens": dict(self.total_tokens),
            "errors": list(self.errors),
            "steps": self.steps,
        }
