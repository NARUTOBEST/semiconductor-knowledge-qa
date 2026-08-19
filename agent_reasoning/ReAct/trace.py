# -*- coding: utf-8 -*-
"""ReAct 流程结构化轨迹记录器。

负责累积一次 react_stream 请求中的全部中间数据:
  - 每轮 LLM 响应(finish_reason / usage / thought 预览 / 解析后的 tool_calls / 计时)
  - 每次工具调用(参数 / 成功与否 / 耗时 / 结果大小与预览 / 错误)
  - grounding 结果
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
    """单次 ReAct 请求的轨迹累积器。"""

    def __init__(self, trace_id, t0, message, sub_queries=None):
        self.trace_id = trace_id
        self.t0 = t0
        self.message_preview = preview(message, 200)
        self.sub_queries = list(sub_queries) if sub_queries else []
        self.steps = []
        self.total_tokens = {"prompt": 0, "completion": 0, "total": 0}
        self.grounding = None          # {"passed": bool, "warnings": [...]}
        self.errors = []               # 过程中的非致命/致命错误
        self.final_reason = None       # answer / timeout / max_steps / error
        self.plan = None               # {"steps": [...], "question": ...} 或 None
        self.coverage = None           # {"covered":[...], "uncovered_steps":[...], "reason":...} 或 None
        # P&E 嵌套结构(仅 complex 路径填充,其它路径为 None):
        # {"question","planned_steps","planner":{"steps","error","duration_ms"},
        #  "step_results":[{"index","instruction","missing","retried",
        #                   "steps":[...子 TraceRecorder.steps...],
        #                   "tokens":{...},"errors":[...],
        #                   "answer_preview","sources_count","search_count",
        #                   "final_reason","elapsed_ms"}],
        #  "synthesizer":{"answer_preview","usage","error","duration_ms"}|None}
        self.plan_execute = None

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
                    ok=True, duration_ms=None, result=None, error=None):
        """追加一次工具调用记录。"""
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
        }
        step_doc["tools"].append(entry)
        return entry

    # ---- grounding / error ----
    def record_grounding(self, passed, warnings):
        self.grounding = {"passed": bool(passed), "warnings": list(warnings or [])}

    def record_plan(self, steps, question=""):
        """记录本轮检索计划(plan_node 产出,事后 trace 可见)。"""
        self.plan = {"steps": list(steps or []), "question": question or ""}

    def record_coverage(self, coverage):
        """记录计划步骤覆盖判定(coverage tracker 产出,事后 trace 可见)。"""
        self.coverage = coverage

    # ---- P&E 嵌套结构(complex 路径)----
    def begin_plan_execute(self, question, planned_steps):
        """初始化 P&E 嵌套 trace。每个执行步用独立子 TraceRecorder 记录,
        再通过 :meth:`record_pe_step` 折叠进父 trace,实现步骤隔离(Q3)。
        """
        self.plan_execute = {
            "question": question or "",
            "planned_steps": list(planned_steps or []),
            "planner": None,
            "step_results": [],
            "synthesizer": None,
        }
        return self.plan_execute

    def record_pe_planner(self, *, steps=None, error=None, duration_ms=None):
        if self.plan_execute is None:
            return
        self.plan_execute["planner"] = {
            "steps": list(steps or []) if steps is not None else None,
            "error": error,
            "duration_ms": duration_ms,
        }

    def record_pe_step(self, index, instruction, child, *,
                       answer="", sources_count=0, search_count=0,
                       final_reason=None, missing=False, retried=False,
                       elapsed_ms=None):
        """折叠一个 P&E 执行步的子 TraceRecorder 到父 trace,并把 token 用量累加。

        每步用独立子 recorder 记录(步骤间隔离,Q3),完成后把其 steps/errors/tokens
        复制进父 trace 的嵌套结构。
        """
        if self.plan_execute is None:
            return
        if child is not None:
            self.total_tokens["prompt"] += child.total_tokens["prompt"]
            self.total_tokens["completion"] += child.total_tokens["completion"]
            self.total_tokens["total"] += child.total_tokens["total"]
        entry = {
            "index": index,
            "instruction": instruction,
            "missing": bool(missing),
            "retried": bool(retried),
            "elapsed_ms": elapsed_ms,
            "answer_preview": preview(answer),
            "answer_len": len(answer or ""),
            "sources_count": sources_count,
            "search_count": search_count,
            "final_reason": final_reason,
            "tokens": dict(child.total_tokens) if child is not None
                      else {"prompt": 0, "completion": 0, "total": 0},
            "steps": list(child.steps) if child is not None else [],
            "errors": list(child.errors) if child is not None else [],
        }
        self.plan_execute["step_results"].append(entry)
        return entry

    def record_pe_synthesizer(self, *, answer="", usage=None, error=None,
                              duration_ms=None):
        if self.plan_execute is None:
            return
        self.plan_execute["synthesizer"] = {
            "answer_preview": preview(answer),
            "answer_len": len(answer or ""),
            "usage": dict(usage) if usage else None,
            "error": str(error) if error else None,
            "duration_ms": duration_ms,
        }

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
            "sub_queries": self.sub_queries,
            "started_at_offset_ms": 0,
            "total_elapsed_ms": self._elapsed_ms(),
            "steps_count": len(self.steps),
            "final_reason": self.final_reason,
            "total_tokens": dict(self.total_tokens),
            "grounding": self.grounding,
            "plan": self.plan,
            "coverage": self.coverage,
            "plan_execute": self.plan_execute,
            "errors": list(self.errors),
            "steps": self.steps,
        }
