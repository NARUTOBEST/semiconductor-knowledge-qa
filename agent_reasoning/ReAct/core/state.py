# -*- coding: utf-8 -*-
"""Agent Graph 的 State 定义。

字段来源是 server/chat/service.py:react_stream 中实际流转的数据,
并补齐记忆系统所需字段。全部字段必须 JSON 可序列化(会被 PostgresSaver 写入 checkpoint)。

reducer 说明:
  - messages          : add_messages,支持多节点追加/按 id 更新
  - usage             : 累加合并(多步 token 不被覆盖)
  - collected_sources : 按 chunk_id 合并,高分覆盖低分(复刻原 score 比较逻辑)
  - 其余标量          : 默认"后者覆盖",由节点返回完整值

注意:LangGraph checkpoint 的 thread_id 取自 config["configurable"]["thread_id"],
不是 State;State 里保留 thread_id 仅为节点读取方便。真正的会话续跑靠 configurable。
"""
from __future__ import annotations

from typing import Any, Optional, TypedDict, Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


# ---------------------------------------------------------------- reducers
def _add_usage(left: Optional[dict[str, int]],
               right: Optional[dict[str, int]]) -> dict[str, int]:
    """多步 token usage 累加,不被后一步覆盖。"""
    if not left:
        return dict(right or {})
    if not right:
        return dict(left)
    out = dict(left)
    for k, v in right.items():
        out[k] = out.get(k, 0) + (v or 0)
    return out


def _merge_sources(left: Optional[dict[str, dict[str, Any]]],
                   right: Optional[dict[str, dict[str, Any]]]
                   ) -> dict[str, dict[str, Any]]:
    """合并知识库来源;同 chunk_id 保留 score 更高者(复刻原 collected_sources 逻辑)。

    哨兵约定:right 含 "__reset__" 键表示显式清空(setup_node 每轮重置跨轮残留;
    普通节点返回空 dict 只表示"本步无新增",不清空已积累的来源)。
    """
    if right and "__reset__" in right:
        return {}
    if not left:
        return dict(right or {})
    if not right:
        return dict(left)
    merged = dict(left)
    for key, src in right.items():
        old = merged.get(key)
        if old is None or float(src.get("score", 0.0)) > float(old.get("score", 0.0)):
            merged[key] = src
    return merged


class AgentState(TypedDict, total=False):
    """一次对话 thread 在图中流转的状态。

    输入字段(调用方传入):
      thread_id / user_id / session_id : 记忆与审计主键
      question                         : 本轮用户原始问题
      history                          : 前端传入的最近对话 [{role,content}]
      max_steps / max_total_seconds    : 推理步数与总耗时上限
      started_at                       : 请求开始时间戳(time.time()),用于超时判断,
                                         进 State 以保证断点续跑后超时仍正确

    运行期字段(节点逐步填充):
      trace_id    : 短 uuid,日志/事件关联
      sub_queries : query 改写产物
      recalled_memories : 召回网关返回的长期记忆(注入 system message + 审计)
      summary     : 跨轮旧消息的增量摘要(会话级工作记忆压缩,见 summarize.py)
      messages    : LLM 对话消息,add_messages
      step        : 当前步数
      full_reply  : 累积的最终回答文本(token 流拼接)
      collected_sources : {chunk_key: source_dict},合并 reducer
      retrieval_down    : 检索工具是否失败
      usage             : 累积 token usage,累加 reducer
      final_reason      : answer/tool_calls/max_steps/timeout/error
      grounding         : {passed, warnings} 或 None
      error             : 终端错误 {phase, message} 或 None
      reflect_count     : 已反思重生成的次数(每轮对话内,setup 重置)
      reflect_feedback  : 上一次反思的修正意见(重生成时注入 messages)
      task_plan         : 复杂问题检索计划 {need_plan: bool, steps: [str]}
                          (plan_node 产出,build_messages 拼进 system message;
                          setup 每轮重置,不跨轮复用)
      search_count      : 本轮已执行检索工具调用次数(粗粒度计划进度信号,
                          tools_node 累加,build_messages 展示"已检索 N 次")
      coverage_rollbacks: 因计划某步未被检索资料覆盖而回退重检索的次数(有界,
                          coverage_check_node 累加,setup 每轮重置)
    """

    # ---- 输入主键 ----
    thread_id: str
    user_id: Optional[str]
    session_id: Optional[str]
    question: str
    history: list[dict[str, Any]]
    max_steps: int
    max_total_seconds: int
    started_at: float

    # ---- 运行期 ----
    trace_id: str
    sub_queries: list[str]
    recalled_memories: list[dict[str, Any]]
    summary: str
    messages: Annotated[list[BaseMessage], add_messages]
    step: int
    full_reply: str
    collected_sources: Annotated[dict[str, dict[str, Any]], _merge_sources]
    retrieval_down: bool
    usage: Annotated[dict[str, int], _add_usage]
    # tool_call_id -> 参数解析错误文本(每轮 agent 写入,tools 读取后随 tool_call 事件暴露)
    tool_parse_errors: dict[str, str]
    final_reason: Optional[str]
    grounding: Optional[dict[str, Any]]
    error: Optional[dict[str, str]]
    reflect_count: int
    reflect_feedback: str
    task_plan: dict[str, Any]
    search_count: int
    coverage_rollbacks: int


# 运行时对象通过 config["configurable"] 传递,不进 State(不可 JSON 序列化):
#   trace_recorder   : TraceRecorder 实例
#   coverage_tracker : CoverageTracker 实例(异步维护计划步骤覆盖文档,见 plan_grounding.py)
#   writer           : get_stream_writer() 发出的 SSE 事件流
#   on_event         : 后端二次消费回调(落短期记忆/转发监控)
#   client           : OpenAI 客户端单例
# 节点内用 get_stream_writer() 取 writer,用 RunnableConfig 取 configurable。
