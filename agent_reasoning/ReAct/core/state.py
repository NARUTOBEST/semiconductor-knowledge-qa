# -*- coding: utf-8 -*-
"""Agent Graph 的 State 定义。

字段来源是 server/chat/service.py:react_stream 中实际流转的数据,
并补齐记忆系统所需字段。全部字段必须 JSON 可序列化(会被 RedisSaver 写入 checkpoint)。

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


def _merge_tool_status(left: Optional[dict[str, str]],
                       right: Optional[dict[str, str]]) -> dict[str, str]:
    """合并各工具类别的可用性状态(category -> "up"/"down")。

    按 key 合并,后写覆盖前写;"down" 是粘性的(一旦某类别标记 down,
    本轮不会被后续 "up" 覆盖,除非 setup 用 "__reset__" 显式复位)。
    哨兵:right 含 "__reset__" 表示 setup 每轮复位(其值即复位后的默认 dict)。
    """
    if right and "__reset__" in right:
        return {k: v for k, v in right.items() if k != "__reset__"}
    if not left:
        return dict(right or {})
    if not right:
        return dict(left)
    merged = dict(left)
    for cat, status in right.items():
        # down 粘性:已经是 down 则不被 up 覆盖
        if merged.get(cat) == "down" and status == "up":
            continue
        merged[cat] = status
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
      summary     : 跨轮旧消息的增量摘要(会话级工作记忆压缩,见 summarize.py)
      messages    : LLM 对话消息,add_messages
      step        : 当前步数
      full_reply  : 累积的最终回答文本(token 流拼接)
      collected_sources : {chunk_key: source_dict},合并 reducer
      tool_status       : {category: "up"/"down"},按工具类别的故障隔离标记,
                          合并 reducer(down 粘性,setup 每轮复位)
      usage             : 累积 token usage,累加 reducer
      final_reason      : answer/max_steps/timeout/error
      error             : 终端错误 {phase, message} 或 None
      search_count      : 本轮已执行检索工具调用次数(tools_node 累加)
      retrieval_max_score : 本轮检索到的最高 rerank 相关分(tools_node 从
                          collected_sources 计算,标量覆写;setup 复位为 0.0)。
                          用于低置信自适应检索与 react 质检门判定。
      bind_tools        : 是否给 LLM 绑定检索工具 schema。True(默认 react)传 tools
                          走 ReAct;False(simple 直答)不传 tools,模型只作答。
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
    hard_deadline: Optional[float]   # 端到端硬截止墙钟时间戳(service 层跨升级/重做统一下发)
    # 质检/升级反馈(service 层显式下发,重做/升级那轮注入 user 槽位):此前经
    # history 夹带,但冷启动种子以 Redis 短期流水为权威时会忽略 history,反馈丢失。
    qc_feedback: str

    # ---- 运行期 ----
    trace_id: str
    summary: str
    messages: Annotated[list[BaseMessage], add_messages]
    step: int
    full_reply: str
    collected_sources: Annotated[dict[str, dict[str, Any]], _merge_sources]
    tool_status: Annotated[dict[str, str], _merge_tool_status]
    usage: Annotated[dict[str, int], _add_usage]
    # tool_call_id -> 参数解析错误文本(每轮 agent 写入,validate_runtime 读取回灌)
    tool_parse_errors: dict[str, str]
    # 本轮待执行的合法工具调用(validate_generation→validate_runtime→execute 节点间透传):
    # [{"id","name","args"}]。每轮由校验节点重建,普通标量覆写 reducer。
    pending_tool_calls: list[dict[str, Any]]
    # execute 节点产出的每个调用结果(供 reflect 决策研判):
    # [{"name","category","ok","kind","empty",...}],reflect 消费后清空。
    tool_outcomes: list[dict[str, Any]]
    # 工具名 -> 本次请求内连续失败次数(reflect 累加,达阈值摘工具);setup 复位。
    tool_fail_streak: dict[str, int]
    # 工具名 -> 本次请求内"空结果/低置信换词提示"累计次数(Req8):reflect 累加,
    # 达 REACT_TOOL_REQUERY_MAX 后不再为该工具注入换词 hint(改为"内部资料未覆盖"口径);
    # setup 复位。普通标量覆写(reflect 每轮读全量、回写全量)。
    tool_requery_count: dict[str, int]
    # 记忆链内部 LLM(升迁门/摘要)token 用量,与作答 usage【分账】(Req12):累加 reducer,
    # 仅观测/metrics 用,绝不混入对外 billable 的 usage / meta / done。
    usage_internal: Annotated[dict[str, int], _add_usage]
    final_reason: Optional[str]
    error: Optional[dict[str, str]]
    search_count: int
    # 服务层投机检索结果(原始问题的 search_text 块列表,
    # runner 已把 future 解析为可序列化结果;None=无)。
    # react 首轮注入为预检索上下文,模型可直接引用(免一次
    # 检索步);来源同步并入 collected_sources。
    pre_search: Optional[list]
    retrieval_max_score: float
    bind_tools: bool

    # 注:记忆维护已迁出主图(后台记忆管道,见 memories/orchestration/memory_loop),
    # state 不再承载记忆路由字段;每轮记忆原料快照经 config["configurable"]["mem_snapshot"]
    # 传递(运行时对象不进 state)。


# 运行时对象通过 config["configurable"] 传递,不进 State(不可 JSON 序列化):
#   trace_recorder   : TraceRecorder 实例
#   writer           : get_stream_writer() 发出的 SSE 事件流
#   on_event         : 后端二次消费回调(落短期记忆/转发监控)
#   client           : OpenAI 客户端单例
# 节点内用 get_stream_writer() 取 writer,用 RunnableConfig 取 configurable。
