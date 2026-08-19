# -*- coding: utf-8 -*-
"""断点续跑验证用的最小 LangGraph 图(非生产编排代码)。

图结构: START -> weather_tool -> END
- weather_tool 节点每次执行都会把"工具被调用次数"写进 SIDE_EFFECT_FILE(跨进程证据),
  并写入一条 AI 消息 + usage token 统计。
- 编译时挂 working_saver()(PostgresSaver),状态按 thread_id 落 PG。
"""
import json
import os
import sys

# 项目根 + server 入 path(AgentState 现位于 agent_reasoning/ReAct/core/state.py)
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SERVER = os.path.join(_ROOT, "server")
for _p in (_ROOT, _SERVER):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph

from agent_reasoning.ReAct.core.state import AgentState
from memories.storage import working_saver

# 跨进程的"工具调用次数"证据文件
SIDE_EFFECT_FILE = os.path.join(os.path.dirname(__file__), ".weather_tool_calls.json")


def _read_calls() -> int:
    if not os.path.exists(SIDE_EFFECT_FILE):
        return 0
    with open(SIDE_EFFECT_FILE, "r", encoding="utf-8") as f:
        return int(json.load(f).get("calls", 0))


def _bump_calls() -> int:
    n = _read_calls() + 1
    with open(SIDE_EFFECT_FILE, "w", encoding="utf-8") as f:
        json.dump({"calls": n}, f)
    return n


def reset_calls() -> None:
    with open(SIDE_EFFECT_FILE, "w", encoding="utf-8") as f:
        json.dump({"calls": 0}, f)


def weather_tool(state: AgentState) -> dict:
    """模拟"调用天气查询外部工具",带可观测副作用。"""
    n = _bump_calls()
    question = state.get("question") or ""
    if not question:
        for m in state.get("messages", []):
            q = getattr(m, "content", None)
            if q:
                question = q
                break
    reply = f"[第{n}次调用工具] 北京天气:晴,25℃。(已收到问题:{question})"
    return {
        "messages": [AIMessage(content=reply)],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        "step": 1,
        "full_reply": reply,
    }


def build_graph():
    """编译图并返回 (graph, saver_cm)。

    saver_cm 必须在 graph 使用期间保持打开(其 __enter__ 返回 PostgresSaver),
    用完由调用方 close。这样连接池不会在 compile 后被立即收回。
    """
    builder = StateGraph(AgentState)
    builder.add_node("weather_tool", weather_tool)
    builder.add_edge(START, "weather_tool")
    builder.add_edge("weather_tool", END)

    cm = working_saver()
    cp = cm.__enter__()
    graph = builder.compile(checkpointer=cp)
    return graph, cm


if __name__ == "__main__":
    g, cm = build_graph()
    try:
        print("graph built; checkpointer =", "PostgresSaver (working_db)")
    finally:
        cm.__exit__(None, None, None)
