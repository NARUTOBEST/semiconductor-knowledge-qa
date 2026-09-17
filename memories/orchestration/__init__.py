# -*- coding: utf-8 -*-
"""记忆系统编排层(记忆与 LangGraph 图的装配/生命周期),按层分目录。

  - working/ : checkpoint 线程生命周期(删除级联 + 30 天滚动 TTL)
  - short/   : custom-stream 事件 -> 短期 session_events 白名单落库
  - long/    : 长期偏好(PG):后台抽取、对话前召回注入、注销级联

LangGraph 图本身(State/节点/边/图编译)在 agent_reasoning/;存储访问在 memories/storage/。
依赖方向:agent_reasoning -> memories.orchestration -> memories.storage,不反向。
"""
from .short import persist_event, recent_dialogue_block
from .working import (
    delete_thread_artifacts,
    delete_user_artifacts,
    prune_inactive,
    start_prune_daemon,
)

# 长期记忆是旁路增强:导入失败(缺驱动等)不应影响短期/工作记忆与主流程。
try:
    from .long import (
        schedule_extraction,
        recall_memories,
        format_memory_block,
        memory_tool_schema,
        MEMORY_TOOL_NAME,
        delete_user_long_term,
    )
except Exception:  # noqa: BLE001
    schedule_extraction = lambda *a, **k: None  # type: ignore
    recall_memories = lambda *a, **k: ([], None)  # type: ignore
    format_memory_block = lambda *a, **k: ""  # type: ignore
    memory_tool_schema = lambda *a, **k: {}  # type: ignore
    MEMORY_TOOL_NAME = "recall_memory"  # type: ignore
    delete_user_long_term = lambda *a, **k: 0  # type: ignore

__all__ = [
    "persist_event",
    "recent_dialogue_block",
    "delete_thread_artifacts",
    "delete_user_artifacts",
    "prune_inactive",
    "start_prune_daemon",
    "schedule_extraction",
    "recall_memories",
    "format_memory_block",
    "memory_tool_schema",
    "MEMORY_TOOL_NAME",
    "delete_user_long_term",
]
