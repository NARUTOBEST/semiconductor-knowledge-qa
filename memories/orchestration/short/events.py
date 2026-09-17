# -*- coding: utf-8 -*-
"""把 LangGraph custom-stream 事件持久化到短期 session_events。

属于记忆编排层(memories/orchestration),被 react/runner 在消费层(节点之外)调用:
一边 yield SSE 一边写库,避免 LangGraph 重放未完成节点导致节点内重复写库。

短期记忆【只存对话问答】(user_message / assistant_message),读回喂 LLM 做对话
上下文;工具调用与故障/收尾事件(tool_call/tool_result/error/error_trace/done)
已迁到独立追踪存储(顶层 `trace` 包,Redis `trace:*` 键),供测试/运维事后复查,
不再混入短期记忆。逐 token 流不入库(量太大、无意义)。
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from memories.storage.short import short_term

logger = logging.getLogger("agent")

# 写入短期流水的事件类型白名单:短期记忆只记对话问答,喂模型做上下文。
# 工作流/故障事件(tool_call/tool_result/error/error_trace/done)归 trace 包追踪存储。
_PERSIST_TYPES = {
    "user_message",       # runner 在流开始前补写
    "assistant_message",  # finalize 发出的完整回复
}


def persist_event(ev: dict[str, Any], *,
                  thread_id: str,
                  user_id: Optional[str] = None,
                  session_id: Optional[str] = None) -> bool:
    """把一个 SSE 事件写入短期流水。

    返回 True 表示已落库;False 表示不在白名单被跳过。
    任何异常都被吞掉并打印——记忆写入失败不得影响主聊天流程。
    """
    etype = ev.get("type")
    if etype not in _PERSIST_TYPES:
        return False
    # payload 去掉可能很大的 trace(审计流水里 trace 另有 done 事件承载,这里精简)
    payload = {k: v for k, v in ev.items() if k != "trace"}
    try:
        short_term.append_event(
            thread_id, etype, payload,
            user_id=user_id, session_id=session_id,
        )
        return True
    except Exception as e:
        logger.warning("persist_event append_event failed (ignored): %s: %s",
                       type(e).__name__, e)
        return False
