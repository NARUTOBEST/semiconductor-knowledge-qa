# -*- coding: utf-8 -*-
"""短期记忆【读取/注入】:从会话事件流水取近期问答,格式化为上下文文本。

与 events.py(把 stream 事件落库)相对,本模块负责把已落库的近期对话读回,
供 recall_memory 专用节点在模型按需召回时作为【服务端权威的近期上下文】返回,
保证会话连贯性(不再依赖前端重发 history)。

只取 user_message / assistant_message 两类事件(跳过 tool_call/tool_result 等
过程事件),按时间正序拼接。旁路:Redis 不可用 / 无流水时返回 ""(调用方降级)。
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from memories.storage.short import short_term
from memories.storage.connections import redis_ready_fast

logger = logging.getLogger("agent")

# 短期对话注入条数上限(user+assistant 合计)。默认 None = 注入【本会话全部】短期
# 对话(在 TTL 留存范围内);仅当显式设置 SHORT_MEM_RECALL_LIMIT 为正整数时限量。
_env_limit = os.getenv("SHORT_MEM_RECALL_LIMIT", "").strip()
_DEFAULT_LIMIT: Optional[int] = int(_env_limit) if _env_limit.isdigit() and int(_env_limit) > 0 else None


def recent_high_watermark(thread_id: Optional[str]) -> int:
    """返回该 thread 短期流水当前高水位 seq(摘要/compact 落盘时记录,实现游标互斥)。

    Redis 不可用/无流水返回 0。
    """
    if not thread_id or not redis_ready_fast():
        return 0
    try:
        return int(short_term.current_seq(thread_id) or 0)
    except Exception as e:  # noqa: BLE001
        logger.info("recent_high_watermark skipped: %s: %s",
                    type(e).__name__, str(e)[:120])
        return 0


def recent_dialogue_block(thread_id: Optional[str], limit: Optional[int] = None,
                          *, current_question: Optional[str] = None,
                          since_seq: int = 0) -> str:
    """返回当前 thread 短期问答拼成的上下文文本;无内容返回 ""。

    :param thread_id: 已按用户隔离的存储键(= config.configurable.thread_id,
                      即 scoped_thread_id 的结果),与写入流水时用的键一致。
    :param limit: 最多返回的消息条数(user+assistant 合计);``None``(默认)表示
                      返回本会话【全部】短期对话,传正整数才只取最近 N 条。
    :param current_question: 本轮当前问题。流水末尾通常正是它(图运行前刚写入),
                      而它已作为 HumanMessage 在上下文中,故剔除与之相同的末尾
                      user 消息,避免当前问题重复出现;只回传更早的历史轮次。
    :param since_seq: 游标高水位(Req1)。>0 时只取 seq 大于该值的事件——游标之前
                      的对话已被会话摘要覆盖,严格互斥、不重复注入。默认 0=不设下界。
    """
    if not thread_id:
        return ""
    # 快探:Redis 不可用时亚秒级返回(失败后冷却),绝不拖慢主流程。
    if not redis_ready_fast():
        return ""
    n = limit if (limit and limit > 0) else _DEFAULT_LIMIT  # None=全部
    try:
        turns = short_term.recent_dialogue(
            thread_id, limit=n, since_seq=max(0, int(since_seq or 0)))
    except Exception as e:  # noqa: BLE001  旁路:Redis 缺失/异常都降级为空
        logger.info("recent_dialogue_block skipped: %s: %s",
                    type(e).__name__, str(e)[:120])
        return ""
    if not turns:
        return ""

    # 剔除末尾与当前问题相同的 user 消息(当前问题已在 HumanMessage 里,无需重复)
    cq = (current_question or "").strip()
    if cq:
        while turns and turns[-1].get("role") == "user" \
                and (turns[-1].get("content") or "").strip() == cq:
            turns = turns[:-1]
    if not turns:
        return ""

    lines = []
    for t in turns:
        role = "用户" if t.get("role") == "user" else "助手"
        content = (t.get("content") or "").strip()
        if content:
            lines.append(f"{role}:{content}")
    if not lines:
        return ""

    return (
        "【会话历史对话】以下是本次会话在此之前的【全部】问答记录,用于保持上下文连贯"
        "(请据此理解指代与承接关系,不要逐句复述这些历史):\n"
        + "\n".join(lines)
    )


def recent_dialogue_messages(thread_id: Optional[str],
                             current_question: Optional[str] = None,
                             limit: Optional[int] = 10) -> list[dict]:
    """从短期流水取近期问答,返回结构化 [{"role","content"}](冷启动种子 messages 用)。

    与 recent_dialogue_block(返回拼好的【文本块】并入 system)相对:本函数返回
    【结构化多轮消息】,供首轮(checkpointer 无 messages)直接铺成 user/assistant
    多轮历史,使服务端 Redis 短期流水成为首轮上下文的【权威来源】,不再依赖前端
    重发 history。

    - 剔除末尾与本轮 current_question 相同的 user 消息(图运行前已预写进流水,
      而它随后会作为本轮 HumanMessage 出现,避免重复);
    - Redis 不可用 / 无流水返回 [](调用方据此降级/兜底)。
    """
    if not thread_id or not redis_ready_fast():
        return []
    n = limit if (limit and limit > 0) else None
    try:
        turns = short_term.recent_dialogue(thread_id, limit=n)
    except Exception as e:  # noqa: BLE001  旁路:Redis 缺失/异常都降级为空
        logger.info("recent_dialogue_messages skipped: %s: %s",
                    type(e).__name__, str(e)[:120])
        return []
    if not turns:
        return []
    cq = (current_question or "").strip()
    if cq:
        while turns and turns[-1].get("role") == "user" \
                and (turns[-1].get("content") or "").strip() == cq:
            turns = turns[:-1]
    return [{"role": t.get("role"), "content": t.get("content")}
            for t in turns if t.get("role") in ("user", "assistant")
            and (t.get("content") or "").strip()]
