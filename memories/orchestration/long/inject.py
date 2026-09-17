# -*- coding: utf-8 -*-
"""长期记忆【按需召回】:模型调用 recall_memory 工具时取回用户偏好,回灌为 ToolMessage。

- recall_memories(username, question):语义召回相关偏好(embed 问题 -> pgvector 余弦近邻)
  + 稳定画像 profile;任一依赖不可用都降级(embed 失败则只给 profile;PG 不可用则空)。
- format_memory_block(profile, relevant):拼成一段【用户长期记忆】文本,无内容返回 ""。

记忆【不】在 build_messages_node 每轮注入;唯一拿取入口是 recall_memory 专用图节点
(validate_nodes.recall_memory_node),它把本长期块与短期近期对话块合并成一条 ToolMessage。
记忆是旁路增强:出错/无记忆不影响主流程。
"""
from __future__ import annotations

import logging

import config as C

from ...storage.long import long_term, embed_one

logger = logging.getLogger("agent")

# 记忆召回的工具名:模型以 function-calling(tool_choice="auto")自主决定是否调用;
# 但它【不注册进 tools registry】,而由 ReAct 子图的专用节点 recall_memory_node 拦截执行
# (见 agent_reasoning/ReAct/core/{loop,validate_nodes}.py),不走通用 execute_tools/熔断管线。
MEMORY_TOOL_NAME = "recall_memory"

# 模型不可用时给模型的降级文案(仍作为 ToolMessage 回灌,保证 tool_call 有配对结果)。
_MEMORY_EMPTY_TEXT = "(当前没有可用的历史偏好信息,可按一般情况作答)"


def memory_tool_schema() -> dict:
    """recall_memory 的 OpenAI function-calling schema(独立于 tools registry)。

    由 agent_node 在【已登录且开启长期记忆】时追加到下发的 tools 中,模型据此 auto 决策。
    """
    return {
        "type": "function",
        "function": {
            "name": MEMORY_TOOL_NAME,
            "description": (
                "按需召回【记忆上下文】,一次返回两部分:"
                "(1) 本次会话此前的近期对话,用于理解指代、省略与上下文连贯;"
                "(2) 当前登录用户的长期记忆:跨会话记住的身份角色、专业背景、关注领域、"
                "语言/语气/格式偏好与长期约束。"
                "当问题依赖上文(出现'它/那个/上面说的'等指代、承接前文)或需要结合该用户个人背景、"
                "偏好、历史关注点时调用(例如'按我的情况/像往常一样/我之前关注的/接着上面说'),"
                "用当前问题或其核心诉求作为 query。"
                "纯半导体文档/术语检索请用 search_text/search_image;与上下文和用户背景都无关的"
                "独立新问题、闲聊寒暄不要调用本工具,以免冗余。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "要召回的记忆主题,通常即用户当前的问题或其核心诉求。",
                    },
                },
                "required": ["query"],
            },
        },
    }


def recall_memories(username: str, question: str):
    """返回 (relevant: list[dict], profile: dict|None)。不可用时返回 ([], None)。"""
    if not getattr(C, "LONG_MEM_ENABLED", True) or not username:
        return [], None
    profile = long_term.get_profile(username)
    qvec = embed_one(question or "")
    relevant = long_term.search_relevant(
        username, qvec, k=getattr(C, "LONG_MEM_TOP_K", 5)) if qvec else []
    return relevant, profile


def format_memory_block(relevant, profile) -> str:
    """把召回的偏好拼成 system 块;无任何内容返回 ""。"""
    lines: list[str] = []

    # 稳定画像:一句话总结 + 关注领域
    if profile:
        summary = (profile.get("summary") or "").strip()
        interests = profile.get("top_interests") or []
        prefs = profile.get("display_prefs") or {}
        if summary:
            lines.append("用户画像:" + summary)
        if interests:
            lines.append("关注领域:" + "; ".join(str(x) for x in interests[:6]))
        if prefs:
            pref_txt = "; ".join(str(v) for v in prefs.values() if v)
            if pref_txt:
                lines.append("稳定偏好:" + pref_txt)

    # 与本轮问题语义相关的偏好事实
    if relevant:
        # 距离越近越相关;去重后取内容
        seen = set()
        rel_txt = []
        for h in sorted(relevant, key=lambda x: x.get("distance", 1.0)):
            c = (h.get("content") or "").strip()
            if c and c not in seen:
                seen.add(c)
                rel_txt.append(c)
        if rel_txt:
            lines.append("与本问题相关的已知偏好/背景:" + "; ".join(rel_txt[:6]))

    if not lines:
        return ""
    return (
        "【用户长期记忆】以下是跨会话记住的该用户信息,用于个性化作答"
        "(若与当前问题无关可忽略;不要向用户复述你在读取记忆):\n"
        + "\n".join("- " + ln for ln in lines)
    )
