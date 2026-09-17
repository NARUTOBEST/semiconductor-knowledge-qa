# -*- coding: utf-8 -*-
"""构建 LLM 输入消息列表(system + history + user)。

属于上下文管理(Context management):拼装发给 LLM 的 messages 结构。
与 system_prompt 同处一个 sys.path 目录(目录名含空格,以顶级模块名 import)。
"""
import logging

from system_prompt import SYSTEM_PROMPT

logger = logging.getLogger("message")

MAX_CONTEXT_CHARS = 50000  # 总上下文字符上限(约 25k token)


def build_messages(message, history=None):
    """构建 LLM 输入消息列表。

    上下文结构:
      [system]  角色定义 + 检索工具说明
      [user/assistant]  最近若干轮原文(总量不超过 MAX_CONTEXT_CHARS)
      [user]  当前问题
    """
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]

    total = len(SYSTEM_PROMPT) + len(message)
    if history:
        for h in history[-10:]:
            role, cont = h.get("role"), h.get("content")
            if role not in ("user", "assistant") or not cont:
                continue
            if total + len(cont) > MAX_CONTEXT_CHARS:
                logger.debug(f"上下文超长({total + len(cont)} > {MAX_CONTEXT_CHARS}),截断历史")
                break
            total += len(cont)
            msgs.append({"role": role, "content": cont})

    msgs.append({"role": "user", "content": message})
    return msgs
