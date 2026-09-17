# -*- coding: utf-8 -*-
"""会话(thread_id)按用户隔离的命名空间工具。

背景:LangGraph checkpoint(工作记忆)与短期流水(session_events)都仅以
``thread_id`` 为主键,而 thread_id 由前端客户端可控。若不做用户隔离,登录用户 A
把请求里的 thread_id 改成 B 的会话 id,即可读到 B 的对话历史 / 向其流水写入 /
删除其 checkpoint(IDOR 越权)。会话表(conv.db)虽按 user_id 过滤,但这两套
记忆没有。

做法:凡用 thread_id 触达"工作记忆 checkpoint + 短期流水"的入口,统一改用
``scoped_thread_id(thread_id, username)`` 派生存储键,把用户名并入命名空间。
- 有用户名:``f"{username}|{thread_id}"``,不同用户即使 thread_id 相同也落到不同键;
- 无用户名(异常/未登录兜底):退回原 thread_id,不阻断(fail-open,且未登录本就
  拿不到他人 JWT)。

长期记忆本就按 user_id 隔离,无需改名;delete_thread_artifacts 只在会话表行确属
本人(已校验 user_id)后才被调用,删除时用同一派生函数,键自然对齐。
"""
from __future__ import annotations

from typing import Optional


def scoped_thread_id(thread_id: Optional[str], username: Optional[str]) -> str:
    """返回按用户隔离后的存储用 thread 键。"""
    tid = str(thread_id or "")
    if not username:
        return tid
    return f"{username}|{tid}"
