# -*- coding: utf-8 -*-
"""聊天请求 Pydantic 校验模型。

三道防线:
  1. 字段级:message 长度上限、history 条数上限、role 白名单;
  2. 内容级:每条 history content 截断,防止构造超长上下文;
  3. 结构级:非法 role / 空内容直接 422 -> 400 拒绝。
"""
from pydantic import BaseModel, field_validator

MAX_MESSAGE_LENGTH = 2000      # 单条消息最大字符数(约 1000 汉字)
MAX_HISTORY_ITEMS = 10         # 最多保留 10 条历史(5 轮对话)
MAX_HISTORY_CONTENT = 4000     # 单条历史消息最大字符数(超出截断)


class HistoryItem(BaseModel):
    role: str
    content: str

    @field_validator("role")
    @classmethod
    def validate_role(cls, v):
        if v not in ("user", "assistant"):
            raise ValueError("role 必须是 user 或 assistant")
        return v

    @field_validator("content")
    @classmethod
    def validate_content(cls, v):
        if not v or not v.strip():
            raise ValueError("content 不能为空")
        # 截断超长内容,防止构造超长上下文
        if len(v) > MAX_HISTORY_CONTENT:
            v = v[:MAX_HISTORY_CONTENT]
        return v


class ChatRequest(BaseModel):
    message: str
    history: list[HistoryItem] = []
    # 可选:会话/任务标识,对应 LangGraph checkpoint thread_id。
    # 不传则后端生成随机 uuid(无跨请求续跑能力,但不影响单次问答)。
    thread_id: str | None = None
    session_id: str | None = None

    @field_validator("message")
    @classmethod
    def validate_message(cls, v):
        v = (v or "").strip()
        if not v:
            raise ValueError("消息不能为空")
        if len(v) > MAX_MESSAGE_LENGTH:
            raise ValueError(
                f"消息不能超过 {MAX_MESSAGE_LENGTH} 字符(当前 {len(v)} 字符)"
            )
        return v

    @field_validator("history")
    @classmethod
    def validate_history(cls, v):
        if not v:
            return []
        # 只保留最近 N 条,丢弃更早的
        return v[-MAX_HISTORY_ITEMS:]
