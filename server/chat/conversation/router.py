# -*- coding: utf-8 -*-
"""会话 CRUD API:GET 列表 / PUT 同步 / DELETE 删除。

所有操作都需认证,且只能操作自己的会话(user_id 校验)。
"""
import logging
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from auth.deps import get_current_user
from chat.conversation.db import upsert_conversation, get_conversations, delete_conversation

logger = logging.getLogger("conversation")
router = APIRouter()


class ConversationItem(BaseModel):
    id: str
    title: str
    messages: list  # list[dict], 结构由前端定义
    createdAt: int
    updatedAt: int


@router.get("")
def list_conversations(user=Depends(get_current_user)):
    """获取当前用户的全部会话(含消息)。"""
    return get_conversations(user["username"])


@router.put("")
def sync_conversations(
    convs: list[ConversationItem],
    user=Depends(get_current_user),
):
    """批量同步会话(upsert)。前端定期调用,把本地变更推到服务端。"""
    username = user["username"]
    for conv in convs:
        upsert_conversation(
            conv.id, username, conv.title,
            conv.messages, conv.createdAt, conv.updatedAt,
        )
    logger.info(f"synced {len(convs)} conversations for user={username}")
    return {"ok": True, "synced": len(convs)}


@router.delete("/{conv_id}")
def remove_conversation(conv_id: str, user=Depends(get_current_user)):
    """删除一条会话(校验 user_id 防越权)。

    会话 id 即 LangGraph thread_id,删除时级联清理工作记忆 checkpoint 与
    短期流水(长期记忆不按单会话删除)。清理失败不阻断 SQLite 行删除。
    仅当行确属本人并被真正删除时才级联清理,防止他人凭会话 id
    清掉别人的 checkpoint(SQLite 行删除虽被 user_id 挡住,但
    delete_thread_artifacts 只按 thread_id 工作)。
    """
    deleted = delete_conversation(conv_id, user["username"])
    if deleted:
        try:
            from memories.orchestration import delete_thread_artifacts
            delete_thread_artifacts(conv_id)
        except Exception:
            logger.exception(f"cleanup artifacts failed for conversation {conv_id}")
        logger.info(f"deleted conversation {conv_id} for user={user['username']}")
    else:
        logger.warning(
            f"delete conversation {conv_id} denied for user={user['username']}(非本人或不存在)"
        )
    return {"ok": True}
