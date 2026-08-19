# -*- coding: utf-8 -*-
"""聊天接口路由:POST /api/chat -> SSE 流式对话。"""
import json
import time
import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from chat.service import react_stream
from chat_schemas import ChatRequest
from auth.deps import get_current_user
from support.ratelimit import (
    acquire_user_slot, acquire_global_slot,
    release_user_slot, release_all,
)
from support.metrics import metrics

logger = logging.getLogger("chat")

router = APIRouter()


@router.post("/chat")
async def api_chat(req: ChatRequest, user=Depends(get_current_user)):
    """收前端 {message, history} -> 委托 react_stream -> SSE 流式返回。"""
    username = user["username"]
    logger.info(
        f"chat request from user={username}, "
        f"msg_len={len(req.message)}, history_len={len(req.history)}"
    )

    message = req.message
    history = [{"role": h.role, "content": h.content} for h in req.history]
    thread_id = req.thread_id
    session_id = req.session_id

    def gen():
        """SSE 生成器。

        限流槽位在生成器内获取:若客户端在 handler 返回后、生成器被迭代前
        断开,生成器根本不会执行,槽位也就从未占用,不会泄漏
        (之前在 handler 里获取、只在生成器 finally 释放,存在该缺口)。
        """
        t_start = time.time()
        had_error = False
        acquired = False
        try:
            try:
                acquire_user_slot(username)
                try:
                    acquire_global_slot(username)
                    acquired = True
                except Exception:
                    release_user_slot(username)
                    raise
            except HTTPException as e:
                # 429(同用户并发 / 全局排队超时):以 SSE error 事件告知前端
                had_error = True
                yield f'data: {json.dumps({"type": "error", "message": e.detail}, ensure_ascii=False)}\n\n'
                yield 'data: {"type": "done"}\n\n'
                return

            for ev in react_stream(message, history,
                                   thread_id=thread_id,
                                   username=username,
                                   session_id=session_id):
                if ev.get("type") == "error":
                    had_error = True
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
        except Exception:
            had_error = True
            logger.exception("react_stream 未捕获异常")
            yield 'data: {"type": "error", "message": "内部错误,请重试"}\n\n'
            yield 'data: {"type": "done"}\n\n'
        finally:
            latency_ms = int((time.time() - t_start) * 1000)
            metrics.record_request(username, latency_ms, error=had_error)
            if acquired:
                release_all(username)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
