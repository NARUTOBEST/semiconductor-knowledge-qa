# -*- coding: utf-8 -*-
"""聊天接口路由:POST /api/chat -> SSE 流式对话。"""
import asyncio
import json
import queue
import threading
import time
import logging

from fastapi import APIRouter, Depends, HTTPException, Request
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

# 断连轮询间隔(秒):sync SSE 生成器无法直接 await is_disconnected,
# 由事件循环里的 watcher 轮询并置位 threading.Event 通知工作线程。
_DISCONNECT_POLL_S = 0.5
# SSE 心跳间隔(秒):超时无事件即下发 ": ping" 注释行,防代理/客户端空闲切断
_HEARTBEAT_S = 5.0


@router.post("/chat")
async def api_chat(req: ChatRequest, request: Request,
                   user=Depends(get_current_user)):
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

    # 客户端断连信号:事件循环轮询 request.is_disconnected(),置位后 sync 生成器
    # 在事件边界停止迭代并关闭内部图生成器(及时释放限流槽、停掉后续 LLM/工具调用),
    # 避免用户已离开仍把整条请求跑完的无效成本。
    cancel_event = threading.Event()

    async def _watch_disconnect():
        try:
            while not cancel_event.is_set():
                if await request.is_disconnected():
                    cancel_event.set()
                    logger.info("client disconnected, cancel stream user=%s", username)
                    return
                await asyncio.sleep(_DISCONNECT_POLL_S)
        except Exception:
            pass

    watcher = asyncio.create_task(_watch_disconnect())

    def gen():
        """SSE 生成器。

        限流槽位在生成器内获取:若客户端在 handler 返回后、生成器被迭代前
        断开,生成器根本不会执行,槽位也就从未占用,不会泄漏
        (之前在 handler 里获取、只在生成器 finally 释放,存在该缺口)。
        """
        t_start = time.time()
        had_error = False
        acquired = False
        user_token = None         # 限流槽位 token(Redis 模式获取;内存回退为 None)
        global_token = None
        saw_done = False          # 是否已向客户端发过终端 done 事件
        cancelled = False         # 是否因客户端断连而中断
        # 按范式分维度记账所需状态(阶段 8.4)
        cur_tier = None            # 当前正在产出的 tier
        final_tier = None          # 最终产出答案的 tier
        escalated = False
        try:
            try:
                user_token = acquire_user_slot(username)
                try:
                    global_token = acquire_global_slot(username)
                    acquired = True
                except Exception:
                    release_user_slot(username, user_token)
                    raise
            except HTTPException as e:
                # 429(同用户并发 / 全局排队超时):以 SSE error 事件告知前端
                had_error = True
                yield f'data: {json.dumps({"type": "error", "message": e.detail}, ensure_ascii=False)}\n\n'
                yield 'data: {"type": "done"}\n\n'
                return

            stream = react_stream(message, history,
                                  thread_id=thread_id,
                                  username=username,
                                  session_id=session_id)
            try:
                for ev in stream:
                    # 断连:在事件边界停止迭代并关闭内部图生成器(触发其 finally,
                    # 释放限流槽/关闭 tracker),不再跑后续 LLM 与工具。
                    if cancel_event.is_set():
                        cancelled = True
                        break
                    etype = ev.get("type")
                    if etype == "error":
                        had_error = True
                    elif etype == "tier":
                        cur_tier = ev.get("tier")
                        final_tier = cur_tier
                    elif etype == "escalation":
                        escalated = True
                        metrics.record_escalation(ev.get("from_tier", ""),
                                                  ev.get("to_tier", ""))
                    elif etype == "done":
                        saw_done = True
                    yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
            finally:
                # 正常结束也停掉断连 watcher;断连 break 时额外关闭内部生成器。
                cancel_event.set()
                if hasattr(stream, "close"):
                    try:
                        stream.close()
                    except Exception:
                        pass
            # 兜底:done 是 SSE 契约的唯一终端事件。流正常走完(未断连、无异常)
            # 但没发过 done 时补一条,保证前端加载态一定收尾,不依赖连接关闭。
            # (断连/异常路径不补:断连客户端已不在;异常路径 except 分支已发 done。)
            if not cancelled and not saw_done:
                yield 'data: {"type": "done"}\n\n'
        except Exception:
            had_error = True
            logger.exception("react_stream 未捕获异常")
            yield 'data: {"type": "error", "message": "内部错误,请重试"}\n\n'
            yield 'data: {"type": "done"}\n\n'
        finally:
            latency_ms = int((time.time() - t_start) * 1000)
            metrics.record_request(username, latency_ms, error=had_error)
            if final_tier:
                metrics.record_tier_result(
                    final_tier, latency_ms, error=had_error,
                    escalated=escalated,
                )
            if acquired:
                release_all(username, user_token=user_token,
                            global_token=global_token)

    # SSE 心跳:GLM 等思考模型静默思考期间(流式缓冲/reasoning 不产生下发字节),
    # 连接可长达 50s+ 无任何字节,Next.js 代理 ~53s 空闲即切断、部分客户端也按
    # 空闲超时断开。生产者线程消费 gen(),消费者带超时取队列,空闲时下发
    # SSE 注释行 ": ping"(所有 EventSource/行解析客户端天然忽略)。
    def _heartbeat_gen():
        _q: queue.Queue = queue.Queue()
        _SENTINEL = object()

        def _produce():
            try:
                for chunk in gen():
                    _q.put(chunk)
            except BaseException as e:  # noqa: BLE001  原样转交消费者抛出
                _q.put(e)
            finally:
                _q.put(_SENTINEL)

        threading.Thread(target=_produce, daemon=True).start()
        while True:
            try:
                item = _q.get(timeout=_HEARTBEAT_S)
            except queue.Empty:
                yield ": ping\n\n"
                continue
            if item is _SENTINEL:
                break
            if isinstance(item, BaseException):
                raise item
            yield item

    return StreamingResponse(
        _heartbeat_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
