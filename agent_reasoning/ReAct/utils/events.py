# -*- coding: utf-8 -*-
"""日志与元事件工具。"""
import json
import time
import logging

logger = logging.getLogger("agent")


def log_done(trace_id, t0, step, reason=""):
    """记录请求结束日志。"""
    logger.info(json.dumps({
        "trace_id": trace_id,
        "event": "done",
        "elapsed_ms": int((time.time() - t0) * 1000),
        "steps": step,
        "reason": reason,
    }, ensure_ascii=False))


def meta_event(trace_id, t0, step, collected_sources,
               tokens=None, tools_count=0):
    """生成成本/性能元数据事件。

    tokens: {"prompt":..,"completion":..,"total":..} 或 None
    """
    return {
        "type": "meta",
        "trace_id": trace_id,
        "elapsed_ms": int((time.time() - t0) * 1000),
        "steps": step,
        "sources_count": len(collected_sources),
        "tools_count": tools_count,
        "tokens": tokens,
    }
