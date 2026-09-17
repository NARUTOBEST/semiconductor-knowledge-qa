# -*- coding: utf-8 -*-
"""长期记忆生命周期:账号注销级联清理。

被 memories.orchestration.working.lifecycle.delete_user_artifacts(username) 调用,
与工作记忆 checkpoint / 短期流水一并清除该用户全部数据。失败记日志、不抛异常。
"""
from __future__ import annotations

import json
import logging

from ...storage.long import long_term

logger = logging.getLogger("agent")


def delete_user_long_term(username: str) -> int:
    """删除某用户的全部长期记忆(画像 + 分片偏好)。返回删除条目数。"""
    if not username:
        return 0
    try:
        n = long_term.delete_user(username)
        logger.info(json.dumps({
            "event": "cleanup_long_term", "username": username, "deleted": n,
        }, ensure_ascii=False))
        return n
    except Exception as e:  # noqa: BLE001
        logger.warning("delete_user_long_term failed (ignored): %s: %s",
                       type(e).__name__, str(e)[:160])
        return 0
