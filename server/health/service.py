# -*- coding: utf-8 -*-
"""健康检查业务逻辑:探测关键依赖是否可用。"""


def check_qdrant():
    """检查 Qdrant 向量库是否可用。"""
    try:
        from tools._common import Q
        Q.get_client().get_collections()
        return "ok"
    except Exception as e:
        return "error: {}".format(str(e)[:80])


def check_llm():
    """检查 LLM 客户端是否可初始化(不发实际推理请求)。"""
    try:
        from chat.service import get_client
        get_client()
        return "ok"
    except Exception as e:
        return "error: {}".format(str(e)[:80])
