# -*- coding: utf-8 -*-
"""工具分发器:按 (name, args) 路由到对应工具函数并执行。

工具函数通过 HTTP 调用检索微服务,超时由 httpx 管理(关连接,无僵尸线程)。
dispatch 只做路由 + 参数过滤 + 错误捕获,不再起子线程。
"""
import inspect
import logging

from .search_tools import search_text, search_image, get_chunk

logger = logging.getLogger("dispatch")

_HANDLERS = {
    "search_text": search_text,
    "search_image": search_image,
    "get_chunk": get_chunk,
}


def dispatch(name, args=None, timeout=None):
    """按工具名分发调用。

    工具函数内部通过 httpx 调用检索微服务,自带 30s 超时。
    超时 = httpx 关闭 HTTP 连接 = 干干净净,无僵尸线程。

    返回:
      - 工具原样结果(list/dict/None);
      - 出错时返回 {"error": "..."},不抛异常。
    """
    fn = _HANDLERS.get(name)
    if fn is None:
        return {"error": f"未知工具: {name}(可用: {sorted(_HANDLERS)})"}
    args = args or {}
    try:
        params = inspect.signature(fn).parameters
        accepted = {k: v for k, v in args.items() if k in params}
        return fn(**accepted)
    except Exception as e:
        if isinstance(e, TypeError):
            return {"error": f"参数错误({name}): {e}"}
        return {"error": f"{name} 执行失败: {type(e).__name__}: {e}"}
