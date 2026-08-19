# -*- coding: utf-8 -*-
"""工具层包:三个检索工具 + search_tools schema 注册表 + dispatch 分发器。

目录结构:
  _common.py        共享:stage2 路径、Q/C 导入、_pid、_text_dict、_image_dict
  search_tools.py   检索工具集合(search_text / search_image / get_chunk)
                    + OpenAI function-calling schema 注册表(search_tools)
  dispatch.py       按 (name,args) 分发执行的分发器

对外接口(主循环无需改):
    from tools import search_tools, dispatch, search_text, search_image, get_chunk
"""
from . import _common  # noqa: F401  副作用:最早执行,把 stage2 加入 sys.path
from .search_tools import (
    search_tools, SEARCH_TOOLS_BY_NAME,
    search_text, search_image, get_chunk,
)
from .dispatch import dispatch

__all__ = ["search_tools", "SEARCH_TOOLS_BY_NAME", "dispatch",
           "search_text", "search_image", "get_chunk"]