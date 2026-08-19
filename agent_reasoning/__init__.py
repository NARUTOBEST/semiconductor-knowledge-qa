# -*- coding: utf-8 -*-
"""agent_reasoning 包:Agent 推理核心(独立于 server 的顶层包)。

此处统一做 sys.path 引导,使 ReAct 包不依赖 server/chat/__init__.py
即可独立导入。子包无需重复设置。
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))        # agent_reasoning/
_PROJECT = os.path.dirname(_HERE)                          # C:/project3
for _p in (
    _PROJECT,
    os.path.join(_PROJECT, "config"),
    os.path.join(_PROJECT, "RAG"),
    os.path.join(_PROJECT, "context management"),
    os.path.join(_PROJECT, "server"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
