# -*- coding: utf-8 -*-
"""chat 包:Agent 对话核心。

路径设置在此统一执行,子模块无需重复。
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(os.path.dirname(_HERE))
_RAG = os.path.join(_PROJECT, "RAG")
_CONFIG = os.path.join(_PROJECT, "config")
# Qdrant 查询封装(query.py)已迁至检索工具 settings 下;embed.py 仍在 RAG/
_RETRIEVAL_SETTINGS = os.path.join(_PROJECT, "tools", "retrieval", "settings")
# 目录名含空格,不能作为包名,故将目录本身加入 sys.path,以模块名 import
_CONTEXT_MGMT = os.path.join(_PROJECT, "context management")
for _p in (_PROJECT, _RAG, _CONFIG, _RETRIEVAL_SETTINGS, _CONTEXT_MGMT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
