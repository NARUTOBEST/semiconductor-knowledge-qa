# -*- coding: utf-8 -*-
"""认证模块:JWT + SQLite 用户管理。

路径设置在导入时执行,确保 config 可被裸 import。
init_db() 在包导入时自动创建 users 表(幂等)。
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))            # server/auth
_PROJECT = os.path.dirname(os.path.dirname(_HERE))            # project root
_CONFIG = os.path.join(_PROJECT, "config")
if _CONFIG not in sys.path:
    sys.path.insert(0, _CONFIG)

from .db import init_db  # noqa: E402
init_db()
