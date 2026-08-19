# -*- coding: utf-8 -*-
"""短期记忆编排层(stream 事件 -> session_events 落库)。"""
from .events import persist_event

__all__ = ["persist_event"]
