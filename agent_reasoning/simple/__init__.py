# -*- coding: utf-8 -*-
"""simple 范式:单轮直答(无工具,lite 模型)。

适用闲聊/寒暄/元问题等明确不需要半导体领域检索的请求。入口 ``run_simple``
经 ReAct.support.runner.run_path 包装,与其它范式共享短期流水、长期升迁与异常兜底。
"""
from .stream import simple_answer_stream
from .runner import run_simple

__all__ = ["run_simple", "simple_answer_stream"]
