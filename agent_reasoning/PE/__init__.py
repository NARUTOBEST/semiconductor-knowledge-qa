# -*- coding: utf-8 -*-
"""PE(Plan-and-Execute)范式:必经规划 + 逐步隔离 react_loop + synthesizer。

入口 ``run_plan_execute``;planner 失败降级为普通 ReAct(medium)。
经 ReAct.support.runner.run_path 包装,与其它范式共享短期流水、长期升迁与异常兜底。
"""
from .plan_execute import plan_execute_stream
from .runner import run_plan_execute

__all__ = ["run_plan_execute", "plan_execute_stream"]
