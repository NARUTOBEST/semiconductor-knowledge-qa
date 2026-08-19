# -*- coding: utf-8 -*-
"""三级范式的各条推理路径(simple / react / plan_execute)。

每条路径都是一个 SSE 事件流生成器,由 support.runner.run_path 统一包裹
(短期流水落库、长期升迁、异常兜底)。
"""
