# -*- coding: utf-8 -*-
"""Agent 记忆系统(两层:working checkpoint + short 流水)。

storage/ — 存储访问层(PG/Redis 读写),按 working/short 细分。

LangGraph 编排层(State/节点/边/图编译)在 agent_reasoning/。

导入约定:从项目根以包方式导入,例如
  from memories.storage.short import short_term
  from memories.storage import working_saver
"""
