# -*- coding: utf-8 -*-
"""Agent 记忆系统。

storage/ — 存储访问层(PG/Redis 读写、升迁、召回网关),按工作/短期/长期细分。

LangGraph 编排层(State/节点/边/图编译)已迁至 server/chat/react/。

导入约定:从项目根以包方式导入,例如
  from memories.storage.short import short_term
  from memories.storage.long import recall_memories
"""
