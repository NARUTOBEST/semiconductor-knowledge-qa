# -*- coding: utf-8 -*-
"""工具注册中心:name -> ToolSpec,支持按 category 查询、schema 聚合、enabled 过滤。

新增工具只需 ``registry.register(ToolSpec(...))``,dispatch 与 nodes 即自动识别,
无需修改核心逻辑。
"""
from __future__ import annotations

import logging
from typing import Optional

from .base import ToolSpec

logger = logging.getLogger("tools")


class Registry:
    """工具注册中心(进程内单例,非线程安全——注册发生在 import 期)。"""

    def __init__(self):
        self._specs: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        """注册一个工具。重复同名注册记录 warning 并覆盖(开发期友好)。"""
        if spec.name in self._specs:
            logger.warning("工具 %s 重复注册,覆盖旧声明", spec.name)
        self._specs[spec.name] = spec
        if not spec.enabled:
            logger.info("工具 %s 已注册但 disabled(不会 bind 给 LLM)", spec.name)

    def get(self, name: str) -> Optional[ToolSpec]:
        return self._specs.get(name)

    def require(self, name: str) -> ToolSpec:
        spec = self._specs.get(name)
        if spec is None:
            raise KeyError(f"未知工具: {name}(可用: {sorted(self._specs)})")
        return spec

    def by_category(self, category: str, enabled_only: bool = True) -> list[ToolSpec]:
        return [s for s in self._specs.values()
                if s.category == category and (not enabled_only or s.enabled)]

    def all(self, enabled_only: bool = True) -> list[ToolSpec]:
        return [s for s in self._specs.values() if not enabled_only or s.enabled]

    def schemas(self, enabled_only: bool = True) -> list[dict]:
        """聚合导出全部(启用)工具的 OpenAI function-calling schema,供 LLM bind_tools。"""
        return [s.to_schema() for s in self.all(enabled_only=enabled_only)]

    def names(self, enabled_only: bool = True) -> set[str]:
        return {s.name for s in self.all(enabled_only=enabled_only)}

    def names_that_produce_sources(self, enabled_only: bool = True) -> set[str]:
        """产生来源(结果进 collected_sources 引用卡片)的工具名集合。"""
        return {s.name for s in self.all(enabled_only=enabled_only) if s.produces_sources}

    def reset(self) -> None:
        """仅供测试清空注册表。"""
        self._specs.clear()


# 模块级单例
registry = Registry()
