# -*- coding: utf-8 -*-
"""工具分发器(极简单次路由)。

本模块只做一件事:按 name 从 registry 取 ToolSpec,调用【一次】 handler。
  - 不做重试 / 熔断 / 限流 / 错误分类(这些「机械重试」已上移到
    agent_reasoning.ReAct.support.tool_resilience 韧性中间件);
  - 不做参数校验 / JSON 解析 / schema 检查(这些「决策」在 ReAct 的
    validate_runtime 节点完成;进入这里的 args 已校验、已规范化)。

handler 抛出的异常(httpx 超时/5xx、handler bug 等)由本函数【原样上抛】,
交给韧性中间件捕获分类与重试;中间件是"不抛异常"的边界。
未知/禁用工具返回统一错误 dict(正常已被 validate_generation 拦截,此处仅兜底)。

向后兼容:``dispatch(name, args)`` 两参签名不变,现有节点调用与测试 monkeypatch 继续生效。
"""
from .base import ErrorType, make_error
from .registry import registry


def dispatch(name, args=None, **_ignored):
    """按工具名分发【单次】调用。

    :param name: 工具名(须已注册)
    :param args: 已校验的参数字典(仅含 handler 声明的参数)
    :returns: 工具裸结果;未知/禁用工具返回 ``{"error","error_type","tool"}``;
              handler 异常原样上抛(由韧性中间件处理)。
    """
    args = args or {}
    spec = registry.get(name)
    if spec is None:
        return make_error(
            f"未知工具: {name}(可用: {sorted(registry.names())})",
            ErrorType.FATAL, name)
    if not spec.enabled:
        return make_error(f"工具 {name} 暂不可用", ErrorType.FATAL, name)

    # 单次执行;异常不在此捕获,交由韧性中间件分类/重试。
    return spec.handler(**args)
