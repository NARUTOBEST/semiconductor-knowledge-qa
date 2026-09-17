# -*- coding: utf-8 -*-
"""工具调用统一错误契约(贯穿 生成校验 → runtime 校验 → 执行 三阶段)。

设计:
- 每个失败的工具调用都被归一化成 ``ToolCallError``(stage + kind + message + retryable),
  再由对应节点转成一条 ``ToolMessage``(带正确 tool_call_id)回灌给 LLM,让模型看到
  「为什么这步不行、下一步该怎么改」并自纠,而不是抛异常中断整轮。
- kind 枚举见下方 Kind;stage 枚举见 Stage。
- 机械类错误(timeout/rate_limited/circuit_open/upstream/crash/auth)由韧性中间件
  (tool_resilience)产出;决策类错误(format/unknown_tool/json_parse/schema_violation/
  unknown_arg/blocked/empty_result)由三个决策节点产出。

本模块不含编排/重试逻辑,只是数据结构 + 分类 + 面向 LLM 的文案构造。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import httpx
from langchain_core.messages import ToolMessage


class Stage:
    GENERATION = "generation"   # LLM 生成 tool_calls 阶段(validate_generation 节点)
    RUNTIME = "runtime"         # 运行时解析/参数校验阶段(validate_runtime 节点)
    EXECUTION = "execution"     # 工具执行阶段(韧性中间件 + execute/reflect 节点)


class Kind:
    # —— 决策类(图节点兜底)——
    FORMAT = "format"                    # tool_call 结构坏:缺 id/name、name 非字符串等
    UNKNOWN_TOOL = "unknown_tool"        # 工具名幻觉/不存在/disabled
    JSON_PARSE = "json_parse"            # arguments JSON 解析失败
    SCHEMA_VIOLATION = "schema_violation"  # 参数类型/enum/必填/超长不符
    UNKNOWN_ARG = "unknown_arg"          # 传了 handler 未声明的参数(已忽略)
    BLOCKED = "blocked"                  # 安全策略/guard 拦截
    EMPTY_RESULT = "empty_result"        # 执行成功但无结果(决策:换关键词)
    # —— 机械类(韧性中间件)——
    TIMEOUT = "timeout"                  # 网络/读取超时
    RATE_LIMITED = "rate_limited"        # 429 限流(退避重试后仍失败)
    CIRCUIT_OPEN = "circuit_open"        # 熔断打开,中间件短路未调用
    UPSTREAM = "upstream"                # 5xx / 传输故障(重试后仍失败)
    CRASH = "crash"                      # handler 内部异常(重试后仍失败)
    AUTH = "auth"                        # 401/403 鉴权失败(不重试)
    BUDGET_TIMEOUT = "budget_timeout"    # 端到端硬预算耗尽,future 有界等待超时


# 机械类 kind 集合(由中间件产出,reflect 据此判定"服务持久不可用")
MECHANICAL_KINDS = frozenset({
    Kind.TIMEOUT, Kind.RATE_LIMITED, Kind.CIRCUIT_OPEN,
    Kind.UPSTREAM, Kind.CRASH, Kind.AUTH, Kind.BUDGET_TIMEOUT,
})

# 计入熔断失败的机械 kind(auth/fatal 不代表可重试的服务故障,但仍标记服务不可用)
BREAKER_FAILURE_KINDS = frozenset({
    Kind.TIMEOUT, Kind.RATE_LIMITED, Kind.UPSTREAM, Kind.CRASH,
})


@dataclass
class ToolCallError:
    """单个工具调用的归一化错误。"""
    stage: str
    kind: str
    message: str
    retryable: bool = False
    tool: Optional[str] = None
    # 机械类附加信息
    retries: int = 0          # 中间件已重试次数
    recovered: bool = False   # 重试后是否恢复(恢复则不应作为错误)

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": self.message,
            "error_type": self.kind,
            "tool": self.tool,
            "stage": self.stage,
            "retryable": self.retryable,
            "retries": self.retries,
        }


def classify_exception(exc: BaseException) -> tuple[str, bool]:
    """把执行期异常分类为 (kind, retryable)。机械类,供中间件使用。

    - httpx 超时                -> timeout,      可重试
    - httpx HTTP 429            -> rate_limited, 可重试
    - httpx HTTP 5xx            -> upstream,     可重试
    - httpx HTTP 401/403        -> auth,         不可重试
    - httpx 其它 4xx            -> upstream(非重试)/ 这里归 crash 之外的不可重试
    - httpx 传输故障(连接拒绝等) -> upstream,     可重试
    - 其它未预期异常(handler bug)-> crash,       可重试一次
    """
    if isinstance(exc, httpx.TimeoutException):
        return Kind.TIMEOUT, True
    if isinstance(exc, httpx.HTTPStatusError):
        code = getattr(getattr(exc, "response", None), "status_code", 0) or 0
        if code == 429:
            return Kind.RATE_LIMITED, True
        if code in (401, 403):
            return Kind.AUTH, False
        if code >= 500:
            return Kind.UPSTREAM, True
        # 其它 4xx:参数/请求问题,handler 本应在 runtime 校验拦住;落到这里不重试
        return Kind.UPSTREAM, False
    if isinstance(exc, httpx.TransportError):
        return Kind.UPSTREAM, True
    return Kind.CRASH, True


def error_tool_message(tcid: str, err: ToolCallError) -> ToolMessage:
    """把归一化错误包成回灌给 LLM 的 ToolMessage(带 tool_call_id)。

    内容面向模型:说明错因 + 修正指引,引导下一步自纠。
    """
    content = _LLM_GUIDANCE.get(err.kind, _DEFAULT_GUIDANCE)(err)
    return ToolMessage(content=content, tool_call_id=tcid)


# ----------------- 面向 LLM 的纠错文案 -----------------
def _g_unknown_tool(e: ToolCallError) -> str:
    return (f"工具调用失败:不存在名为「{e.tool}」的工具。{e.message} "
            f"请只从可用工具列表中选择,并确保工具名拼写完全一致后重新调用;"
            f"若无需工具即可作答,也可直接给出答案。")


def _g_format(e: ToolCallError) -> str:
    return (f"工具调用格式错误:{e.message} "
            f"请按 function-calling 规范重新发出完整的工具调用(含正确的工具名与参数)。")


def _g_json_parse(e: ToolCallError) -> str:
    return (f"参数 JSON 解析失败:{e.message} "
            f"请检查 arguments 是否为合法 JSON(引号、逗号、括号配对),修正后重新调用该工具。")


def _g_schema(e: ToolCallError) -> str:
    return (f"参数不符合工具 schema:{e.message} "
            f"请按该工具 parameters 定义修正参数类型/取值/必填项后重新调用。")


def _g_unknown_arg(e: ToolCallError) -> str:
    return (f"提示:{e.message} 该参数已被忽略。请仅使用工具声明的参数重新调用。")


def _g_blocked(e: ToolCallError) -> str:
    return (f"工具调用被安全策略拦截:{e.message} "
            f"请不要尝试绕过;改用合规的参数或其他途径完成。")


def _g_timeout(e: ToolCallError) -> str:
    return (f"工具「{e.tool}」响应超时(已重试 {e.retries} 次仍未成功)。"
            f"请稍后减少条件/换更具体的关键词重试一次;若持续超时,请基于已有信息作答"
            f"并说明本地知识库暂时不可用。")


def _g_rate_limited(e: ToolCallError) -> str:
    return (f"工具「{e.tool}」触发限流(429,已退避重试 {e.retries} 次)。"
            f"请降低调用频率、稍后重试;若持续,基于已有信息谨慎作答。")


def _g_circuit_open(e: ToolCallError) -> str:
    return (f"工具「{e.tool}」所在的检索服务暂时不可用(熔断中),已停止重复调用。"
            f"请不要再调用该工具;基于通用知识谨慎作答,并明确告知用户本地知识库暂时不可用、"
            f"答案未经内部资料核实。")


def _g_upstream(e: ToolCallError) -> str:
    return (f"工具「{e.tool}」调用下游服务失败(已重试 {e.retries} 次):{e.message}。"
            f"可换关键词重试一次;若仍失败,按「内部资料未覆盖/服务不可用」如实说明。")


def _g_crash(e: ToolCallError) -> str:
    return (f"工具「{e.tool}」内部执行异常(已重试 {e.retries} 次):{e.message}。"
            f"请不要重复相同调用;可调整参数重试一次,或基于已有信息作答。")


def _g_auth(e: ToolCallError) -> str:
    return (f"工具「{e.tool}」鉴权失败(401/403),无法访问该服务。"
            f"请不要重试该工具;基于通用知识谨慎作答并说明内部资料暂不可用。")


def _g_budget(e: ToolCallError) -> str:
    return (f"工具「{e.tool}」等待超过本次响应的总时限,已跳过。"
            f"请基于已获取的资料直接作答,不要再次长时间等待该工具。")


def _g_empty(e: ToolCallError) -> str:
    return (f"工具「{e.tool}」执行成功但未返回任何结果。{e.message} "
            f"若这是设备型号/报警代码/操作步骤类问题,请换用设备型号、报警代码、工序别名、"
            f"故障现象等关键词再检索一次;若仍无结果,按「内部资料未覆盖」如实说明。")


_DEFAULT_GUIDANCE = lambda e: (  # noqa: E731
    f"工具「{e.tool}」调用失败:{e.message}。请据此修正后重试,或改用其他方式作答。")

_LLM_GUIDANCE = {
    Kind.UNKNOWN_TOOL: _g_unknown_tool,
    Kind.FORMAT: _g_format,
    Kind.JSON_PARSE: _g_json_parse,
    Kind.SCHEMA_VIOLATION: _g_schema,
    Kind.UNKNOWN_ARG: _g_unknown_arg,
    Kind.BLOCKED: _g_blocked,
    Kind.TIMEOUT: _g_timeout,
    Kind.RATE_LIMITED: _g_rate_limited,
    Kind.CIRCUIT_OPEN: _g_circuit_open,
    Kind.UPSTREAM: _g_upstream,
    Kind.CRASH: _g_crash,
    Kind.AUTH: _g_auth,
    Kind.BUDGET_TIMEOUT: _g_budget,
    Kind.EMPTY_RESULT: _g_empty,
}
