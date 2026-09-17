# -*- coding: utf-8 -*-
"""上下文管理(Context Management)—— 控制发给 LLM 的 messages 体积。

与 Working Memory 的职责边界:
  - Working Memory(memories/Working memory/):存储每轮工作数据、按需拿取,不裁剪消息
  - 本模块:纯函数,决定"一条工具结果以多长的形式进 messages",不存储、不碰 messages 结构

当前仅实现一项优化:
  truncate_tool_result(result, max_chars, spec=None)
    —— 工具结果过长则按合理范围截断,保留 chunk_id/source_stem/score 等短元数据,
       长文本字段截到 max_chars 并提示 LLM 可用 get_chunk 取全文。
    spec 为 ToolSpec 时按其声明的 TruncatePolicy 截断;None 时用默认 policy
    (与历史硬编码字段集合一致,保证旧调用行为不变)。

后续的轮次淘汰、token 预算裁剪等上下文管理策略也放在本模块,不污染存储层。
"""
import json

# 默认截断策略(与历史硬编码字段集合一致;spec=None 时使用,行为不变)
DEFAULT_LONG_FIELDS = {"content", "table_html", "description"}
DEFAULT_SHORT_FIELDS = {
    "chunk_id", "source_stem", "source_path", "source_path_orig",
    "page_num", "page_start", "page_end", "page",
    "score", "heading_path", "caption", "content_type", "image_path",
    "has_table", "heading", "char_count", "chunk_index", "item_type",
    "parent_text_chunk_id",
}
DEFAULT_LIST_LONG_FIELDS = {"image_descriptions", "image_paths"}

# 统一错误 dict 允许的键(用于识别 error 结果,原样透传给 LLM)
_ERROR_KEYS = frozenset({"error", "error_type", "tool"})


def _is_error_result(result) -> bool:
    """识别 dispatch 统一错误结构。

    - 含 "error" 键;
    - 且所有键都属于错误结构白名单(避免把恰好含 error 字段的业务数据误判)。
    支持历史的单键 {"error": ...} 与新的三键统一错误。
    """
    return (isinstance(result, dict)
            and "error" in result
            and set(result.keys()).issubset(_ERROR_KEYS))


def truncate_tool_result(result, max_chars=800, *, spec=None):
    """返回截断后的 JSON 字符串,用于塞进 role=tool 消息发给 LLM。

    - {"error": ...} 原样返回(LLM 需要看到错误)
    - list[chunk] / 单个 chunk:深拷贝后截断长文本字段,保留短元数据
    - 未识别的长字符串字段兜底截断

    :param spec: 可选 ToolSpec,按其 truncate 策略截断;None 用默认策略。
    """
    if _is_error_result(result):
        return json.dumps(result, ensure_ascii=False)

    if spec is not None and getattr(spec, "truncate", None) is not None:
        policy = spec.truncate
        # 纯文本结果(如 recall_memory 记忆块):原样截断返回,保留换行、不走 JSON 转义。
        if getattr(policy, "passthrough_text", False) and isinstance(result, str):
            effective_max = policy.max_chars_override or max_chars
            return result if len(result) <= effective_max else result[:effective_max] + "…"
        if policy.custom_fn is not None:
            truncated = policy.custom_fn(result, max_chars)
            return json.dumps(truncated, ensure_ascii=False, default=str)
        long_fields = policy.long_fields or set()
        short_fields = policy.short_fields or set()
        list_long_fields = policy.list_long_fields or set()
        effective_max = policy.max_chars_override or max_chars
    else:
        long_fields = DEFAULT_LONG_FIELDS
        short_fields = DEFAULT_SHORT_FIELDS
        list_long_fields = DEFAULT_LIST_LONG_FIELDS
        effective_max = max_chars

    truncated = _truncate_value(
        result, effective_max, long_fields, short_fields, list_long_fields)
    return json.dumps(truncated, ensure_ascii=False, default=str)


def _truncate_value(value, max_chars, long_fields, short_fields, list_long_fields):
    if isinstance(value, list):
        return [_truncate_value(v, max_chars, long_fields, short_fields, list_long_fields)
                for v in value]
    if isinstance(value, dict):
        return {k: _truncate_field(k, v, max_chars, long_fields, short_fields,
                                   list_long_fields)
                for k, v in value.items()}
    if isinstance(value, str) and len(value) > max_chars:
        return value[:max_chars] + "…[已截断]"
    return value


def _truncate_field(key, value, max_chars, long_fields, short_fields, list_long_fields):
    if key in long_fields and isinstance(value, str):
        if len(value) <= max_chars:
            return value
        return value[:max_chars] + "…[已截断,完整内容请用 get_chunk 获取]"
    if key in list_long_fields and isinstance(value, list):
        joined = "\n".join(str(x) for x in value)
        if len(joined) > max_chars:
            return joined[:max_chars] + "…[已截断]"
        return value
    if isinstance(value, (dict, list)):
        return _truncate_value(value, max_chars, long_fields, short_fields,
                               list_long_fields)
    if (isinstance(value, str) and key not in short_fields
            and len(value) > max_chars):
        return value[:max_chars] + "…[已截断]"
    return value
