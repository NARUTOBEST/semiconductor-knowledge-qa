# -*- coding: utf-8 -*-
"""上下文管理(Context Management)—— 控制发给 LLM 的 messages 体积。

与 Working Memory 的职责边界:
  - Working Memory(memories/Working memory/):存储每轮工作数据、按需拿取,不裁剪消息
  - 本模块:纯函数,决定"一条工具结果以多长的形式进 messages",不存储、不碰 messages 结构

当前仅实现一项优化:
  truncate_tool_result(result, max_chars) —— 工具结果过长则按合理范围截断,
  保留 chunk_id/source_stem/score 等短元数据,长文本字段截到 max_chars 并提示
  LLM 可用 get_chunk 取全文。

后续的轮次淘汰、token 预算裁剪等上下文管理策略也放在本模块,不污染存储层。
"""
import json

# 已知长文本字段 -> 截断
LONG_FIELDS = {"content", "table_html", "description"}
# 短元数据字段 -> 保留原样(LLM 据此引用 / 调 get_chunk)
SHORT_FIELDS = {
    "chunk_id", "source_stem", "source_path", "source_path_orig",
    "page_num", "page_start", "page_end", "page",
    "score", "heading_path", "caption", "content_type", "image_path",
    "has_table", "heading", "char_count", "chunk_index", "item_type",
    "parent_text_chunk_id",
}
# 先 join 再按长文本处理的 list 字段
LIST_LONG_FIELDS = {"image_descriptions", "image_paths"}


def truncate_tool_result(result, max_chars=800):
    """返回截断后的 JSON 字符串,用于塞进 role=tool 消息发给 LLM。

    - {"error": ...} 原样返回(LLM 需要看到错误)
    - list[chunk] / 单个 chunk:深拷贝后截断长文本字段,保留短元数据
    - 未识别的长字符串字段兜底截断
    """
    if isinstance(result, dict) and "error" in result and len(result) <= 2:
        return json.dumps(result, ensure_ascii=False)

    truncated = _truncate_value(result, max_chars)
    return json.dumps(truncated, ensure_ascii=False, default=str)


def _truncate_value(value, max_chars):
    if isinstance(value, list):
        return [_truncate_value(v, max_chars) for v in value]
    if isinstance(value, dict):
        return {k: _truncate_field(k, v, max_chars) for k, v in value.items()}
    if isinstance(value, str) and len(value) > max_chars:
        return value[:max_chars] + "…[已截断]"
    return value


def _truncate_field(key, value, max_chars):
    if key in LONG_FIELDS and isinstance(value, str):
        if len(value) <= max_chars:
            return value
        return value[:max_chars] + "…[已截断,完整内容请用 get_chunk 获取]"
    if key in LIST_LONG_FIELDS and isinstance(value, list):
        joined = "\n".join(str(x) for x in value)
        if len(joined) > max_chars:
            return joined[:max_chars] + "…[已截断]"
        return value
    if isinstance(value, (dict, list)):
        return _truncate_value(value, max_chars)
    if (isinstance(value, str) and key not in SHORT_FIELDS
            and len(value) > max_chars):
        return value[:max_chars] + "…[已截断]"
    return value
