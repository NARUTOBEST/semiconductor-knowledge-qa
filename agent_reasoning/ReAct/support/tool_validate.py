# -*- coding: utf-8 -*-
"""工具参数 schema 校验(轻量,不依赖 jsonschema)。

validate_runtime 节点用它把 LLM 给出的参数对齐 ToolSpec.parameters:
  - 必填检查;
  - 类型检查 + 宽松转换(number/integer 可由数字串转、boolean 可由 "true"/"false" 转);
  - enum 取值检查;
  - 字符串长度上限(spec.max_input_length);
  - 未知参数(未在 properties 声明)识别(交节点决定提示/忽略)。

纯函数、无编排、无 LLM,便于单测。
"""
from __future__ import annotations

from typing import Any

from tools.base import ToolSpec

_TYPE_NAMES = {
    "string": "字符串", "integer": "整数", "number": "数字",
    "boolean": "布尔(true/false)", "array": "数组", "object": "对象",
}


def _coerce(value: Any, jtype: str) -> tuple[bool, Any]:
    """按 JSON schema 类型校验并在可行时宽松转换。返回 (ok, coerced_value)。"""
    if jtype == "string":
        if isinstance(value, str):
            return True, value
        # 数字/布尔标量转字符串可接受
        if isinstance(value, (int, float, bool)):
            return True, str(value).lower() if isinstance(value, bool) else str(value)
        return False, value
    if jtype == "integer":
        if isinstance(value, bool):  # bool 是 int 子类,排除
            return False, value
        if isinstance(value, int):
            return True, value
        if isinstance(value, float) and value.is_integer():
            return True, int(value)
        if isinstance(value, str):
            s = value.strip()
            try:
                return True, int(s)
            except ValueError:
                try:
                    f = float(s)
                    if f.is_integer():
                        return True, int(f)
                except ValueError:
                    pass
        return False, value
    if jtype == "number":
        if isinstance(value, bool):
            return False, value
        if isinstance(value, (int, float)):
            return True, value
        if isinstance(value, str):
            try:
                return True, float(value.strip())
            except ValueError:
                pass
        return False, value
    if jtype == "boolean":
        if isinstance(value, bool):
            return True, value
        if isinstance(value, str) and value.strip().lower() in ("true", "false"):
            return True, value.strip().lower() == "true"
        return False, value
    if jtype == "array":
        return (True, value) if isinstance(value, list) else (False, value)
    if jtype == "object":
        return (True, value) if isinstance(value, dict) else (False, value)
    # 未声明类型:放行
    return True, value


def validate_arguments(spec: ToolSpec, args: dict) -> tuple[dict, list[str], list[str]]:
    """校验并规范化一次调用的参数。

    :returns: (clean_args, errors, unknown_args)
        clean_args  : 通过校验、类型转换后、仅含声明参数的 dict(可直接喂 handler)
        errors      : 人类可读的校验失败说明列表(空=通过)
        unknown_args: 模型多传、未在 schema 声明的参数名列表
    """
    args = args if isinstance(args, dict) else {}
    schema = spec.parameters or {}
    props: dict = schema.get("properties", {}) or {}
    required = set(schema.get("required", []) or [])
    errors: list[str] = []
    unknown: list[str] = []

    # 未知参数
    for key in args:
        if key not in props:
            unknown.append(key)

    # 必填
    for p in required:
        if p not in args or args.get(p) in (None, ""):
            errors.append(f"缺少必填参数「{p}」")

    clean: dict[str, Any] = {}
    for key, val in args.items():
        if key not in props:
            continue  # 未知参数不进 clean(等价过滤)
        p_schema = props[key] or {}
        jtype = p_schema.get("type")

        # 长度上限(字符串)
        limit = (spec.max_input_length or {}).get(key)
        if limit and isinstance(val, str) and len(val) > limit:
            errors.append(f"参数「{key}」长度 {len(val)} 超过上限 {limit}")
            continue

        if jtype:
            ok, coerced = _coerce(val, jtype)
            if not ok:
                errors.append(
                    f"参数「{key}」应为{_TYPE_NAMES.get(jtype, jtype)}类型,"
                    f"实际收到 {type(val).__name__}: {str(val)[:40]}")
                continue
            val = coerced

        # enum
        enum = p_schema.get("enum")
        if enum and val not in enum:
            errors.append(f"参数「{key}」取值须为 {enum} 之一,实际为 {str(val)[:40]}")
            continue

        clean[key] = val

    return clean, errors, unknown
