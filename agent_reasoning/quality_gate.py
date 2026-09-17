# -*- coding: utf-8 -*-
"""共享质检门:两条路径统一的答案质量判定 + 升级建议。

``check(answer, context, tier)`` 返回三态:
  - ``passed``          质量合格,可放行;
  - ``failed``          质量不达标,带 feedback 退回当前路径重做(最多 1 次);
  - ``needs_escalation`` 当前 tier 能力不足以保证质量,建议升级到下一 tier 重跑。

两级范式(simple / raglite / react):
  - simple(L1): 不检索、不调 LLM,只做空答案 + 领域内容启发式;答案涉及半导体领域
    事实(术语命中) -> needs_escalation(升级 react 检索);
  - raglite(快路径): 1 次检索 + 单次作答。空答案 -> failed 重做;检索完全失败
    (search_count=0)-> 升级 react;低置信 -> passed + 可见警告(QC_LOW_CONF_REDO=0);
  - react(L2):  空答案 -> failed 重做;检索低置信默认 passed + 警告(旧行为整轮
    redo 可用 QC_LOW_CONF_REDO=1 恢复)。其余非空即放行。
"""
from __future__ import annotations

from typing import Any

import config as C

from .router import _has_domain_signal

VERDICTS = ("passed", "failed", "needs_escalation")

# 答案被视为"空回答"的最小长度(少于该字符数判不合格)
_MIN_ANSWER_LEN = 2


def _empty_answer(answer: str) -> bool:
    return not (answer or "").strip() or len((answer or "").strip()) < _MIN_ANSWER_LEN


def _check_simple(answer: str, context: dict[str, Any]) -> dict[str, Any]:
    """simple:不检索、不调 LLM。只做空答案 + 领域内容启发式。"""
    warnings: list[str] = []
    if _empty_answer(answer):
        return {"verdict": "failed",
                "feedback": "未生成有效回答,请重试。",
                "warnings": warnings}
    # 答案冒出领域术语/型号/参数 -> 不是 lite 单轮能兜住的,升级检索作答
    if _has_domain_signal(answer):
        return {"verdict": "needs_escalation",
                "feedback": "回答涉及半导体领域事实,需检索验证,升级到 react 检索作答。",
                "warnings": warnings}
    return {"verdict": "passed", "feedback": "", "warnings": warnings}


def _check_react(answer: str, context: dict[str, Any]) -> dict[str, Any]:
    """react:空答案 -> failed 重做;非空但【检索低置信且确实检索过】-> failed
    触发一次"换 thread 干净重进 + 改写关键词"的自适应重试。

    低置信判定仅在自适应开关开启、且本轮确实调用过检索工具(search_count>0)
    时生效——避免检索服务没跑/模型压根没检索时(分数 0)误触发重做;后者由空答案
    或既有拒答逻辑处理。非空且检索置信度达标 -> 放行。
    """
    if _empty_answer(answer):
        return {"verdict": "failed", "feedback": "未生成有效回答,请重试。",
                "warnings": []}

    adaptive = bool(getattr(C, "REACT_ADAPTIVE_RETRIEVAL", True))
    max_score = float(context.get("retrieval_max_score") or 0.0)
    searched = int(context.get("search_count") or 0)
    if (adaptive and searched > 0
            and max_score < float(C.RETRIEVAL_CONFIDENT_SCORE)):
        # 低置信默认不再整轮 redo(QC_LOW_CONF_REDO=0):循环内 reflect_node 的
        # 自适应 requery 已覆盖"可修复"场景,整轮重跑约 +12s 且收益接近重掷骰子。
        # 此时放行并附带可见警告(服务层会把 warnings 以 status 事件透出)。
        if not getattr(C, "QC_LOW_CONF_REDO", False):
            return {
                "verdict": "passed",
                "feedback": "",
                "warnings": [
                    f"检索置信度较低(最高相关分 {max_score:.2f}),"
                    "回答可能不完整,请核实原始文档。"],
            }
        return {
            "verdict": "failed",
            "feedback": (
                "上一轮未检索到高相关资料(最高相关分 "
                f"{max_score:.2f} < {float(C.RETRIEVAL_CONFIDENT_SCORE):.2f})。"
                "请换用设备型号/系列名、报警或故障代码、工序别名、故障现象等"
                "不同关键词重新检索后再作答;若多次检索仍无高相关结果,再如实说明"
                "内部资料未覆盖。"),
            "warnings": [],
        }
    return {"verdict": "passed", "feedback": "", "warnings": []}


def _check_raglite(answer: str, context: dict[str, Any]) -> dict[str, Any]:
    """raglite 快路径:检索 1 次即作答,判定原则:
    - 空答案 -> failed(同层重做一次);
    - 检索完全失败(search_count=0) -> needs_escalation(react 会自行措辞检索);
    - 低置信 -> passed + 可见警告(与 react 的 QC_LOW_CONF_REDO=0 口径一致)。
    注意:刻意不继承 _check_simple 的领域术语升级——raglite 引用领域术语是预期行为。
    """
    warnings: list[str] = []
    if _empty_answer(answer):
        return {"verdict": "failed",
                "feedback": "未生成有效回答,请重试。",
                "warnings": warnings}
    if int(context.get("search_count") or 0) == 0:
        return {"verdict": "needs_escalation",
                "feedback": "检索服务未返回结果,升级 react 换关键词检索作答。",
                "warnings": warnings}
    max_score = float(context.get("retrieval_max_score") or 0.0)
    if max_score < float(C.RETRIEVAL_CONFIDENT_SCORE):
        warnings.append(
            f"检索置信度较低(最高相关分 {max_score:.2f}),"
            "回答可能不完整,请核实原始文档。")
    return {"verdict": "passed", "feedback": "", "warnings": warnings}


def check(answer: str, context: dict[str, Any] | None = None,
          tier: str = "react") -> dict[str, Any]:
    """统一质检入口。

    :param answer:  待检答案文本(assistant_message 内容)。
    :param context: 可选上下文(``question`` / ``history`` 等,供启发式扩展)。
    :param tier:    ``simple`` / ``raglite`` / ``react``。未知 tier 按 react 处理。
    :return: ``{"verdict", "feedback", "warnings"}``。
    """
    context = context or {}
    if tier == "simple":
        return _check_simple(answer, context)
    if tier == "raglite":
        return _check_raglite(answer, context)
    return _check_react(answer, context)
