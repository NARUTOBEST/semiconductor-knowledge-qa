# -*- coding: utf-8 -*-
"""共享质检门(阶段 4):三条路径统一的答案质量判定 + 升级建议。

``check(answer, context, tier)`` 返回三态:
  - ``passed``          质量合格,可放行;
  - ``failed``          质量不达标,带 feedback 退回当前路径重做(最多 1 次);
  - ``needs_escalation`` 当前 tier 能力不足以保证质量,建议升级到下一 tier 重跑。

按 tier 分深度(4.2):
  - simple: 安全/格式检查;答案若涉及半导体领域事实(启发式术语命中) -> needs_escalation;
  - medium: 调用现有 grounding_check(引用 + 忠实度);若问题本身含复杂特征且
    grounding 失败 -> needs_escalation(升级 P&E),否则 failed 退回重做;
  - complex: 引用 + 忠实度 + 覆盖度(复用 CoverageTracker 产出);complex 无法再升级,
    失败只返回 failed(由调用方重做一次后带警示放行)。

质检 LLM 失败时由 grounding_check 内部 fail-open(返回 passed + 可见警示),
本门绝不因质检异常阻断对话(4.5)。
"""
from __future__ import annotations

import logging
from typing import Any

from .ReAct.support.answer_grounding import grounding_check
from .router import _COMPLEX_MARKERS, _has_domain_signal

logger = logging.getLogger("agent")

VERDICTS = ("passed", "failed", "needs_escalation")

# 答案被视为"空回答"的最小长度(少于该字符数且无引用,判格式不合格)
_MIN_ANSWER_LEN = 2


def _empty_answer(answer: str) -> bool:
    return not (answer or "").strip() or len((answer or "").strip()) < _MIN_ANSWER_LEN


def _question_looks_complex(question: str) -> bool:
    """问题含强复杂特征词或多个问号 -> 更适合 complex P&E。"""
    q = question or ""
    if any(m in q for m in _COMPLEX_MARKERS):
        return True
    # 多个子问题(中英文问号计数 >=2)
    if q.count("?") + q.count("？") >= 2:
        return True
    return False


def _check_simple(answer: str, context: dict[str, Any]) -> dict[str, Any]:
    """simple:不检索、不调 LLM。只做格式 + 领域内容启发式。"""
    warnings: list[str] = []
    if _empty_answer(answer):
        return {"verdict": "failed",
                "feedback": "未生成有效回答,请重试。",
                "warnings": warnings}
    # 答案冒出领域术语/型号/参数 -> 不是 lite 单轮能兜住的,升级检索
    if _has_domain_signal(answer):
        return {"verdict": "needs_escalation",
                "feedback": "回答涉及半导体领域事实,需检索验证,升级到 medium。",
                "warnings": warnings}
    return {"verdict": "passed", "feedback": "", "warnings": warnings}


def _grounding_result(answer: str, context: dict[str, Any]) -> dict[str, Any]:
    """拿到 grounding 结果,任何异常都 fail-open(passed + 警示)。

    若 context 已带 ``grounding``(medium 图内部已算过、P&E synthesizer 后已算),
    直接复用,避免重复一次 LLM 忠实度调用;否则当场调 grounding_check。
    """
    precomputed = context.get("grounding")
    if isinstance(precomputed, dict) and "passed" in precomputed:
        return precomputed
    sources = context.get("sources") or []
    try:
        return grounding_check(answer, sources)
    except Exception as e:
        logger.warning("quality gate grounding error (fail-open): %s", e)
        return {"passed": True,
                "warnings": ["答案质检因临时异常暂不可用,请自行核实答案"]}


def _check_medium(answer: str, context: dict[str, Any]) -> dict[str, Any]:
    """medium:grounding(引用 + 忠实度)。

    grounding 失败时:问题本身含复杂特征 -> 升级 complex;否则 failed 退回重做。
    """
    if _empty_answer(answer):
        return {"verdict": "failed", "feedback": "未生成有效回答,请重试。",
                "warnings": []}

    g = _grounding_result(answer, context)
    warnings = list(g.get("warnings") or [])
    if g.get("passed"):
        return {"verdict": "passed", "feedback": "", "warnings": warnings}

    feedback = "答案未通过来源校验:" + "; ".join(warnings[:2])
    if _question_looks_complex(context.get("question", "")):
        return {"verdict": "needs_escalation",
                "feedback": feedback + "。问题含多个子任务,升级 complex 走计划执行。",
                "warnings": warnings}

    # medium 图内部已用 reflect 机制重做过一次(MAX_REFLECT=1)再把 grounding 结果
    # 写到 trace["grounding"]。此时外层不应再把整张图重跑一遍(会重复同样的失败);
    # 按 fail-open 带可见警示放行。只有当场新算 grounding(无预计算结果,如直接调用)
    # 才返回 failed 让外层重做。
    precomputed = context.get("grounding")
    if isinstance(precomputed, dict) and "passed" in precomputed:
        return {"verdict": "passed", "feedback": "", "warnings": warnings}
    return {"verdict": "failed", "feedback": feedback, "warnings": warnings}


def _check_complex(answer: str, context: dict[str, Any]) -> dict[str, Any]:
    """complex:grounding + 覆盖度。complex 不再升级,失败只返回 failed。"""
    if _empty_answer(answer):
        return {"verdict": "failed", "feedback": "未生成有效回答,请重试。",
                "warnings": []}

    warnings: list[str] = []
    g = _grounding_result(answer, context)
    warnings.extend(g.get("warnings") or [])

    # 覆盖度:CoverageTracker 产出 {"uncovered_steps": [...]};None 表示未启用/未判定,
    # 按 fail-open 不计为失败(阶段 5.7 决定 P&E 是否挂 tracker)。
    coverage = context.get("coverage")
    uncovered = []
    if isinstance(coverage, dict):
        uncovered = coverage.get("uncovered_steps") or []
    if uncovered:
        warnings.append(f"计划中有 {len(uncovered)} 个步骤未在答案中体现:"
                        f"{', '.join(map(str, uncovered[:3]))}")

    if g.get("passed") and not uncovered:
        return {"verdict": "passed", "feedback": "", "warnings": warnings}

    feedback = "答案未通过 complex 质检:" + "; ".join(warnings[:2])
    return {"verdict": "failed", "feedback": feedback, "warnings": warnings}


def check(answer: str, context: dict[str, Any] | None = None,
          tier: str = "medium") -> dict[str, Any]:
    """统一质检入口。

    :param answer:  待检答案文本(assistant_message 内容)。
    :param context: 可选上下文:``question`` / ``sources``(list[dict]) /
                    ``coverage``(CoverageTracker 产出) / ``history``。
    :param tier:    ``simple`` / ``medium`` / ``complex``。未知 tier 按 medium 处理。
    :return: ``{"verdict", "feedback", "warnings"}``。
    """
    context = context or {}
    if tier == "simple":
        return _check_simple(answer, context)
    if tier == "complex":
        return _check_complex(answer, context)
    return _check_medium(answer, context)
