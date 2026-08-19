# -*- coding: utf-8 -*-
"""复杂度路由器:把用户问题分到 simple / medium / complex 三条推理路径。

策略(3.1-3.3):
  1. 规则预筛(省一次 LLM 调用):
     - 纯问候/寒暄/关于助手自身的元问题 -> simple
     - 含强复杂特征词(对比/分别/优缺点/流程/步骤/综合...)或多子问题 -> complex
  2. 其余问题用 lite 模型(TIER_MODEL_SIMPLE)做一次短调用,只输出
     {"tier": "...", "confidence": 0.x},超时 ROUTER_TIMEOUT。
  3. 解析失败 / 置信度低于 ROUTER_CONFIDENCE_MIN -> 兜底 medium。

分类口径(3.2):
  - simple 仅限闲聊/元问题/明确不需要领域知识;
  - 领域事实题哪怕很短也走 medium;
  - complex 为多子问题、多维度对比、含"对比/分别/优缺点/流程/步骤/综合"等特征。

任何异常都 fail-open 到 medium(绝不因路由器故障阻断对话)。
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

import config as C  # noqa: E402
from .ReAct.support.llm import get_client, llm_create_with_retry  # noqa: E402

logger = logging.getLogger("agent")

VALID_TIERS = ("simple", "medium", "complex")

# 纯问候/寒暄/元问题(命中即规则判 simple,不调 LLM)
_GREETING_RE = re.compile(
    r"^\s*(你好|您好|hi|hello|hey|哈喽|嗨|早(上好)?|晚上好|下午好|在吗|在不在|"
    r"谢谢|多谢|感谢|好的?|嗯|ok|okay|bye|再见|拜拜)[\s!！。.?？~]*$",
    re.IGNORECASE,
)
_META_QUESTION_RE = re.compile(
    r"(你是谁|你能做什么|你会什么|介绍一下你自己|你是(什么|哪个|啥)|怎么用你|"
    r"你(的)?(名字|功能|模型)|help|帮助)",
    re.IGNORECASE,
)

# 强复杂特征词(命中即规则判 complex,省 LLM 调用)。与 nodes._COMPLEX_MARKERS 对齐并扩充。
_COMPLEX_MARKERS = (
    "对比", "比较", "分别", "优缺点", "区别", "差异", "流程", "步骤",
    "综合", "总结", "以及", "并且", "同时", "两者", "多个", "各自",
)

# 领域信号词(出现在很短的问题里也说明需要检索 -> 至少 medium)
_DOMAIN_SIGNALS = (
    "半导体", "芯片", "晶圆", "光刻", "刻蚀", "沉积", "薄膜", "离子注入", "CMP",
    "ALD", "CVD", "PVD", "MOCVD", "EUV", "DUV", "FET", "MOSFET", "IGBT",
    "键合", "贴装", "封装", "良率", "工艺节点", "制程", "外延", "光刻胶",
)

_ROUTER_PROMPT = (
    "你是问题复杂度分类器。判断下面的用户问题属于哪一类,只输出 JSON,"
    '格式 {{"tier": "simple|medium|complex", "confidence": 0.0~1.0}}。\n'
    "分类标准:\n"
    "- simple:仅闲聊、寒暄、关于助手自身的元问题,明确不需要任何领域知识或检索;\n"
    "- medium:需要半导体领域知识、事实查询、单点原理/术语/型号/参数解释,需要检索;\n"
    "  注意:领域事实题哪怕很短也必须是 medium,不能因为短就判 simple;\n"
    "- complex:包含多个子问题、多维度对比/比较、需要分步骤流程或综合多份资料,"
    "含'对比/分别/优缺点/流程/步骤/综合'等特征。\n"
    "只输出 JSON,不要解释。\n\n"
    "问题:{question}"
)


def _rule_prescreen(question: str) -> tuple[str | None, float]:
    """规则预筛。返回 (tier, confidence);无法判定返回 (None, 0.0)。"""
    q = (question or "").strip()
    if not q:
        return "medium", 1.0  # 空问题兜底 medium,由后续正常处理

    # 纯问候/寒暄 -> simple
    if _GREETING_RE.match(q):
        return "simple", 0.95
    # 关于助手自身的元问题(且不含领域术语) -> simple
    if _META_QUESTION_RE.search(q) and not _has_domain_signal(q):
        return "simple", 0.9

    # 强复杂特征 -> complex
    if any(m in q for m in _COMPLEX_MARKERS):
        return "complex", 0.85

    # 超短且无领域信号、无复杂特征 -> simple(日常短句)
    if len(q) <= C.ROUTER_SHORT_LEN and not _has_domain_signal(q):
        return "simple", 0.7

    return None, 0.0


def _has_domain_signal(text: str) -> bool:
    return any(sig.lower() in text.lower() for sig in _DOMAIN_SIGNALS)


def _parse_router_output(text: str) -> tuple[str | None, float]:
    """从 LLM 输出解析 {tier, confidence},容忍 markdown 代码块/前后缀。"""
    if not text:
        return None, 0.0
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None, 0.0
    try:
        obj = json.loads(text[start:end + 1])
    except Exception:
        return None, 0.0
    tier = str(obj.get("tier", "")).strip().lower()
    if tier not in VALID_TIERS:
        return None, 0.0
    try:
        conf = float(obj.get("confidence", 0.0))
    except Exception:
        conf = 0.0
    return tier, conf


def classify_complexity(question: str,
                        history: list | None = None) -> dict[str, Any]:
    """分类问题复杂度。

    :return: ``{"tier": "simple|medium|complex", "confidence": float,
        "source": "rule"|"llm"|"fallback"}``。任何异常都兜底 medium。
    """
    # 1. 规则预筛
    tier, conf = _rule_prescreen(question)
    if tier is not None:
        return {"tier": tier, "confidence": conf, "source": "rule"}

    # 2. LLM 分类
    prompt = _ROUTER_PROMPT.format(question=question)
    try:
        resp, err = llm_create_with_retry(
            get_client(), trace_id="router",
            model=C.TIER_MODEL_SIMPLE,
            messages=[{"role": "user", "content": prompt}],
            temperature=0, timeout=C.ROUTER_TIMEOUT,
        )
        if err is not None:
            logger.warning("router LLM failed, fallback medium: %s", err)
            return {"tier": "medium", "confidence": 0.0, "source": "fallback"}
        text = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        logger.warning("router LLM exception, fallback medium: %s", e)
        return {"tier": "medium", "confidence": 0.0, "source": "fallback"}

    tier, conf = _parse_router_output(text)
    if tier is None:
        logger.info("router output unparseable (%r), fallback medium", text[:120])
        return {"tier": "medium", "confidence": 0.0, "source": "fallback"}
    if conf < C.ROUTER_CONFIDENCE_MIN:
        logger.info("router low confidence %.2f (%s), fallback medium", conf, tier)
        return {"tier": "medium", "confidence": conf, "source": "fallback"}

    return {"tier": tier, "confidence": conf, "source": "llm"}
