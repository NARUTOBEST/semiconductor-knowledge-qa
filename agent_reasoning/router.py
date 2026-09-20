# -*- coding: utf-8 -*-
"""复杂度路由器:把用户问题分到 simple / raglite / react 三条推理路径。

  L1 simple   日常闲聊/关于助手自身的元问题,单轮直答无工具;
  L2 raglite  单一事实点快路径:1 次检索 + 1 次主模型作答(不走 ReAct 循环);
  L3 react    复杂知识问答,ReAct 检索循环(检索三件套),无旁路 LLM。

策略:
  1. 规则预筛(省一次 LLM 调用):问候/元问题 -> simple;超短无领域 -> simple;
     领域关键词命中且非复杂标记 -> raglite。
  2. 其余问题用 lite 模型做一次短调用,只输出 {"tier", "confidence"},超时 ROUTER_TIMEOUT。
  3. 解析失败 / 置信度低 / 经济模式 -> 兜底 react。

任何异常都 fail-open 到 react(绝不因路由器故障阻断对话)。

融合入口 ``classify_and_clarify``(ROUTER_FUSED=1 时服务层使用):把澄清判定与
路由合并为单次 light 调用,规则可判定时零 LLM,省掉旧两段式的一倍前置延迟。
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

import config as C  # noqa: E402
from .ReAct.support.llm import get_client, llm_create_with_retry, no_think_extra  # noqa: E402

logger = logging.getLogger("agent")

VALID_TIERS = ("simple", "raglite", "react")

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

# 领域信号词(出现在很短的问题里也说明需要检索 -> 至少 react)
_DOMAIN_SIGNALS = (
    # 工艺/器件
    "半导体", "芯片", "晶圆", "光刻", "刻蚀", "沉积", "薄膜", "离子注入", "CMP",
    "ALD", "CVD", "PVD", "MOCVD", "EUV", "DUV", "FET", "MOSFET", "IGBT",
    "键合", "贴装", "封装", "良率", "工艺节点", "制程", "外延", "光刻胶",
    # 设备机型/部件(公司设备知识库主场景)
    "键合机", "焊线机", "固晶机", "贴片机", "塑封机", "模切机", "划片机", "切割机",
    "磨削机", "研磨机", "清洗机", "光刻机", "压印机", "纳米压印", "检测机", "分选机",
    "对准器", "倒装机", "回流焊", "EFEM", "chuck", "主轴", "导轨", "丝杠", "机械手",
    "气浮", "空气轴承", "密封圈", "滤芯",
    # 设备支持类问法
    "报警", "告警", "故障码", "报警代码", "报错", "异常", "保养", "点检", "维护",
    "维修", "操作规程", "SOP", "换型", "校准", "对中", "参数设置", "规格书",
    "手册", "说明书", "厂商", "SEMI",
)

_ROUTER_PROMPT = (
    "你是问题复杂度分类器。判断下面的用户问题属于哪一类,只输出 JSON,"
    '格式 {{"tier": "simple|raglite|react", "confidence": 0.0~1.0}}。\n'
    "分类标准:\n"
    "- simple:仅闲聊、寒暄、关于助手自身的元问题,明确不需要任何领域知识或检索;\n"
    "- raglite:单一事实点问题——一个参数/术语/一个报警或故障代码的含义与处理/"
    "某型号的某项规格/某个保养点等,一次检索即可作答;"
    "即使问法带\"是什么原因/怎么处理\",只要围绕单一报警码或单一对象,也算 raglite;\n"
    "- react:需要多步检索或跨资料综合的问题——两个/多个对象的对比或区别、"
    "多设备横向比较、跨章节的枚举汇总(如\"划片工艺涉及哪些工具\")等。\n"
    "只输出 JSON,不要解释。\n\n"
    "问题:{question}"
)

# 融合分类 + 澄清判定(ROUTER_FUSED=1):单次 light 调用同时完成两件事,
# 替代旧的 clarify + router 两次串行调用。
_ROUTER_FUSED_PROMPT = (
    "你是公司内部半导体设备知识问答系统的入口判定器,一次完成两件事:"
    "(1)判断问题是否因缺少关键信息而需要先反问澄清;"
    "(2)判断应走哪条回答路径。只输出 JSON,不要解释,格式:"
    '{{"need_clarify": true/false, "question": "澄清反问(不需要澄清时为空串)",'
    ' "options": ["..."], "tier": "simple|raglite|react", "confidence": 0.0~1.0}}\n'
    "澄清判定标准:\n"
    "- 问题已指明对象(设备名/机型系列/型号/明确术语,如 键合机、ASML、F200、wafer chuck)"
    "或属于闲聊/问候/关于助手自身的问题 -> need_clarify=false;\n"
    "- 问题依赖代词(它/这个/那个)且对话历史里找不到可指代的具体对象,或过于宽泛"
    "(只说'温度多少''怎么保养'却没说哪台设备) -> need_clarify=true,"
    "question 给一句简短自然的中文反问,options 给 1~4 个候选机型(推测不出给空数组);\n"
    "- 有对话历史时,最近讨论的设备/机型能明确消解代词 -> need_clarify=false。\n"
    "路径判定标准:\n"
    "- simple:仅闲聊、寒暄、关于助手自身的元问题,明确不需要任何领域知识或检索;\n"
    "- raglite:单一事实点问题——一个参数/术语/一个报警或故障代码的含义与处理/"
    "某型号的某项规格/某个保养点等,一次检索即可作答;"
    "即使问法带\"是什么原因/怎么处理\",只要围绕单一报警码或单一对象,也算 raglite;\n"
    "- react:需要多步检索或跨资料综合的问题——两个/多个对象的对比或区别、"
    "多设备横向比较、跨章节的枚举汇总(如\"划片工艺涉及哪些工具\")等。\n\n"
    "对话历史(最近几轮,可能为空):\n{history}\n\n"
    "当前问题:{question}"
)

# 复杂标记:命中即不走 raglite 规则快路。只保留"确实需要多步检索/跨资料综合"
# 的信号——对比类、枚举汇总类、多步骤流程、组成构成枚举。注意:原因/排查/维修/
# 调试等词常出现在单一事实点问法里(如"1416 报警是什么原因怎么处理"),
# 不应单独触发 react(实测 10 题里 3 道单点题被误路由)。
_RAGLITE_COMPLEX_RE = re.compile(
    r"对比|区别|差异|比较|优缺点|哪些|几种|几类|总结|汇总|归纳|全部|所有|"
    r"组成|构成|流程|步骤|"
    r"和.{0,12}(的?区别|相比)|与.{0,12}(的?区别|相比)"
)

# 跨条目聚合强标记:即使命中单点报警码,这类问法也要 react(对比/汇总多个条目)
_RAGLITE_AGG_RE = re.compile(r"对比|区别|差异|比较|总结|汇总|归纳|全部|所有|哪些")

# 单一事实点强信号:明确报警/错误代码类问题,即使带"为什么/怎么处理"的措辞,
# 也是一次检索即可作答的单点问题(如"报 1416 是什么原因"、"报警代码 E0063")
_SINGLE_FACT_RE = re.compile(
    r"报\s*了?\s*[A-Za-z]?\d{2,5}"
    r"|(?:报警|故障|错误|警示)(?:码|代码|信息|号)?\s*[:：]?\s*[A-Za-z]?\d{2,5}"
    r"|[A-Za-z]?\d{2,5}\s*(?:报警|报错|警报|故障代码|错误代码)"
)


def _raglite_eligible(question: str) -> bool:
    """单一事实点快路径的规则准入:领域信号命中 + 非复杂标记 + 长度不超限。

    明确的报警/故障码单点问题强制准入(仅让位于跨条目聚合问法)。
    """
    q = (question or "").strip()
    if not q or len(q) > C.RAGLITE_MAX_QUESTION_LEN:
        return False
    single = _SINGLE_FACT_RE.search(q)
    if not _has_domain_signal(q) and not single:
        return False  # 命中报警码本身即视为领域信号("报了1416"无领域词)
    if single:
        return not _RAGLITE_AGG_RE.search(q)
    return not _RAGLITE_COMPLEX_RE.search(q)


def _has_domain_signal(text: str) -> bool:
    return any(sig.lower() in text.lower() for sig in _DOMAIN_SIGNALS)


def _obviously_simple(text: str) -> bool:
    """明显闲聊判定(投机检索跳过条件):问候、或超短且无领域信号。
    与 _rule_prescreen 的 simple 规则对齐,但比领域词表宽松——词表无法
    穷举(实测"重掺砷硅单晶/离子发生器/3D IC 检测"均漏),投机检索宁滥勿缺。"""
    q = (text or "").strip()
    if not q:
        return True
    if _GREETING_RE.match(q):
        return True
    if len(q) <= C.ROUTER_SHORT_LEN and not _has_domain_signal(q):
        return True
    return False


def _rule_prescreen(question: str) -> tuple[str | None, float]:
    """规则预筛。返回 (tier, confidence);无法判定返回 (None, 0.0)。"""
    q = (question or "").strip()
    if not q:
        return "react", 1.0  # 空问题兜底 react,由后续正常处理

    # 纯问候/寒暄 -> simple
    if _GREETING_RE.match(q):
        return "simple", 0.95
    # 关于助手自身的元问题(且不含领域术语) -> simple
    if _META_QUESTION_RE.search(q) and not _has_domain_signal(q):
        return "simple", 0.9

    # 超短且无领域信号、无复杂特征 -> simple(日常短句)
    if len(q) <= C.ROUTER_SHORT_LEN and not _has_domain_signal(q):
        return "simple", 0.7

    # 领域关键词命中 + 非复杂标记 + 长度不超限 -> raglite 快路径(零 LLM)
    if _raglite_eligible(q):
        return "raglite", 0.85

    # 复杂度强信号 + 领域词命中 -> react 快路径(零 LLM):对比/汇总/枚举类问法
    # 按路由判定标准必属 react。省去每题一次串行 light 融合调用(20 并发下
    # 摊薄后 TTFT+解码 ~1-2s)。误伤面:单点问题若含"区别/哪些"等词,落到
    # react 也会被多步检索正常作答,质量不降只是路径更重。
    if getattr(C, "ROUTER_REACT_FAST", True) and _has_domain_signal(q) \
            and _RAGLITE_AGG_RE.search(q):
        return "react", 0.85

    return None, 0.0


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

    :return: ``{"tier": "simple|react", "confidence": float,
        "source": "rule"|"llm"|"fallback"|"economy"}``。任何异常都兜底 react。
    """
    # 1. 规则预筛
    tier, conf = _rule_prescreen(question)
    if tier is not None:
        return {"tier": tier, "confidence": conf, "source": "rule"}

    # 经济模式(回退云端单账号):跳过 LLM 分类以省调用,规则未命中一律兜底 react。
    if not getattr(C, "ROUTER_LLM_ENABLED", True):
        return {"tier": "react", "confidence": 0.0, "source": "economy"}

    # 2. LLM 分类
    prompt = _ROUTER_PROMPT.format(question=question)
    try:
        resp, err = llm_create_with_retry(
            get_client(), trace_id="router",
            model=C.TIER_MODEL_SIMPLE,
            messages=[{"role": "user", "content": prompt}],
            temperature=0, timeout=C.ROUTER_TIMEOUT,
            **no_think_extra(),
        )
        if err is not None:
            logger.warning("router LLM failed, fallback react: %s", err)
            return {"tier": "react", "confidence": 0.0, "source": "fallback"}
        text = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        logger.warning("router LLM exception, fallback react: %s", e)
        return {"tier": "react", "confidence": 0.0, "source": "fallback"}

    tier, conf = _parse_router_output(text)
    if tier is None:
        logger.info("router output unparseable (%r), fallback react", text[:120])
        return {"tier": "react", "confidence": 0.0, "source": "fallback"}
    if conf < C.ROUTER_CONFIDENCE_MIN:
        logger.info("router low confidence %.2f (%s), fallback react", conf, tier)
        return {"tier": "react", "confidence": conf, "source": "fallback"}

    return {"tier": tier, "confidence": conf, "source": "llm"}


def _parse_fused_output(text: str) -> dict | None:
    """解析融合分类输出 JSON,容忍 markdown 代码块/前后缀。失败返回 None。"""
    if not text:
        return None
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(text[start:end + 1])
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    need_clarify = bool(obj.get("need_clarify"))
    tier = str(obj.get("tier", "")).strip().lower()
    if tier not in VALID_TIERS:
        return None
    try:
        conf = float(obj.get("confidence", 0.0))
    except Exception:
        conf = 0.0
    options = obj.get("options") or []
    if not isinstance(options, list):
        options = []
    options = [str(o).strip() for o in options if str(o).strip()][:4]
    return {
        "need_clarify": need_clarify,
        "question": str(obj.get("question", "")).strip(),
        "options": options,
        "tier": tier,
        "confidence": conf,
    }


def classify_and_clarify(question: str,
                         history: list | None = None) -> dict[str, Any]:
    """融合入口:澄清判定 + 复杂度路由,单次 light 调用完成(规则可判时零 LLM)。

    :return: ``{"need_clarify": bool, "question": str, "options": [str],
        "tier": str|None, "confidence": float, "source": "rule"|"llm"|"fallback"}``。
        need_clarify=True 时 tier 为 None(服务层直接走澄清反问,不作答)。
        任何异常 fail-open:不澄清 + react。
    """
    from . import clarify as _clarify  # 局部导入避免环(clarify 不依赖 router)

    # 1. 澄清规则预筛:命中必澄清的确定性场景直接短路(零 LLM)
    ruled = _clarify._rule_prescreen(question, history)
    if ruled is not None and ruled.get("need_clarify"):
        return {**ruled, "tier": None, "confidence": 1.0}

    # 2. 路由规则预筛:问候/元问题/领域快路命中 -> 零 LLM 定 tier(不澄清)
    tier, conf = _rule_prescreen(question)
    if tier is not None:
        return {"need_clarify": False, "question": "", "options": [],
                "tier": tier, "confidence": conf, "source": "rule"}

    # 3. 融合 LLM 判定(单次调用)
    history_text = "(无)"
    if history:
        lines = []
        for h in history[-6:]:
            role = h.get("role", "")
            content = (h.get("content") or "")[:200]
            if role and content:
                lines.append(f"{role}: {content}")
        if lines:
            history_text = "\n".join(lines)
    prompt = _ROUTER_FUSED_PROMPT.format(history=history_text, question=question)
    try:
        resp, err = llm_create_with_retry(
            get_client(), trace_id="router-fused",
            model=C.TIER_MODEL_SIMPLE,
            messages=[{"role": "user", "content": prompt}],
            temperature=0, max_tokens=200,
            timeout=max(getattr(C, "CLARIFY_TIMEOUT", 8), C.ROUTER_TIMEOUT),
            **no_think_extra(),
            retries=2,
        )
        if err is not None:
            logger.warning("fused router LLM failed, fallback react: %s", err)
            return {"need_clarify": False, "question": "", "options": [],
                    "tier": "react", "confidence": 0.0, "source": "fallback"}
        text = (resp.choices[0].message.content or "").strip()
    except Exception as e:  # noqa: BLE001
        logger.warning("fused router LLM exception, fallback react: %s", e)
        return {"need_clarify": False, "question": "", "options": [],
                "tier": "react", "confidence": 0.0, "source": "fallback"}

    parsed = _parse_fused_output(text)
    if parsed is None:
        logger.info("fused router output unparseable (%r), fallback react",
                    text[:120])
        return {"need_clarify": False, "question": "", "options": [],
                "tier": "react", "confidence": 0.0, "source": "fallback"}
    # 澄清优先:需要澄清时不作答
    if parsed["need_clarify"] and parsed["question"]:
        return {**parsed, "tier": None, "confidence": parsed["confidence"]}
    tier = parsed["tier"] or "react"
    # 防提示漂移:模型判 react 但问题符合 raglite 规则准入 -> 降级 raglite
    if tier == "react" and _raglite_eligible(question):
        tier = "raglite"
    conf = parsed["confidence"]
    if conf < C.ROUTER_CONFIDENCE_MIN:
        logger.info("fused router low confidence %.2f (%s), fallback react",
                    conf, tier)
        return {"need_clarify": False, "question": "", "options": [],
                "tier": "react", "confidence": conf, "source": "fallback"}
    return {"need_clarify": False, "question": parsed["question"],
            "options": parsed["options"], "tier": tier,
            "confidence": conf, "source": "llm"}
