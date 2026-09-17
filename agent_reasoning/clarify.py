# -*- coding: utf-8 -*-
"""澄清判定(ambiguity -> ask back):在复杂度路由之前判断问题是否信息不足。

生产级 Agent 的标志之一是"知道自己什么时候不知道":当问题缺少关键实体
(设备名/型号/指代无法从上下文消解)时,与其靠查询改写硬猜、答错后一本正经
地误导用户,不如先反问澄清。

判定策略(与 router.classify_complexity 同构:规则预筛 + lite 模型 + fail-open):
  1. 规则预筛(零 LLM 成本,确定性场景):
     - 纯问候/元问题/明确设备型号 -> 不澄清;
     - **无历史上下文**且整句是指代词(那它呢/这个呢/什么意思...) -> 必澄清,
       因为此时没有任何上下文可消解指代,必猜必错。
  2. 其余问题用 lite 模型(TIER_MODEL_SIMPLE)做一次短调用,结合最近对话,
     输出 {"clarify": true/false, "question": "...", "options": ["...", ...]}。
  3. 任何异常 / 解析失败 / 关闭开关 -> 返回 need_clarify=False(fail-open,
     绝不因澄清模块故障阻断对话)。

返回 dict::
    {
      "need_clarify": bool,
      "question": str,        # 澄清时要问用户的话
      "options": [str, ...],  # 可选候选答案(0~4 个,供前端做成可点按钮)
      "source": "rule"|"llm"|"off",
    }
"""
from __future__ import annotations

import json
import logging
import re

import config as C  # noqa: E402
from .ReAct.support.llm import get_client, llm_create_with_retry  # noqa: E402

logger = logging.getLogger("agent")

# 整句为明确的"指代/省略"句式 -> 无历史时必澄清。
# 只匹配最确定的歧义句式(含代词或裸谓词),其余交给 LLM 判定,避免误伤正常短问。
_AMBIGUOUS_STANDALONE_RE = re.compile(
    r"^\s*("
    r"(那|这)(个|台|种|些)?(设备|机器|装置|东西|参数)?.{0,6}(呢|啊|吧|嘛)?|"  # 这个/那台设备呢
    r"它(呢|啊|吧|怎么样|咋样|如何|怎么.{0,8})?|"                              # 它呢/它怎么保养
    r"(然后|接着|后来|之后|接下来).{0,4}(呢|啊|吧)?|"                            # 然后呢/接下来呢
    r"(什么|啥)意思|"                                                          # 什么意思
    r"怎么(弄|搞|做|办|处理|保养|维护|设置|调|用|操作|解决).{0,6}|"               # 怎么保养/怎么弄
    r"怎么样|咋样|如何是好"
    r")\s*[?？。.~！!]*\s*$"
)

# 明确的型号/设备名模式:出现即视为已指明对象,不澄清。
# 如 F200、ASML、AX5500、TMA、wafer chuck、XXX-123 等(字母数字型号 / 已知专名)。
_EXPLICIT_MODEL_RE = re.compile(
    r"([A-Za-z]{2,}[- ]?\d{2,}|"          # F200 / AX-5500 / EUV-3400
    r"\b[A-Z]{2,}\b|"                      # ASML / TMA / ALD / CVD 等缩写专名
    r"wafer\s*chuck|"
    # 公司设备机型/部件类专名(出现即视为已指明对象,不澄清)
    r"光刻机|刻蚀机|沉积设备|键合机|焊线机|固晶机|贴片机|塑封机|模切机|"
    r"划片机|切割机|磨削机|研磨机|清洗机|压印机|纳米压印|检测机|分选机|"
    r"对准器|倒装机|回流焊|EFEM|chuck|主轴|导轨|丝杠|机械手)",
    re.IGNORECASE,
)

# 纯问候 / 元问题(复用 router 的口径,这里只做轻量本地判定)。
_GREETING_RE = re.compile(
    r"^\s*(你好|您好|hi|hello|hey|哈喽|嗨|早(上好)?|晚上好|下午好|在吗|在不在|"
    r"谢谢|多谢|感谢|好的?|嗯|ok|okay|bye|再见|拜拜)[\s!！。.?？~]*$",
    re.IGNORECASE,
)

_CLARIFY_PROMPT = (
    "你是公司内部半导体设备知识问答系统的入口判定器。判断用户当前问题是否因为缺少关键信息"
    "(没说清是哪台设备/哪个机型系列/哪个参数,或代词在当前对话里无法消解)而无法可靠检索作答。\n"
    "判定标准:\n"
    "- 若问题本身已指明对象(含设备名/机型系列/型号/明确术语,如 键合机、固晶机、ASML、F200、"
    "wafer chuck),或属于闲聊/问候/关于助手自身的问题 -> clarify=false。\n"
    "- 若问题依赖代词(它/这个/那个/上述设备)且在给出的对话历史里找不到可指代的具体对象,"
    "或问题过于宽泛(如只说'温度多少''怎么保养''报警怎么处理'却没说哪台/哪款设备) -> clarify=true。\n"
    "- 有对话历史时,若历史里最近讨论的设备/机型能明确消解代词,则 clarify=false。\n"
    "clarify=true 时:\n"
    "- question:一句简短自然的中文反问,问清缺失的关键信息(如'请问您指的是哪款设备或机型系列?');\n"
    "- options:给出 1~4 个可能的候选机型/系列(若无法合理推测可给空数组)。\n"
    "只输出 JSON,不要解释,格式:"
    '{{"clarify": true/false, "question": "...", "options": ["...", "..."]}}\n\n'
    "对话历史(最近几轮,可能为空):\n{history}\n\n"
    "当前问题:{question}"
)


def _rule_prescreen(question: str, history: list | None) -> dict | None:
    """规则预筛。返回可直接定论的结果 dict;无法规则判定返回 None。"""
    q = (question or "").strip()
    if not q:
        return {"need_clarify": False, "question": "", "options": [], "source": "rule"}

    # 问候 / 元问题:不需要澄清(交给 simple 路径)
    if _GREETING_RE.match(q):
        return {"need_clarify": False, "question": "", "options": [], "source": "rule"}

    # 超短(<=1 字,如 "q"、"?"):无意义占位,交给路由器,不调澄清 LLM
    if len(q) <= 1:
        return {"need_clarify": False, "question": "", "options": [], "source": "rule"}

    # 句中含明确型号/设备名:对象已指明,不澄清
    if _EXPLICIT_MODEL_RE.search(q):
        return {"need_clarify": False, "question": "", "options": [], "source": "rule"}

    has_history = bool([h for h in (history or []) if (h.get("content") or "").strip()])

    # 无历史 + 整句是指代词 -> 必澄清(没有任何上下文可消解,必猜)
    if not has_history and _AMBIGUOUS_STANDALONE_RE.match(q):
        # 排除"什么是 ALD"这类其实含术语、但已被上面型号正则放行之外的裸问:
        # 能走到这里说明不含型号/专名,且是纯指代短句。
        return {
            "need_clarify": True,
            "question": "请问您指的是哪台设备、哪个型号或哪个工艺参数?补充后我才能准确查找资料。",
            "options": [],
            "source": "rule",
        }

    return None


def _parse_output(text: str) -> dict | None:
    """解析 LLM 输出 JSON,容忍 markdown 代码块/前后缀。失败返回 None。"""
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
    if not bool(obj.get("clarify")):
        return {"need_clarify": False, "question": "", "options": [], "source": "llm"}
    question = str(obj.get("question", "")).strip()
    if not question:
        return None
    options = obj.get("options") or []
    if not isinstance(options, list):
        options = []
    options = [str(o).strip() for o in options if str(o).strip()][:4]
    return {"need_clarify": True, "question": question, "options": options, "source": "llm"}


def check_clarify(question: str, history: list | None = None) -> dict:
    """判断问题是否需要先澄清。

    :return: ``{"need_clarify": bool, "question": str, "options": [str],
        "source": "rule"|"llm"|"off"}``。任何异常都 fail-open 为不澄清。
    """
    if not getattr(C, "CLARIFY_ENABLED", True):
        return {"need_clarify": False, "question": "", "options": [], "source": "off"}

    # 1. 规则预筛
    ruled = _rule_prescreen(question, history)
    if ruled is not None:
        return ruled

    # 2. LLM 判定(结合最近对话)
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

    prompt = _CLARIFY_PROMPT.format(history=history_text, question=question)
    try:
        resp, err = llm_create_with_retry(
            get_client(), trace_id="clarify",
            model=C.TIER_MODEL_SIMPLE,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=200,
            timeout=getattr(C, "CLARIFY_TIMEOUT", C.ROUTER_TIMEOUT),
            retries=2,
        )
        if err is not None:
            logger.warning("clarify LLM failed, skip: %s", err)
            return {"need_clarify": False, "question": "", "options": [], "source": "llm"}
        text = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        logger.warning("clarify LLM exception, skip: %s", e)
        return {"need_clarify": False, "question": "", "options": [], "source": "llm"}

    parsed = _parse_output(text)
    if parsed is None:
        logger.info("clarify output unparseable (%r), skip", text[:120])
        return {"need_clarify": False, "question": "", "options": [], "source": "llm"}
    return parsed
