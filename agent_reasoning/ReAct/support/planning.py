# -*- coding: utf-8 -*-
"""共享检索规划(阶段 5.1):从原 ``core/nodes.plan_node`` 抽出的 LLM 规划逻辑。

medium ReAct(条件触发,steps 作为软引导)与 complex Plan-and-Execute(必经,
steps 驱动执行循环)都调用本模块的 ``generate_plan()``,保证两条路径的规划
prompt、JSON 解析、步数上限与失败降级口径一致。

本模块只做"调 LLM + 解析 JSON",不发 SSE 事件、不碰 TraceRecorder——事件与
trace 由调用方按各自路径风格处理(medium 走图节点 writer,P&E 走路径生成器)。
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import config as C  # noqa: E402

from .llm import get_client, llm_create_with_retry, LLM_TIMEOUT  # noqa: E402

logger = logging.getLogger("agent")

# 计划最多拆解步数(与原 nodes.MAX_PLAN_STEPS 对齐;9.x 可按 tier 配置化)
MAX_PLAN_STEPS = 4
# 复杂问题粗筛阈值(与原 nodes.PLAN_TRIGGER_MIN_LEN 对齐)
PLAN_TRIGGER_MIN_LEN = 30
# 剩余总预算低于此值则跳过规划(避免规划 LLM 挤占生成时间)
PLAN_MIN_REMAINING_SECONDS = 10

# 强复杂特征词(与 router._COMPLEX_MARKERS 对齐并扩充)
COMPLEX_MARKERS = (
    "对比", "比较", "分别", "优缺点", "区别", "差异", "流程", "步骤",
    "综合", "总结", "以及", "并且", "同时", "两者", "多个", "各自",
)


def looks_complex(question: str, sub_queries: Optional[list[str]] = None) -> bool:
    """复杂问题粗筛:长问题 / 改写多子查询 / 含多意图特征词,任一命中即复杂。"""
    q = (question or "").strip()
    if len(q) >= PLAN_TRIGGER_MIN_LEN:
        return True
    if sub_queries and len(sub_queries) > 1:
        return True
    return any(m in q for m in COMPLEX_MARKERS)


def parse_plan_json(text: str) -> dict[str, Any] | None:
    """从 LLM 输出解析计划 JSON(容忍 markdown 代码块/前后缀文本)。"""
    import json
    if not text:
        return None
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(text[start:end + 1])
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def generate_plan(question: str,
                  *,
                  max_steps: int = MAX_PLAN_STEPS,
                  force: bool = False,
                  timeout: Optional[float] = None,
                  trace_id: str = "plan",
                  model: Optional[str] = None) -> tuple[list[str], str | None]:
    """调一次 LLM 生成检索计划步骤。

    :param question:  用户原问题。
    :param max_steps: 步数上限。
    :param force:     True=P&E 必经规划,prompt 要求必须拆解(不要返回 need_plan=false);
                      False=medium 条件规划,允许 LLM 判定单点事实题返回空。
    :param timeout:   LLM 超时,默认 LLM_TIMEOUT。
    :param trace_id:  追踪 id。
    :param model:     覆盖模型(默认 OPENAI_TEXT_MODEL)。
    :return: ``(steps, error)``。``steps`` 为步骤字符串列表(LLM 判定无需规划时为
        ``[]``);``error`` 为非空字符串表示 LLM 调用/解析失败(调用方可据此给可见
        警示),正常或"无需规划"时为 ``None``。P&E 调用方只看 steps 是否为空即可降级。
    """
    if force:
        plan_instruction = (
            "该问题较复杂,请把它拆解成多步检索计划,need_plan 必须为 true,"
            f"steps 最多 {max_steps} 步,每步一句话说明该检索什么/做什么。"
        )
    else:
        plan_instruction = (
            "need_plan 仅当问题包含多个子任务或需要多轮不同角度检索时为 true,"
            f"单点事实型问题为 false;steps 最多 {max_steps} 步,每步一句话。"
        )

    prompt = (
        "你是检索规划器。判断下面的问题是否需要拆解成多步检索计划,"
        '只输出 JSON,格式:{"need_plan": true, "steps": ["步骤1", "步骤2"]}。'
        f"{plan_instruction}\n\n问题:{question}"
    )

    try:
        resp, err = llm_create_with_retry(
            get_client(), trace_id=trace_id,
            model=model or C.OPENAI_TEXT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            timeout=timeout if timeout is not None else LLM_TIMEOUT,
        )
    except Exception as e:
        logger.warning("generate_plan exception: %s", e)
        return [], f"{type(e).__name__}: {e}"
    if err is not None:
        logger.warning("generate_plan failed: %s", err)
        return [], f"{type(err).__name__}: {err}"

    try:
        text = (resp.choices[0].message.content or "").strip()
    except Exception:
        text = ""
    plan = parse_plan_json(text)
    if not plan:
        # 解析失败也算错误(给 medium 可见提示);空 steps 但合法 JSON 且 need_plan=false
        # 则是"无需规划",不算错误。
        if not force and text and "need_plan" in text and "false" in text.lower():
            return [], None
        return [], "计划结果解析失败"
    if not force and not plan.get("need_plan"):
        return [], None
    steps = [str(s).strip() for s in (plan.get("steps") or [])
             if str(s).strip()][:max_steps]
    return steps, None
