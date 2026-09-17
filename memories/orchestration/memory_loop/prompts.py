# -*- coding: utf-8 -*-
"""memory-loop 节点二(会话摘要)的 prompt。

规范要求"fork 子代理"做摘要;项目无外部 agent CLI,只有 HTTP LLM API,
故用【独立 LLM 调用】模拟子代理(独立 system 角色、与主应答模型调用解耦):
  - 级别1 SUBAGENT:角色化"会话记忆整理子代理",完整 9 章节,多次重试;
  - 级别2 DIRECT :降级为短 prompt 直连 API,单次、短超时;
  - 级别3       :不调 LLM,规则截断(见 session_summary.py)。
"""
from __future__ import annotations

from ...storage.working.session_file import NINE_SECTION_TITLES

_SECTIONS = "\n".join(f"{i+1}. {t}" for i, t in enumerate(NINE_SECTION_TITLES))

# Req3 叙事约束:会话摘要只记录【讨论脉络/已达成结论/待办】,是"发生了什么"的叙事;
# 用户的身份、长期偏好/语气/格式等【偏好断言】以长期记忆(PG)为唯一权威,不写进摘要
# (避免摘要与长期记忆双写、过期偏好污染)。
_NARRATIVE_RULE = (
    "重要:本摘要只记录对话的【叙事脉络】——讨论了什么、达成了什么结论/决策、还有哪些"
    "待办与未决问题。【不要】把用户的身份角色、长期偏好、语气/格式习惯等偏好断言写进来"
    "(这些以独立的长期记忆为准,摘要里写偏好会造成双写与过期污染);“用户身份与偏好”"
    "“关键事实与数据”两章也只写本轮【讨论中明确得出的结论/数据点】,不复述用户画像。"
)

# 级别1:子代理角色(独立上下文,只做记忆整理,不参与作答)
SUBAGENT_SYS = (
    "你是会话记忆整理【子代理】,独立于主应答助手运行,唯一职责是把对话整理成结构化"
    "工作记忆,供未来会话快速恢复上下文。你不回答用户问题、不与用户对话。\n"
    "请严格按以下 9 个章节输出 markdown(章节标题保持不变,无内容的章节写“无”):\n"
    f"{_SECTIONS}\n"
    "要求:结论/决策/数据要具体可复用;待办与下一步要可执行。\n"
    + _NARRATIVE_RULE +
    "\n只输出 9 章节正文,不要任何前言、解释或 markdown 代码块围栏。"
)

# 级别2:直连 API 降级(更短指令、更省 token,仍输出 9 章节)
DIRECT_SYS = (
    "把下面的对话整理为 9 节工作记忆 markdown,标题固定为:\n"
    f"{_SECTIONS}\n"
    "无内容写“无”;只输出正文,简洁中文,不要前言。\n"
    + _NARRATIVE_RULE
)

_INCREMENTAL_INSTR = (
    "【已有摘要】(在其基础上增量合并,保留仍有效的信息,更新过期状态,不要丢失"
    "尚未完成的待办):\n{prev}\n\n"
    "【本次新增对话】:\n{transcript}\n"
)

_FIRST_INSTR = (
    "【会话对话记录】:\n{transcript}\n"
)


def build_user_prompt(transcript: str, prev_summary: str = "",
                      max_chars: int = 12000) -> str:
    """组装摘要请求的 user 段。prev_summary 非空走增量合并,否则首摘。"""
    transcript = (transcript or "")[-max_chars:]
    if prev_summary:
        body = _INCREMENTAL_INSTR.format(
            prev=prev_summary[-max_chars:], transcript=transcript)
    else:
        body = _FIRST_INSTR.format(transcript=transcript)
    return body
