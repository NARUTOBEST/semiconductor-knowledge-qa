# -*- coding: utf-8 -*-
"""Query 理解层:LLM 驱动的查询改写。

解决三个问题:
  1. 指代消解: "那它呢?" -> "FIJI F200 的 wafer chuck 温度"
  2. 术语扩展: "TMA 安全" -> "TMA 三甲基铝 安全 腐蚀性 存储"
  3. 复杂拆分: "ALD 和 CVD 区别" -> ["ALD 特点", "CVD 特点", "ALD CVD 对比"]

在 ReAct 循环前调用,生成 1-3 个子查询用于预检索。
"""
import os
import sys
import json
import logging


logger = logging.getLogger("query_rewrite")

_REWRITE_PROMPT = """你是一个查询改写助手。根据对话历史和当前问题，生成 1-3 个用于知识库检索的搜索查询。

任务:
1. 指代消解: 根据历史对话补全代词、省略语
2. 术语扩展: 补充全称、相关术语
3. 复杂问题拆分: 拆成多个独立可检索的子问题

示例:
历史: 用户: TMA 前驱体有哪些安全注意事项? 助手: TMA 具有高反应活性...
当前: 那它的存储温度是多少?
输出: ["TMA 三甲基铝 存储温度", "TMA 前驱体 存储条件"]

历史: (无)
当前: ALD 和 CVD 的区别
输出: ["ALD 原子层沉积 原理 特点", "CVD 化学气相沉积 原理 特点", "ALD CVD 区别 对比"]

历史: (无)
当前: wafer chuck 温度控制
输出: ["wafer chuck 温度控制"]

规则:
- 只输出 JSON 数组，元素为字符串
- 1-3 个查询
- 查询用关键词形式，不要完整句子
- 原问题已清晰时输出 1 个即可

历史:
{history}

当前问题: {question}

输出:"""


def rewrite_query(message, history=None):
    """用 LLM 改写查询，返回 1-3 个搜索查询字符串。

    失败时回退 [message]，不影响主流程。
    """
    try:
        import config as C

        # 构建历史文本(最近 3 轮)
        history_text = ""
        if history:
            for h in history[-6:]:
                role = h.get("role", "")
                content = (h.get("content") or "")[:200]
                if role and content:
                    history_text += f"{role}: {content}\n"
        if not history_text:
            history_text = "(无)"

        prompt = _REWRITE_PROMPT.format(history=history_text, question=message)

        # 走统一的 LLM 调用助手:带指数退避重试 + 主/备模型 failover + 20s 超时
        from agent_reasoning.ReAct.support.llm import get_client, llm_create_with_retry, LLM_TIMEOUT
        resp, err = llm_create_with_retry(
            get_client(), trace_id="rewrite",
            model=C.OPENAI_TEXT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=200,
            timeout=LLM_TIMEOUT,
            retries=2,
        )
        if err is not None:
            raise err

        raw = resp.choices[0].message.content.strip()
        # 去掉可能的 markdown 代码块标记
        raw = raw.replace("```json", "").replace("```", "").strip()

        queries = json.loads(raw)
        if isinstance(queries, list) and queries:
            result = [str(q).strip() for q in queries if q and str(q).strip()]
            if result:
                logger.info(f"query rewrite: '{message[:50]}' -> {result}")
                return result[:3]  # 最多 3 个

        return [message]
    except Exception as e:
        logger.warning(f"query rewrite failed, fallback to original: {e}")
        return [message]
