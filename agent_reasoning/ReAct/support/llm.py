# -*- coding: utf-8 -*-
"""LLM 客户端单例 + 带指数退避的流式调用(从原 chat.service 抽出)。"""
import json
import threading
import time
import logging

import config as C

logger = logging.getLogger("agent")

LLM_RETRIES = 3
# 非流式 LLM 请求的读超时(秒):必须在此时间内返回完整响应体。
LLM_TIMEOUT = 20
# 流式 LLM 请求的读超时(秒):这是 token 间"空闲超时",持续吐字不会被切断,
# 仅当连续 10s 无任何新数据才判定卡死断开。
STREAM_TIMEOUT = 10

_llm_client = None


def get_client():
    """全局单例 OpenAI 客户端。

    指向"有效端点":配置了自建 LLM 网关(LLM_GATEWAY_BASE_URL)则走网关,
    否则回退云端 ARK。换 GPU 服务器只改 env,此处与业务代码都不动。
    """
    global _llm_client
    if _llm_client is None:
        _llm_client = __import__("openai").OpenAI(
            api_key=getattr(C, "EFFECTIVE_LLM_API_KEY", C.OPENAI_API_KEY),
            base_url=getattr(C, "EFFECTIVE_LLM_BASE_URL", C.OPENAI_BASE_URL),
            timeout=LLM_TIMEOUT,
        )
    return _llm_client


def _apply_thinking_policy(kwargs: dict) -> None:
    """按模型族注入思考开关(云端两族模型行为不同,实测):
    - doubao 系:支持 thinking.type=disabled;路由类短 prompt 带思考 12-14s,
      关闭后 2-3s——所有经此函数的 light 调用(路由/质检/升迁门)统一受益;
    - GLM 系:纯思考模型,请求体带 thinking 参数直接 400 InvalidParameter,
      必须剔除(GLM 经 reasoning_content 独立字段思考,不污染 content)。
    LLM_ENABLE_THINKING=1 整体豁免(恢复思考)。"""
    if getattr(C, "LLM_ENABLE_THINKING", False):
        return
    low = str(kwargs.get("model") or "").lower()
    if "doubao" in low:
        extra = dict(kwargs.get("extra_body") or {})
        extra.setdefault("thinking", {"type": "disabled"})
        kwargs["extra_body"] = extra
    elif "glm" in low:
        kwargs.pop("thinking", None)
        extra = dict(kwargs.get("extra_body") or {})
        extra.pop("thinking", None)
        if extra:
            kwargs["extra_body"] = extra
        else:
            kwargs.pop("extra_body", None)


def llm_create_with_retry(client, trace_id="", retries=LLM_RETRIES, **kwargs):
    """带指数退避重试的 LLM 调用,主模型耗尽后自动切备用模型(P1⑥)。

    返回 (stream, error):成功时 error=None;失败(含备用模型)返回 (None, error)。
    仅对创建请求抛出的同步异常重试/切换;流式中途断线由调用方兜底。
    """
    # retries<1 时循环体一次都不执行,函数会隐式返回 None,
    # 调用方 `stream, err = ...` 解包直接 TypeError -- 钳到至少 1 次
    if retries < 1:
        retries = 1
    # 非流式快速 failover:GLM 纯思考模型在抽象问题上 20s(LLM_TIMEOUT)内
    # 常出不来,同模型重试 3 次只是再等 3 个 20s(实测单调用拖 65s+,整请求
    # 被 QC/reflect 多次调用拖过 200s)。非流式只试一次即切 doubao——
    # 流式调用不受影响(建立连接的报错是秒级,重试代价低)。
    if not kwargs.get("stream", False) and retries > 1:
        retries = 1
    model = kwargs.pop("model", "") or ""
    fallback = getattr(C, "EFFECTIVE_FALLBACK_MODEL", "") or getattr(C, "OPENAI_FALLBACK_MODEL", "") or ""
    models = [model] + ([fallback] if (fallback and model and fallback != model) else [])
    last_err = None
    for mi, m in enumerate(models):
        kwargs["model"] = m
        _apply_thinking_policy(kwargs)
        # 事实型问答默认低温:抑制模型自加资料外的推测/安全提示/延伸建议(忠实度)。
        # 调用方显式传 temperature 时不覆盖。
        kwargs.setdefault("temperature", float(getattr(C, "LLM_TEMPERATURE", 0.3)))
        for i in range(retries):
            try:
                return client.chat.completions.create(**kwargs), None
            except Exception as e:
                last_err = e
                if i == retries - 1:
                    if mi < len(models) - 1:
                        logger.info(json.dumps({
                            "trace_id": trace_id, "event": "llm_failover",
                            "from_model": m, "to_model": models[mi + 1],
                            "error": str(e)[:120],
                        }, ensure_ascii=False))
                        break  # 换下一个模型再试
                    logger.info(json.dumps({
                        "trace_id": trace_id, "event": "llm_fail",
                        "model": m, "attempt": i + 1, "error": str(e)[:120],
                    }, ensure_ascii=False))
                    return None, e
                wait = 1.5 ** i
                logger.info(json.dumps({
                    "trace_id": trace_id, "event": "llm_retry",
                    "model": m, "attempt": i + 1, "wait_s": round(wait, 1),
                    "error": str(e)[:120],
                }, ensure_ascii=False))
                time.sleep(wait)
    return None, last_err


def arm_stream_watchdog(resp, deadline_s: float):
    """给已建立的流式响应装【总时长】看门狗,返回 (cancel, killed)。

    背景:GLM 等思考模型流式输出时 reasoning 块持续到达,基于 read 空闲的
    timeout(STREAM_TIMEOUT)永远不会触发,单步思考实测可拖 5 分钟+(中等难度
    对比题 14K prompt 实测),客户端与整条流水线全被拖死。看门狗到点强制
    close 底层响应:迭代侧表现为流提前结束,调用方按"思考超时截断"处理,
    不得重试(重试等于重新思考一遍)。
    """
    killed = threading.Event()

    def _kill():
        killed.set()
        try:
            resp.close()
        except Exception:  # noqa: BLE001  close 失败无妨,迭代侧另有超时
            pass

    t = threading.Timer(deadline_s, _kill)
    t.daemon = True
    t.start()

    def cancel():
        t.cancel()

    return cancel, killed


def no_think_extra() -> dict:
    """关闭思考模式的透传参数。
    - vLLM 网关(Qwen3 系):chat_template_kwargs.enable_thinking=False;
    - 云端直连:不发任何参数——GLM-5.3-flash 是纯思考模型, Ark 网关对
      thinking.type=disabled 直接报 InvalidParameter(实测 400),
      且其 reasoning_content 走独立字段、不污染 content,只多花解码时间。
    LLM_ENABLE_THINKING=1 可整体回退(恢复思考)。"""
    if getattr(C, "LLM_ENABLE_THINKING", False):
        return {}
    if getattr(C, "LLM_GATEWAY_ACTIVE", False):
        return {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
    return {}
