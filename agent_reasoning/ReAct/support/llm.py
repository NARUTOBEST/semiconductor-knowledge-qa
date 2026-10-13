# -*- coding: utf-8 -*-
"""LLM 客户端单例 + 带指数退避的流式调用(从原 chat.service 抽出)。"""
import json
import threading
import time
import logging

import config as C
from agent_reasoning.ReAct.support.tool_circuit import (
    CircuitOpenError, get_llm_breaker,
)

logger = logging.getLogger("agent")

LLM_RETRIES = 3
# 非流式 LLM 请求的读超时(秒):必须在此时间内返回完整响应体。
LLM_TIMEOUT = 20
# 流式 LLM 请求的读超时(秒):这是 token 间"空闲超时",持续吐字不会被切断,
# 仅当连续 10s 无任何新数据才判定卡死断开。
STREAM_TIMEOUT = 10

_llm_client = None


class LLMServiceUnavailable(Exception):
    """某环节的 LLM 熔断器已放弃(OPEN / 连续 2 次半开试探失败),本次调用拒绝。

    str(e) 即用户可读提示;关键环节(raglite/react 主作答)捕获后把它作为
    SSE error 消息下发并认定本次作答无效,辅助环节(改写/救场/压缩/grounding)
    捕获后按既有降级路径处理。
    """

    def __init__(self, site: str, exhausted: bool = False):
        self.site = site
        self.exhausted = exhausted
        super().__init__(
            f"该环节({site})LLM 服务不可用,请稍后作答,本次作答无效")


def _llm_error_type(e: Exception) -> str:
    """LLM 调用失败的熔断记账分类:配置类 4xx 不代表服务故障,fatal 不计数。"""
    status = getattr(e, "status_code", None)
    if status is None:
        status = getattr(getattr(e, "response", None), "status_code", None)
    if status in (400, 401, 403, 404, 422):
        return "fatal"
    return "timeout"


def _llm_breaker_guard(site: str):
    """进入 LLM 调用前过熔断器;OPEN/放弃期抛 LLMServiceUnavailable。

    返回熔断器实例(供成功/失败记账),特性关闭返回 None。
    """
    brk = get_llm_breaker(site)
    if brk is None:
        return None
    try:
        brk.before_call()
    except CircuitOpenError as e:
        raise LLMServiceUnavailable(site, exhausted=e.exhausted) from e
    return brk


class _LLMGate:
    """LLM 出口波次并发闸:同时最多 size 个调用在途,超出的排队攒波齐发。

    规则(2026-10-10 用户拍板):
    - 在途 < size 且无排队者 → 直通,零等待(低负载完全不感知);
    - 在途满 → 后续请求排队,等当前波【全部结束】后一次性放下一波
      (≤size 个攒批齐发),直到队列穷尽;
    - 排队超过 wait_timeout 秒 → 抛 TimeoutError(调用方报"稍后重试")。
    目的:把 20 并发齐发对 Ark 的秒级突发冲击削成 ≤size 的波次,避免首波
    集体流式挂死(轮次14 实测 600s 超时集中第一波)。
    """

    def __init__(self, size: int, wait_timeout: float = 300.0):
        self.size = max(1, int(size))
        self.wait_timeout = float(wait_timeout)
        self._cond = threading.Condition()
        self._inflight = 0
        self._waiters = 0
        self._promote = 0   # 新一波已放出的名额数(由 release 时一次性分发)

    def acquire(self) -> None:
        with self._cond:
            if self._inflight == 0 and self._waiters == 0:
                self._inflight += 1
                return
            if 0 < self._inflight < self.size and self._waiters == 0:
                self._inflight += 1   # 低负载:当前波有空位且无人排队,直通
                return
            self._waiters += 1
            promoted = False
            try:
                deadline = time.time() + self.wait_timeout
                while self._promote <= 0:
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        raise TimeoutError(
                            f"当前请求较多,已排队 {self.wait_timeout:.0f}s,"
                            "请稍后重试")
                    self._cond.wait(remaining)
                self._promote -= 1
                promoted = True
            finally:
                self._waiters -= 1
                if promoted:
                    self._inflight += 1

    def release(self) -> None:
        with self._cond:
            self._inflight -= 1
            if self._inflight <= 0:
                self._inflight = 0
                if self._waiters > 0:
                    # 当前波清零:开新一波,名额一次性分给队列前排
                    self._promote = min(self.size, self._waiters)
                    self._cond.notify_all()

    @property
    def stats(self) -> dict:
        with self._cond:
            return {"inflight": self._inflight, "waiters": self._waiters}


# 进程内单例;LLM_GATE_ENABLED=False 一键关闭(直通)。
_GATE: "_LLMGate | None" = (
    _LLMGate(getattr(C, "LLM_GATE_SIZE", 10),
             getattr(C, "LLM_GATE_WAIT_TIMEOUT", 300.0))
    if getattr(C, "LLM_GATE_ENABLED", True) else None)


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
    """按模型族注入思考开关(各模型族行为不同,实测):
    - doubao 系:支持 thinking.type=disabled;路由类短 prompt 带思考 12-14s,
      关闭后 2-3s——所有经此函数的 light 调用(路由/质检/升迁门)统一受益;
    - GLM 系:纯思考模型,请求体带 thinking 参数直接 400 InvalidParameter,
      必须剔除(GLM 经 reasoning_content 独立字段思考,不污染 content);
    - 自建网关(Qwen3 混合思考模型,如 Qwen3-32B-FP8/14B-FP8):chat 模板默认
      enable_thinking=True,每次调用先思考数十秒——经 chat_template_kwargs
      显式关闭(思考内容由服务端 --reasoning-parser 分离,不污染正文)。
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
    elif getattr(C, "LLM_GATEWAY_ACTIVE", False):
        # 自建网关(Qwen3 混合思考模型):chat 模板默认 enable_thinking=True,
        # 每次调用先思考数十秒——经 chat_template_kwargs 显式关闭;
        # 思考内容若仍产生,由服务端 --reasoning-parser qwen3 分离到独立字段。
        extra = dict(kwargs.get("extra_body") or {})
        ctk = extra.get("chat_template_kwargs")
        ctk = dict(ctk) if isinstance(ctk, dict) else {}
        ctk.setdefault("enable_thinking", False)
        extra["chat_template_kwargs"] = ctk
        kwargs["extra_body"] = extra


def llm_create_with_retry(client, trace_id="", retries=LLM_RETRIES, site="", **kwargs):
    """带指数退避重试的 LLM 调用,主模型耗尽后自动切备用模型(P1⑥)。

    入口两道闸:
    - 熔断器(三状态,按 site 环节):OPEN/放弃期抛 LLMServiceUnavailable;
    - 波次并发闸(_LLMGate,全局):在途 ≤LLM_GATE_SIZE 直通,超出的攒波
      排队,排队超时返回 (None, TimeoutError)——两者都先于任何网络调用。

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
    brk = _llm_breaker_guard(site or "llm")
    if _GATE is not None:
        try:
            _GATE.acquire()
        except TimeoutError as e:
            # 排队超时=后端自我保护生效,不计入熔断(不是上游故障)
            logger.warning(json.dumps({
                "trace_id": trace_id, "event": "llm_gate_timeout",
                "site": site or "llm", "gate": _GATE.stats,
            }, ensure_ascii=False))
            return None, e
    try:
        return _llm_create_impl(client, trace_id=trace_id, brk=brk,
                                retries=retries, **kwargs)
    finally:
        if _GATE is not None:
            _GATE.release()


def _llm_create_impl(client, trace_id="", brk=None, retries=LLM_RETRIES, **kwargs):
    """主备模型循环 + 指数退避(由 llm_create_with_retry 包裹:熔断/并发闸在外层)。

    brk 为 site 环节的熔断器(可为 None),成功/最终失败在此记账。
    """
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
                resp = client.chat.completions.create(**kwargs)
                if brk is not None:
                    brk.on_success()
                return resp, None
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
                    if brk is not None:
                        brk.on_failure(_llm_error_type(e))
                    return None, e
                wait = 1.5 ** i
                logger.info(json.dumps({
                    "trace_id": trace_id, "event": "llm_retry",
                    "model": m, "attempt": i + 1, "wait_s": round(wait, 1),
                    "error": str(e)[:120],
                }, ensure_ascii=False))
                time.sleep(wait)
    if brk is not None and last_err is not None:
        brk.on_failure(_llm_error_type(last_err))
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
