# -*- coding: utf-8 -*-
"""轻量级 metrics 收集器(内存,线程安全)。

不依赖 Prometheus/Grafana,适合 10 用户内部工具。
数据存在内存中,重启清零(足够用,不需要持久化)。

追踪指标:
  - 请求: 总数/错误数/错误率/延迟(avg/p50/p95)
  - 工具: 每个工具的调用次数/失败次数/成功率
  - 检索: 命中/未命中/命中率
  - Grounding: 通过/失败/通过率
  - Token: prompt/completion/total 用量
  - 用户: 每用户请求次数
"""
import time
import threading
from collections import defaultdict


class Metrics:
    def __init__(self):
        self._lock = threading.Lock()
        self._start_time = time.time()
        self._reset()

    def _reset(self):
        self._requests = 0
        self._errors = 0
        self._latencies = []           # ms 列表(保留最近 1000 条)
        self._tool_calls = defaultdict(int)
        self._tool_failures = defaultdict(int)
        self._search_hits = 0
        self._search_misses = 0
        self._grounding_passed = 0
        self._grounding_failed = 0
        self._tokens = {"prompt": 0, "completion": 0, "total": 0}
        self._per_user = defaultdict(int)
        # 按推理范式(simple/medium/complex)分维度统计(阶段 8.4)
        self._tier_count = defaultdict(int)
        self._tier_escalations = defaultdict(int)   # 该 tier 被升级离开的次数
        self._tier_errors = defaultdict(int)
        self._tier_latencies = defaultdict(list)    # 每 tier 延迟 ms(最近 1000)
        self._tier_grounding_passed = defaultdict(int)
        self._tier_grounding_failed = defaultdict(int)
        self._tier_tokens = defaultdict(lambda: {"prompt": 0, "completion": 0, "total": 0})
        self._escalations_total = 0

    def record_request(self, username, latency_ms, error=False):
        with self._lock:
            self._requests += 1
            if error:
                self._errors += 1
            self._latencies.append(latency_ms)
            if len(self._latencies) > 1000:
                self._latencies = self._latencies[-1000:]
            if username:
                self._per_user[username] += 1

    def record_tool_call(self, tool_name, success=True):
        with self._lock:
            self._tool_calls[tool_name] += 1
            if not success:
                self._tool_failures[tool_name] += 1

    def record_search(self, hit=True):
        with self._lock:
            if hit:
                self._search_hits += 1
            else:
                self._search_misses += 1

    def record_grounding(self, passed=True):
        with self._lock:
            if passed:
                self._grounding_passed += 1
            else:
                self._grounding_failed += 1

    def record_tokens(self, prompt=0, completion=0):
        with self._lock:
            self._tokens["prompt"] += prompt
            self._tokens["completion"] += completion
            self._tokens["total"] += prompt + completion

    def record_escalation(self, from_tier, to_tier):
        """一次范式升级(simple->medium / medium->complex)。"""
        with self._lock:
            self._escalations_total += 1
            if from_tier:
                self._tier_escalations[from_tier] += 1

    def record_tier_result(self, tier, latency_ms, *, error=False,
                           escalated=False, grounding_passed=None,
                           tokens=None):
        """一次请求结束时按最终范式记账(阶段 8.4)。

        :param tier: 最终产出答案的范式(simple/medium/complex)
        :param latency_ms: 整条请求耗时
        :param error: 是否出错
        :param escalated: 本次请求是否经历过升级
        :param grounding_passed: 最终答案 grounding 是否通过(None 表示未做)
        :param tokens: 整条请求的 token 用量 dict(prompt/completion/total)
        """
        with self._lock:
            self._tier_count[tier] += 1
            if error:
                self._tier_errors[tier] += 1
            lats = self._tier_latencies[tier]
            lats.append(latency_ms)
            if len(lats) > 1000:
                del lats[:-1000]
            if grounding_passed is True:
                self._tier_grounding_passed[tier] += 1
            elif grounding_passed is False:
                self._tier_grounding_failed[tier] += 1
            if tokens:
                tt = self._tier_tokens[tier]
                tt["prompt"] += int(tokens.get("prompt", 0) or 0)
                tt["completion"] += int(tokens.get("completion", 0) or 0)
                tt["total"] += int(tokens.get("total", 0) or 0)

    def get_stats(self):
        with self._lock:
            lats = sorted(self._latencies)
            n = len(lats)
            avg = sum(lats) / n if n else 0
            p50 = lats[n // 2] if n else 0
            p95 = lats[int(n * 0.95)] if n else 0
            total_search = self._search_hits + self._search_misses
            total_ground = self._grounding_passed + self._grounding_failed

            return {
                "uptime_seconds": int(time.time() - self._start_time),
                "requests": {
                    "total": self._requests,
                    "errors": self._errors,
                    "error_rate": round(self._errors / self._requests, 4)
                        if self._requests else 0,
                },
                "latency_ms": {
                    "avg": round(avg),
                    "p50": p50,
                    "p95": p95,
                },
                "tools": {
                    name: {
                        "calls": self._tool_calls[name],
                        "failures": self._tool_failures[name],
                        "success_rate": round(
                            1 - self._tool_failures[name] / self._tool_calls[name], 4
                        ) if self._tool_calls[name] else 1,
                    }
                    for name in self._tool_calls
                },
                "search": {
                    "hits": self._search_hits,
                    "misses": self._search_misses,
                    "hit_rate": round(self._search_hits / total_search, 4)
                        if total_search else 0,
                },
                "grounding": {
                    "passed": self._grounding_passed,
                    "failed": self._grounding_failed,
                    "pass_rate": round(self._grounding_passed / total_ground, 4)
                        if total_ground else 0,
                },
                "tokens": dict(self._tokens),
                "per_user": dict(self._per_user),
                "escalations_total": self._escalations_total,
                "escalation_rate": round(
                    self._escalations_total / self._requests, 4
                ) if self._requests else 0,
                "by_tier": self._tier_stats(),
            }

    def _percentile(self, sorted_vals, q):
        n = len(sorted_vals)
        if not n:
            return 0
        return sorted_vals[min(n - 1, int(n * q))]

    def _tier_stats(self):
        out = {}
        for tier in self._tier_count:
            lats = sorted(self._tier_latencies[tier])
            n = len(lats)
            g_pass = self._tier_grounding_passed[tier]
            g_fail = self._tier_grounding_failed[tier]
            g_total = g_pass + g_fail
            count = self._tier_count[tier]
            out[tier] = {
                "requests": count,
                "errors": self._tier_errors[tier],
                "error_rate": round(self._tier_errors[tier] / count, 4) if count else 0,
                "escalations_from": self._tier_escalations[tier],
                "latency_ms": {
                    "avg": round(sum(lats) / n) if n else 0,
                    "p50": self._percentile(lats, 0.50),
                    "p95": self._percentile(lats, 0.95),
                },
                "grounding": {
                    "passed": g_pass,
                    "failed": g_fail,
                    "pass_rate": round(g_pass / g_total, 4) if g_total else 0,
                },
                "tokens": dict(self._tier_tokens[tier]),
            }
        return out


# 全局单例
metrics = Metrics()
