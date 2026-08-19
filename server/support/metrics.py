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
            }


# 全局单例
metrics = Metrics()
