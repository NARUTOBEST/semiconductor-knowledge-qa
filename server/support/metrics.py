# -*- coding: utf-8 -*-
"""轻量级 metrics 收集器(状态外置 Redis,内存回退,线程安全)。

不依赖 Prometheus/Grafana,适合 10 用户内部工具。

存储:默认 Redis(多 worker 聚合统计;键带 TTL,默认 7 天滚动清理,写入续期),
Redis 不可用时自动回退进程内存(=外置前行为:重启清零、单进程),见 state_store。
注意:Redis↔内存切换瞬间统计各算各的,跨切换窗口的数据不合并(尽力而为口径)。

追踪指标(get_stats 输出形状与外置前完全一致):
  - 请求: 总数/错误数/错误率/延迟(avg/p50/p95)
  - 工具: 每个工具的调用次数/失败次数/成功率/耗时/缓存命中
  - 检索: 命中/未命中/命中率
  - Token: prompt/completion/total 用量(含记忆链内部分账)
  - 范式: simple/react 各自请求量/错误率/升级数/延迟/token
  - 用户: 每用户请求次数
"""
import os
import time
import threading
from collections import defaultdict

from support import state_store

_TTL_S = int(os.getenv("METRICS_TTL_DAYS", "7")) * 86400
_LAT_CAP = 1000     # 延迟样本保留条数(与外置前一致)


def _p(key):
    return f"metrics:{key}"


def _expire(p, *keys):
    for k in keys:
        p.expire(_p(k), _TTL_S)


class Metrics:
    def __init__(self):
        self._lock = threading.Lock()
        self._start_time = time.time()
        self._reset()

    def _reset(self):
        # ---- 内存回退存储(Redis 不可用时的唯一事实源)----
        self._requests = 0
        self._errors = 0
        self._latencies = []           # ms 列表(保留最近 1000 条)
        self._tool_calls = defaultdict(int)
        self._tool_failures = defaultdict(int)
        # 工具维度增强(可选 kwarg,不破坏旧调用)
        self._tool_durations = defaultdict(list)   # 最近 1000 条 ms
        self._tool_cache_hits = defaultdict(int)
        self._tool_categories = {}                  # name -> category
        self._category_calls = defaultdict(int)
        self._category_failures = defaultdict(int)
        self._search_hits = 0
        self._search_misses = 0
        self._tokens = {"prompt": 0, "completion": 0, "total": 0}
        # 记忆链内部 LLM(升迁门/会话摘要)token,与作答 billable 分账(Req12),不计入 _tokens。
        self._internal_tokens = {"prompt": 0, "completion": 0, "total": 0}
        self._per_user = defaultdict(int)
        # 按推理范式(simple/react)分维度统计
        self._tier_count = defaultdict(int)
        self._tier_escalations = defaultdict(int)   # 该 tier 被升级离开的次数
        self._tier_errors = defaultdict(int)
        self._tier_latencies = defaultdict(list)    # 每 tier 延迟 ms(最近 1000)
        self._tier_tokens = defaultdict(lambda: {"prompt": 0, "completion": 0, "total": 0})
        self._escalations_total = 0

    # ---------------- 记录 ----------------

    def record_request(self, username, latency_ms, error=False):
        r = state_store.get_state_redis()
        if r is not None:
            try:
                p = r.pipeline(transaction=False)
                p.incr(_p("req:total"))
                if error:
                    p.incr(_p("req:errors"))
                p.lpush(_p("req:lat"), int(latency_ms))
                p.ltrim(_p("req:lat"), 0, _LAT_CAP - 1)
                if username:
                    p.hincrby(_p("per_user"), username, 1)
                _expire(p, "req:total", "req:errors", "req:lat", "per_user")
                p.execute()
                return
            except Exception:
                state_store.note_fail()
        with self._lock:
            self._requests += 1
            if error:
                self._errors += 1
            self._latencies.append(latency_ms)
            if len(self._latencies) > 1000:
                self._latencies = self._latencies[-1000:]
            if username:
                self._per_user[username] += 1

    def record_tool_call(self, tool_name, success=True, *, category=None,
                         duration_ms=0, error_type=None, cache_hit=False):
        """记录一次工具调用(向后兼容:新增 kwarg 均有默认值)。

        :param category: 工具类别(仅 retrieval),用于类别聚合
        :param duration_ms: 本次调用耗时(毫秒),计入每工具 p95
        :param error_type: 失败时的分类(timeout/retryable/fatal/circuit_open...)
        :param cache_hit: 是否命中结果缓存
        """
        r = state_store.get_state_redis()
        if r is not None:
            try:
                p = r.pipeline(transaction=False)
                p.hincrby(_p("tool:calls"), tool_name, 1)
                if category:
                    p.hset(_p("tool:cat"), tool_name, category)
                    p.hincrby(_p("cat:calls"), category, 1)
                if not success:
                    p.hincrby(_p("tool:failures"), tool_name, 1)
                    if category:
                        p.hincrby(_p("cat:failures"), category, 1)
                if duration_ms:
                    p.lpush(_p(f"tool:dur:{tool_name}"), int(duration_ms))
                    p.ltrim(_p(f"tool:dur:{tool_name}"), 0, _LAT_CAP - 1)
                if cache_hit:
                    p.hincrby(_p("tool:cache_hits"), tool_name, 1)
                _expire(p, "tool:calls", "tool:cat", "cat:calls", "tool:failures",
                        "cat:failures", "tool:cache_hits",
                        f"tool:dur:{tool_name}")
                p.execute()
                return
            except Exception:
                state_store.note_fail()
        with self._lock:
            self._tool_calls[tool_name] += 1
            if category:
                self._tool_categories[tool_name] = category
                self._category_calls[category] += 1
            if not success:
                self._tool_failures[tool_name] += 1
                if category:
                    self._category_failures[category] += 1
            if duration_ms:
                durs = self._tool_durations[tool_name]
                durs.append(int(duration_ms))
                if len(durs) > 1000:
                    del durs[:-1000]
            if cache_hit:
                self._tool_cache_hits[tool_name] += 1

    def record_search(self, hit=True):
        r = state_store.get_state_redis()
        if r is not None:
            try:
                p = r.pipeline(transaction=False)
                p.incr(_p("search:hits" if hit else "search:misses"))
                _expire(p, "search:hits", "search:misses")
                p.execute()
                return
            except Exception:
                state_store.note_fail()
        with self._lock:
            if hit:
                self._search_hits += 1
            else:
                self._search_misses += 1

    def _record_tokens_hash(self, r, key, prompt, completion):
        p = r.pipeline(transaction=False)
        p.hincrby(_p(key), "prompt", prompt)
        p.hincrby(_p(key), "completion", completion)
        p.hincrby(_p(key), "total", prompt + completion)
        _expire(p, key)
        p.execute()

    def record_tokens(self, prompt=0, completion=0):
        r = state_store.get_state_redis()
        if r is not None:
            try:
                self._record_tokens_hash(r, "tokens", prompt, completion)
                return
            except Exception:
                state_store.note_fail()
        with self._lock:
            self._tokens["prompt"] += prompt
            self._tokens["completion"] += completion
            self._tokens["total"] += prompt + completion

    def record_internal_tokens(self, prompt=0, completion=0):
        """记忆链内部 LLM(升迁门/摘要)token 分账(Req12):不混入 billable tokens。"""
        r = state_store.get_state_redis()
        if r is not None:
            try:
                self._record_tokens_hash(r, "internal_tokens", prompt, completion)
                return
            except Exception:
                state_store.note_fail()
        with self._lock:
            self._internal_tokens["prompt"] += prompt
            self._internal_tokens["completion"] += completion
            self._internal_tokens["total"] += prompt + completion

    def record_escalation(self, from_tier, to_tier):
        """一次范式升级(simple -> react)。"""
        r = state_store.get_state_redis()
        if r is not None:
            try:
                p = r.pipeline(transaction=False)
                p.incr(_p("escalations:total"))
                if from_tier:
                    p.hincrby(_p("tier:escalations"), from_tier, 1)
                _expire(p, "escalations:total", "tier:escalations")
                p.execute()
                return
            except Exception:
                state_store.note_fail()
        with self._lock:
            self._escalations_total += 1
            if from_tier:
                self._tier_escalations[from_tier] += 1

    def record_tier_result(self, tier, latency_ms, *, error=False,
                           escalated=False, tokens=None):
        """一次请求结束时按最终范式记账。

        :param tier: 最终产出答案的范式(simple/react)
        :param latency_ms: 整条请求耗时
        :param error: 是否出错
        :param escalated: 本次请求是否经历过升级
        :param tokens: 整条请求的 token 用量 dict(prompt/completion/total)
        """
        r = state_store.get_state_redis()
        if r is not None:
            try:
                p = r.pipeline(transaction=False)
                p.hincrby(_p("tier:count"), tier, 1)
                if error:
                    p.hincrby(_p("tier:errors"), tier, 1)
                p.lpush(_p(f"tier:lat:{tier}"), int(latency_ms))
                p.ltrim(_p(f"tier:lat:{tier}"), 0, _LAT_CAP - 1)
                if tokens:
                    for k in ("prompt", "completion", "total"):
                        p.hincrby(_p(f"tier:tok:{tier}"), k,
                                  int(tokens.get(k, 0) or 0))
                _expire(p, "tier:count", "tier:errors",
                        f"tier:lat:{tier}", f"tier:tok:{tier}")
                p.execute()
                return
            except Exception:
                state_store.note_fail()
        with self._lock:
            self._tier_count[tier] += 1
            if error:
                self._tier_errors[tier] += 1
            lats = self._tier_latencies[tier]
            lats.append(latency_ms)
            if len(lats) > 1000:
                del lats[:-1000]
            if tokens:
                tt = self._tier_tokens[tier]
                tt["prompt"] += int(tokens.get("prompt", 0) or 0)
                tt["completion"] += int(tokens.get("completion", 0) or 0)
                tt["total"] += int(tokens.get("total", 0) or 0)

    # ---------------- 读取 ----------------

    def get_stats(self):
        r = state_store.get_state_redis()
        if r is not None:
            try:
                return self._stats_from_redis(r)
            except Exception:
                state_store.note_fail()
        return self._stats_from_memory()

    # ---- Redis 读取:与内存版输出形状逐字段一致 ----

    def _stats_from_redis(self, r):
        p = r.pipeline(transaction=False)
        p.get(_p("req:total"))
        p.get(_p("req:errors"))
        p.lrange(_p("req:lat"), 0, -1)
        p.hgetall(_p("tool:calls"))
        p.hgetall(_p("tool:failures"))
        p.hgetall(_p("tool:cache_hits"))
        p.hgetall(_p("tool:cat"))
        p.hgetall(_p("cat:calls"))
        p.hgetall(_p("cat:failures"))
        p.get(_p("search:hits"))
        p.get(_p("search:misses"))
        p.hgetall(_p("tokens"))
        p.hgetall(_p("internal_tokens"))
        p.hgetall(_p("per_user"))
        p.get(_p("escalations:total"))
        p.hgetall(_p("tier:count"))
        p.hgetall(_p("tier:errors"))
        p.hgetall(_p("tier:escalations"))
        p.get(_p("start"))
        (req_total, req_errors, req_lat, tool_calls, tool_failures,
         tool_cache, tool_cat, cat_calls, cat_failures, s_hits, s_misses,
         tokens, internal_tokens, per_user, esc_total, tier_count, tier_errors,
         tier_esc, start_ts) = p.execute()

        tool_calls = tool_calls or {}
        lats = sorted(int(x) for x in (req_lat or []))
        n = len(lats)
        avg = sum(lats) / n if n else 0
        p50 = lats[n // 2] if n else 0
        p95 = lats[int(n * 0.95)] if n else 0
        s_hits, s_misses = int(s_hits or 0), int(s_misses or 0)
        total_search = s_hits + s_misses
        req_total = int(req_total or 0)
        req_errors = int(req_errors or 0)
        esc_total = int(esc_total or 0)

        # 每工具/每范式明细(逐 key 拉时长列表)
        tool_durs = {name: sorted(int(x) for x in r.lrange(_p(f"tool:dur:{name}"), 0, -1))
                     for name in tool_calls}
        tier_count = tier_count or {}
        tier_lats = {t: sorted(int(x) for x in r.lrange(_p(f"tier:lat:{t}"), 0, -1))
                     for t in tier_count}
        tier_toks = {t: r.hgetall(_p(f"tier:tok:{t}")) or {} for t in tier_count}

        return {
            "uptime_seconds": int(time.time() - float(start_ts)) if start_ts
            else int(time.time() - self._start_time),
            "requests": {
                "total": req_total,
                "errors": req_errors,
                "error_rate": round(req_errors / req_total, 4) if req_total else 0,
            },
            "latency_ms": {
                "avg": round(avg),
                "p50": p50,
                "p95": p95,
            },
            "tools": self._tool_stats_r(tool_calls, tool_failures or {},
                                        tool_cache or {}, tool_cat or {}, tool_durs),
            "tool_categories": self._category_stats_r(cat_calls or {},
                                                      cat_failures or {}),
            "search": {
                "hits": s_hits,
                "misses": s_misses,
                "hit_rate": round(s_hits / total_search, 4) if total_search else 0,
            },
            "tokens": {k: int((tokens or {}).get(k, 0) or 0)
                       for k in ("prompt", "completion", "total")},
            # 记忆链内部 LLM(升迁门/摘要)分账展示,不计入上方 billable tokens。
            "internal_tokens": {k: int((internal_tokens or {}).get(k, 0) or 0)
                                for k in ("prompt", "completion", "total")},
            "per_user": {u: int(v) for u, v in (per_user or {}).items()},
            "escalations_total": esc_total,
            "escalation_rate": round(esc_total / req_total, 4) if req_total else 0,
            "by_tier": self._tier_stats_r(tier_count, tier_errors or {},
                                          tier_esc or {}, tier_lats, tier_toks),
        }

    def _tool_stats_r(self, calls, failures, cache_hits, categories, durs_map):
        out = {}
        for name, calls_n in calls.items():
            calls_n = int(calls_n)
            durs = durs_map.get(name, [])
            n = len(durs)
            f = int(failures.get(name, 0))
            out[name] = {
                "calls": calls_n,
                "failures": f,
                "success_rate": round(1 - f / calls_n, 4) if calls_n else 1,
                "category": categories.get(name),
                "cache_hits": int(cache_hits.get(name, 0)),
                "avg_duration_ms": round(sum(durs) / n) if n else 0,
                "p95_duration_ms": self._percentile(durs, 0.95),
            }
        return out

    def _category_stats_r(self, calls, failures):
        out = {}
        for cat, calls_n in calls.items():
            calls_n = int(calls_n)
            f = int(failures.get(cat, 0))
            out[cat] = {
                "calls": calls_n,
                "failures": f,
                "success_rate": round(1 - f / calls_n, 4) if calls_n else 1,
            }
        return out

    def _tier_stats_r(self, count, errors, escalations, lats_map, toks_map):
        out = {}
        for tier, cnt in count.items():
            cnt = int(cnt)
            lats = lats_map.get(tier, [])
            n = len(lats)
            e = int(errors.get(tier, 0))
            out[tier] = {
                "requests": cnt,
                "errors": e,
                "error_rate": round(e / cnt, 4) if cnt else 0,
                "escalations_from": int(escalations.get(tier, 0)),
                "latency_ms": {
                    "avg": round(sum(lats) / n) if n else 0,
                    "p50": self._percentile(lats, 0.50),
                    "p95": self._percentile(lats, 0.95),
                },
                "tokens": {k: int((toks_map.get(tier) or {}).get(k, 0) or 0)
                           for k in ("prompt", "completion", "total")},
            }
        return out

    # ---- 内存回退读取(外置前原实现)----

    def _stats_from_memory(self):
        with self._lock:
            lats = sorted(self._latencies)
            n = len(lats)
            avg = sum(lats) / n if n else 0
            p50 = lats[n // 2] if n else 0
            p95 = lats[int(n * 0.95)] if n else 0
            total_search = self._search_hits + self._search_misses

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
                "tools": self._tool_stats(),
                "tool_categories": self._category_stats(),
                "search": {
                    "hits": self._search_hits,
                    "misses": self._search_misses,
                    "hit_rate": round(self._search_hits / total_search, 4)
                        if total_search else 0,
                },
                "tokens": dict(self._tokens),
                # 记忆链内部 LLM(升迁门/摘要)分账展示,不计入上方 billable tokens。
                "internal_tokens": dict(self._internal_tokens),
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

    def _tool_stats(self):
        out = {}
        for name, calls in self._tool_calls.items():
            durs = sorted(self._tool_durations.get(name, []))
            n = len(durs)
            failures = self._tool_failures[name]
            out[name] = {
                "calls": calls,
                "failures": failures,
                "success_rate": round(1 - failures / calls, 4) if calls else 1,
                "category": self._tool_categories.get(name),
                "cache_hits": self._tool_cache_hits.get(name, 0),
                "avg_duration_ms": round(sum(durs) / n) if n else 0,
                "p95_duration_ms": self._percentile(durs, 0.95),
            }
        return out

    def _category_stats(self):
        out = {}
        for cat, calls in self._category_calls.items():
            failures = self._category_failures.get(cat, 0)
            out[cat] = {
                "calls": calls,
                "failures": failures,
                "success_rate": round(1 - failures / calls, 4) if calls else 1,
            }
        return out

    def _tier_stats(self):
        out = {}
        for tier in self._tier_count:
            lats = sorted(self._tier_latencies[tier])
            n = len(lats)
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
                "tokens": dict(self._tier_tokens[tier]),
            }
        return out


# 全局单例
metrics = Metrics()
