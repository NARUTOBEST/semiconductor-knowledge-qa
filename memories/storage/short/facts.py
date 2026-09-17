# -*- coding: utf-8 -*-
"""短期记忆【事实表】(Redis,前缀 memf:):存"可查询的记忆事实",与事件流水分离。

与 ``short_term.py``(mem:* append-only 事件流水,审计/回放/近期对话召回用)职责分离、
不重复存储:本表存【去重后的记忆事实条目】(本轮用户问题 + 助手答案提炼),供:
  - 语义/指纹去重(重复问答只累加频次、刷新时间,不重复入库);
  - 升迁门判定后标记 promoted(已升迁长期 PG);
  - 容量受限时按"重要度 + 最近访问"淘汰。

键设计(与 mem:* 同实例、互不冲突):
  - memf:cnt:{tid}        STRING  事实 seq 自增(INCR)
  - memf:facts:{tid}      ZSET    member=fid,score=淘汰分(重要度*1000 + 最近访问小时数;
                                  promoted 已升迁长期的减惩罚分,优先淘汰)
  - memf:fact:{tid}:{fid} HASH    {q,a,fp,importance,freq,created_ts,last_access_ts,
                                  promoted,long_ref,degraded}
  - memf:idx:fp:{tid}     HASH    fp_norm -> fid(规范化指纹精确去重 O(1))
  - memf:keys:{tid}       SET     该 thread 全部事实键(级联删除批量 DEL,避免 SCAN)
  - memf:user:{username}  SET     该用户名下全部 thread(账号注销枚举)

TTL = MEM_FACT_TTL_SECONDS(默认 24h),写时续期,Redis 自身兜底过期;容量上限
MEM_FACT_MAX_PER_THREAD(默认 10000),超限 ZPOPMIN 淘汰。thread 键统一为
scoped_thread_id = f"{username}|{tid}"。

注:语义(embedding)去重默认关闭(MEM_FACT_SEMANTIC_DEDUP=0),用规范化指纹;
跨会话长期去重本就由 PG pgvector 负责,短期表按会话隔离,语义去重价值有限。
旁路:Redis 不可用时调用方降级,本模块不吞连接异常以外的逻辑错误。
"""
from __future__ import annotations

import hashlib
import logging
import re
import time
from typing import Any, Optional

from ..connections import get_redis

logger = logging.getLogger("agent")

# 答案/问题入库截断(控制单条 HASH 体积)
_Q_MAX = 500
_A_MAX = 2000

_PUNCT_RE = re.compile(r"[\s　，。！？；：、,.!?;:\"'（）()\[\]【】<>《》…\-_/\\]+")


def fingerprint(q: str, a: str) -> str:
    """规范化指纹:lowercase + 去空白/标点 + 截断 -> SHA1。用于精确去重。"""
    def norm(s: str) -> str:
        s = (s or "").strip().lower()
        s = _PUNCT_RE.sub("", s)
        return s[:300]
    raw = norm(q) + "|" + norm(a)
    return hashlib.sha1(raw.encode("utf-8", "ignore")).hexdigest()


class FactTable:
    """短期记忆事实表(Redis 后端)。"""

    def _client(self):
        client = get_redis()
        if client is None:
            raise RuntimeError("Redis 不可用(redis 包缺失或未初始化)")
        return client

    # ---- 键 ----
    def _cnt(self, tid): return f"memf:cnt:{tid}"
    def _facts(self, tid): return f"memf:facts:{tid}"
    def _fact(self, tid, fid): return f"memf:fact:{tid}:{fid}"
    def _idx(self, tid): return f"memf:idx:fp:{tid}"
    def _keys(self, tid): return f"memf:keys:{tid}"
    def _user(self, u): return f"memf:user:{u}"

    @staticmethod
    def _user_from_thread(tid: str) -> Optional[str]:
        return tid.split("|", 1)[0] if "|" in tid else None

    def _ttl(self) -> int:
        import config as C
        return max(60, int(getattr(C, "MEM_FACT_TTL_SECONDS", 86400)))

    def _max_per_thread(self) -> int:
        import config as C
        return max(10, int(getattr(C, "MEM_FACT_MAX_PER_THREAD", 10000)))

    def _expire(self, pipe, tid: str, *extra: str) -> None:
        ttl = self._ttl()
        if ttl <= 0:
            return
        for k in (self._cnt(tid), self._facts(tid), self._idx(tid),
                  self._keys(tid), *extra):
            pipe.expire(k, ttl)

    # ---- 淘汰分 ----
    def _score(self, importance: float, last_access_ts: float, promoted: bool) -> float:
        # 重要度*1000 + 最近访问(绝对小时数,越新越大);promoted 减惩罚 -> 优先被淘汰。
        score = float(importance) * 1000.0 + last_access_ts / 3600.0
        if promoted:
            score -= 5000.0   # 已升迁长期(长期有副本),短期优先淘汰
        return score

    # ---- 写入 / 去重 ----
    def add_or_touch(self, thread_id: str, *, q: str, a: str,
                     importance: float = 0.5, degraded: bool = False,
                     user_id: Optional[str] = None,
                     fp: Optional[str] = None) -> dict[str, Any]:
        """去重写入一条事实。返回 {"fid": str, "duplicate": bool, "promoted": bool}。

        - degraded=True(熔断/降级路径):跳过去重,无条件裸写一条(degraded=1)。
        - 否则按规范化指纹查 memf:idx:fp:命中 -> HINCRBY freq + 刷新 last_access/淘汰分,
          不新增;未命中 -> 新建事实条目并登记索引/有序集。
        写入后按容量上限淘汰低分条目。
        """
        client = self._client()
        now = time.time()
        fp_norm = fp or fingerprint(q, a)

        # 降级:跳过去重,直接裸写
        if degraded:
            return self._insert(client, thread_id, q, a, fp_norm,
                                importance=0.3, now=now, degraded=1,
                                user_id=user_id)

        # 精确指纹去重
        existing = client.hget(self._idx(thread_id), fp_norm)
        if existing:
            fid = existing
            client.hincrby(self._fact(thread_id, fid), "freq", 1)
            client.hset(self._fact(thread_id, fid), "last_access_ts", f"{now:.3f}")
            promoted = client.hget(self._fact(thread_id, fid), "promoted") == "1"
            client.zadd(self._facts(thread_id),
                        {fid: self._score(importance, now, promoted)})
            pipe = client.pipeline(transaction=False)
            self._expire(pipe, thread_id, self._fact(thread_id, fid))
            pipe.execute()
            return {"fid": fid, "duplicate": True, "promoted": promoted}

        return self._insert(client, thread_id, q, a, fp_norm,
                            importance=importance, now=now, degraded=0,
                            user_id=user_id)

    def touch_if_exists(self, thread_id: str, fp: str) -> Optional[str]:
        """只查重复并刷新频次/最近访问,【不新建】。命中返回 fid,未命中返回 None。

        供 memory-loop 沉淀节点:先查重(重复问答跳过升迁门 LLM),未命中再走
        升迁门、成功后才 add_or_touch 新建——避免"先插入、LLM 失败、重试被判重复"。
        """
        client = self._client()
        fid = client.hget(self._idx(thread_id), fp)
        if not fid:
            return None
        fact_key = self._fact(thread_id, fid)
        now = time.time()
        client.hincrby(fact_key, "freq", 1)
        client.hset(fact_key, "last_access_ts", f"{now:.3f}")
        promoted = client.hget(fact_key, "promoted") == "1"
        imp = 0.5
        try:
            imp = float(client.hget(fact_key, "importance") or 0.5)
        except (TypeError, ValueError):
            pass
        client.zadd(self._facts(thread_id),
                    {fid: self._score(imp, now, promoted)})
        pipe = client.pipeline(transaction=False)
        self._expire(pipe, thread_id, fact_key)
        pipe.execute()
        return fid

    def _insert(self, client, thread_id, q, a, fp_norm, *,
                importance, now, degraded, user_id) -> dict[str, Any]:
        seq = int(client.incr(self._cnt(thread_id)))
        fid = f"f{seq}"
        fact_key = self._fact(thread_id, fid)
        pipe = client.pipeline(transaction=True)
        pipe.hset(fact_key, mapping={
            "q": (q or "")[:_Q_MAX],
            "a": (a or "")[:_A_MAX],
            "fp": fp_norm,
            "importance": f"{float(importance):.3f}",
            "freq": "1",
            "created_ts": f"{now:.3f}",
            "last_access_ts": f"{now:.3f}",
            "promoted": "0",
            "long_ref": "",
            "degraded": str(degraded),
        })
        pipe.hset(self._idx(thread_id), fp_norm, fid)
        pipe.zadd(self._facts(thread_id),
                  {fid: self._score(importance, now, False)})
        pipe.sadd(self._keys(thread_id), fact_key)
        username = self._user_from_thread(thread_id) or user_id
        if username:
            pipe.sadd(self._user(username), thread_id)
        self._expire(pipe, thread_id, fact_key)
        if username:
            pipe.expire(self._user(username), self._ttl())
        pipe.execute()

        self._evict_if_needed(client, thread_id)
        return {"fid": fid, "duplicate": False, "promoted": False}

    def mark_promoted(self, thread_id: str, fid: str, long_ref: str = "") -> None:
        """标记事实已升迁长期记忆;降低其淘汰分(长期已有副本,短期可优先淘汰)。"""
        client = self._client()
        fact_key = self._fact(thread_id, fid)
        if not client.exists(fact_key):
            return
        client.hset(fact_key, mapping={"promoted": "1", "long_ref": long_ref or ""})
        imp = 0.5
        try:
            imp = float(client.hget(fact_key, "importance") or 0.5)
        except (TypeError, ValueError):
            pass
        last = time.time()
        try:
            last = float(client.hget(fact_key, "last_access_ts") or last)
        except (TypeError, ValueError):
            pass
        client.zadd(self._facts(thread_id),
                    {fid: self._score(imp, last, True)})

    # ---- 容量淘汰 ----
    def _evict_if_needed(self, client, thread_id: str) -> int:
        """超容量时 ZPOPMIN 淘汰到上限;返回淘汰条数。"""
        cap = self._max_per_thread()
        evicted = 0
        for _ in range(cap + 1):
            n = client.zcard(self._facts(thread_id))
            if n <= cap:
                break
            popped = client.zpopmin(self._facts(thread_id), 1)
            if not popped:
                break
            fid = popped[0][0]
            fact_key = self._fact(thread_id, fid)
            fp = client.hget(fact_key, "fp")
            pipe = client.pipeline(transaction=True)
            pipe.delete(fact_key)
            if fp:
                pipe.hdel(self._idx(thread_id), fp)
            pipe.srem(self._keys(thread_id), fact_key)
            pipe.execute()
            evicted += 1
        return evicted

    # ---- 计数 / 查询(测试与观测)----
    def count(self, thread_id: str) -> int:
        return int(self._client().zcard(self._facts(thread_id)))

    def get(self, thread_id: str, fid: str) -> dict[str, str]:
        return self._client().hgetall(self._fact(thread_id, fid))

    # ---- 删除 / 级联 ----
    def delete_thread(self, thread_id: str) -> int:
        """删除某 thread 全部事实(会话删除级联)。返回删除事实条数。"""
        client = self._client()
        fact_keys = list(client.smembers(self._keys(thread_id)))
        deleted = len(fact_keys)
        pipe = client.pipeline(transaction=True)
        if fact_keys:
            pipe.delete(*fact_keys)
        pipe.delete(self._facts(thread_id), self._idx(thread_id),
                    self._cnt(thread_id), self._keys(thread_id))
        username = self._user_from_thread(thread_id)
        if username:
            pipe.srem(self._user(username), thread_id)
        pipe.execute()
        return deleted

    def delete_user(self, username: str) -> dict[str, int]:
        """删除某用户名下全部事实表数据(账号注销级联)。"""
        client = self._client()
        tids = list(client.smembers(self._user(username)))
        total = 0
        for tid in tids:
            try:
                total += self.delete_thread(tid)
            except Exception as e:  # noqa: BLE001
                logger.warning("fact delete_user: thread %s 失败: %s: %s",
                               tid, type(e).__name__, e)
        client.delete(self._user(username))
        return {"facts": total}


# 单例
fact_table = FactTable()
