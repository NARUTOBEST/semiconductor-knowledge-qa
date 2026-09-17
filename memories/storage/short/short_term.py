# -*- coding: utf-8 -*-
"""短期记忆层:session_events 会话流水(只追加),Redis 实现。

职责:审计、回放、溯源。不直接注入 LLM,不做语义召回。

键设计(与工作记忆 checkpoint 的 checkpoint:* 前缀同实例、互不冲突):
  - mem:cnt:{tid}            STRING  自增 seq(INCR 原子分配,替代 PG advisory lock)
  - mem:evt:{tid}:{seq}      HASH    {type, payload(JSON), ts(epoch), sid, uid}
  - mem:evts:{tid}           SET     该 thread 的全部事件键(供整线程删除/计数)
  - mem:threads              ZSET    member=tid,score=最后事件 ts(供滚动 TTL 枚举)
  - mem:user:{username}      SET     该用户名下全部 thread 键(供账号注销枚举)

thread 键统一为 scoped_thread_id(tid, username)= f"{username}|{tid}",
故用户名取 thread_id 中 "|" 前缀;无 "|" 的(部分测试直连)不建用户索引。

所有事件/索引键按 MEMORY_TTL_DAYS 设 EXPIRE(写时续期),Redis 自身兜底过期;
prune_inactive 仍主动清理超期线程(与原 PG 行为一致)。

方法签名与原 PG 版完全一致,上层 lifecycle/events 零改动。
"""
import json
import logging
import time
import uuid
from typing import Any, Optional

from ..connections import get_redis, memory_ttl_seconds
from . import event_spool as _spool

logger = logging.getLogger("agent")

_THREADS_INDEX = "mem:threads"
_TTL_REFRESH = True
# WAL 重放时 processing 标记超过该秒数视为上次崩溃残留,允许重新写入
_REPLAY_STALE_SECONDS = 120.0


def _user_from_thread(thread_id: str) -> Optional[str]:
    """thread_id 形如 'username|conv-id';取 '|' 前的用户名,无则 None。"""
    if "|" in thread_id:
        return thread_id.split("|", 1)[0]
    return None


class ShortTermMemory:
    """append-only 会话事件写入与读取(Redis 后端)。"""

    # ---- 内部小工具 ----
    def _client(self):
        client = get_redis()
        if client is None:
            raise RuntimeError("Redis 不可用(redis 包缺失或未初始化)")
        return client

    def _cnt_key(self, tid: str) -> str:
        return f"mem:cnt:{tid}"

    def _evts_key(self, tid: str) -> str:
        return f"mem:evts:{tid}"

    def _evt_key(self, tid: str, seq: int) -> str:
        return f"mem:evt:{tid}:{seq}"

    def _user_key(self, username: str) -> str:
        return f"mem:user:{username}"

    def _expire(self, pipe, *keys: str) -> None:
        ttl = memory_ttl_seconds()
        if ttl <= 0:
            return
        for k in keys:
            pipe.expire(k, ttl)

    # ---- 写入 ----
    def append_event(self, thread_id: str, event_type: str,
                     payload: dict[str, Any], *,
                     user_id: Optional[str] = None,
                     session_id: Optional[str] = None) -> int:
        """追加一条事件。seq 由 INCR 原子分配(替代 PG pg_advisory_xact_lock)。

        写事件 hash + 登记到 per-thread 事件集合 / 全局线程索引(score=now)/
        用户索引,并续期 TTL。返回新 seq。

        Redis 直写【失败】(不可达/半挂)时不抛异常:落本地 WAL(event_spool)暂存,
        返回 -1(待回填标记),由记忆链兜底节点在 Redis 恢复后幂等 drain 回来,
        保证对话流水不丢。在线成功路径行为与原先完全一致。
        """
        try:
            client = self._client()
            return self._write_live(
                client, thread_id, event_type, payload,
                user_id=user_id, session_id=session_id)
        except Exception as e:  # noqa: BLE001  Redis 不可达:本地 WAL 兜底
            logger.warning("short_term live write failed, spool locally: %s: %s",
                           type(e).__name__, str(e)[:120])
            return self._spool_fallback(
                thread_id, event_type, payload,
                user_id=user_id, session_id=session_id)

    def _write_live(self, client, thread_id: str, event_type: str,
                    payload: dict[str, Any], *,
                    user_id: Optional[str] = None,
                    session_id: Optional[str] = None,
                    now: Optional[float] = None) -> int:
        """直写 Redis(在线路径与 WAL 回填共用)。成功返回新 seq。"""
        now = time.time() if now is None else float(now)
        cnt_key = self._cnt_key(thread_id)
        # INCR 原子取号;首次创建时计数器无 TTL,下方 pipeline 统一续期
        seq = int(client.incr(cnt_key))
        evt_key = self._evt_key(thread_id, seq)
        evts_key = self._evts_key(thread_id)

        pipe = client.pipeline(transaction=True)
        pipe.hset(evt_key, mapping={
            "type": event_type,
            "payload": json.dumps(payload, ensure_ascii=False, default=str),
            "ts": f"{now:.3f}",
            "sid": session_id or "",
            "uid": user_id or "",
        })
        pipe.sadd(evts_key, evt_key)
        pipe.zadd(_THREADS_INDEX, {thread_id: now})
        username = _user_from_thread(thread_id)
        if username:
            pipe.sadd(self._user_key(username), thread_id)
        # 续期:事件 / per-thread 事件集 / 计数器 / 用户索引
        expire_keys = [evt_key, evts_key, cnt_key]
        if username:
            expire_keys.append(self._user_key(username))
        self._expire(pipe, *expire_keys)
        pipe.execute()
        return seq

    def _spool_fallback(self, thread_id: str, event_type: str,
                        payload: dict[str, Any], *,
                        user_id: Optional[str] = None,
                        session_id: Optional[str] = None) -> int:
        """Redis 直写失败:把事件落本地 WAL。返回 -1(待回填)。"""
        record = {
            "eid": uuid.uuid4().hex,
            "tid": thread_id,
            "etype": event_type,
            "payload": payload or {},
            "uid": user_id or "",
            "sid": session_id or "",
            "ts": time.time(),
        }
        _spool.append_record(record)
        return -1

    def _mark_set(self, client, mark: str, value: str, ttl: int, *, nx: bool):
        """写幂等标记;ttl<=0 不过期,nx=True 时仅占位(成功 True/已存在 None)。"""
        kwargs = {"nx": True} if nx else {}
        if ttl and ttl > 0:
            kwargs["ex"] = ttl
        return client.set(mark, value, **kwargs)

    def replay_record(self, rec: dict) -> bool:
        """把一条 WAL 记录【幂等】写回 Redis(drain 回调)。

        两阶段标记 mem:eid:{eid}:不存在->processing->写事件->done;
        已 done 视为成功(幂等跳过、从 WAL 移除);processing 未陈旧则留下轮
        (防并发/崩溃中途)。任何 Redis 异常向上抛,由 drain 记为失败并保留。
        """
        client = self._client()
        tid = rec.get("tid")
        etype = rec.get("etype")
        eid = rec.get("eid")
        if not tid or not etype or not eid:
            return True  # 非法记录直接丢弃
        ttl = memory_ttl_seconds()
        mark = f"mem:eid:{eid}"
        state = client.get(mark)
        if isinstance(state, bytes):
            state = state.decode("utf-8", "ignore")
        if state == "done":
            return True  # 之前已成功落库,幂等跳过
        now = time.time()
        if isinstance(state, str) and state.startswith("processing:"):
            try:
                ts = float(state.split(":", 1)[1])
            except Exception:  # noqa: BLE001
                ts = 0.0
            if now - ts < _REPLAY_STALE_SECONDS:
                return False  # 可能正在写入,留下轮
        # 占位:state 为 None 且 NX 抢不到(并发),保留;陈旧 processing 允许覆盖重写
        claimed = self._mark_set(client, mark, f"processing:{now}", ttl, nx=True)
        if not claimed and state is None:
            return False
        self._write_live(
            client, tid, etype, rec.get("payload") or {},
            user_id=rec.get("uid") or None,
            session_id=rec.get("sid") or None,
            now=rec.get("ts"))
        self._mark_set(client, mark, "done", ttl, nx=False)
        return True

    def drain_spooled(self, *, max_items: int = None,
                      time_budget_s: float = None) -> dict:
        """把本地 WAL 暂存的流水幂等推回 Redis。无积压/Redis 不可用时快速返回。"""
        return _spool.drain(self.replay_record,
                            max_items=max_items, time_budget_s=time_budget_s)

    # ---- 删除 ----
    def delete_thread(self, thread_id: str) -> int:
        """删除某 thread 的全部流水(会话删除级联清理)。返回删除事件数。"""
        client = self._client()
        evts_key = self._evts_key(thread_id)
        evt_keys = list(client.smembers(evts_key))
        deleted = len(evt_keys)
        pipe = client.pipeline(transaction=True)
        if evt_keys:
            pipe.delete(*evt_keys)
        pipe.delete(evts_key, self._cnt_key(thread_id))
        pipe.zrem(_THREADS_INDEX, thread_id)
        username = _user_from_thread(thread_id)
        if username:
            pipe.srem(self._user_key(username), thread_id)
        pipe.execute()
        return deleted

    def delete_user(self, username: str) -> dict[str, int]:
        """删除某用户名下(命名空间 username|)全部短期流水(账号注销级联)。

        从 mem:user:{username} 枚举该用户全部 thread,逐个删,再删用户索引。
        返回 {"events": 删除事件数}。
        """
        client = self._client()
        tids = list(client.smembers(self._user_key(username)))
        total = 0
        for tid in tids:
            try:
                total += self.delete_thread(tid)
            except Exception as e:
                logger.warning("delete_user: 删除 thread %s 失败: %s: %s",
                               tid, type(e).__name__, e)
        client.delete(self._user_key(username))
        return {"events": total}

    def current_seq(self, thread_id: str) -> int:
        """返回该 thread 流水当前高水位 seq(计数器值);无流水/Redis 不可用返回 0。

        供摘要/compact 落盘时记录"已覆盖到哪条流水",与摘要游标同一处写入,
        使 recent 窗口与摘要严格互斥(游标前只进摘要、游标后才进 recent)。
        """
        try:
            client = self._client()
            raw = client.get(self._cnt_key(thread_id))
            return int(raw) if raw else 0
        except Exception:  # noqa: BLE001
            return 0

    # ---- 读取 / 枚举 ----
    def recent_dialogue(self, thread_id: str,
                        limit: Optional[int] = None,
                        *, since_seq: int = 0) -> list[dict[str, str]]:
        """按时间正序返回该 thread 的 user/assistant 问答。

        从短期流水(会话事件)里取 ``user_message`` / ``assistant_message`` 两类事件,
        跳过 tool_call/tool_result 等过程事件,返回 [{"role": "user"/"assistant",
        "content": ...}]。

        :param limit: 最多返回条数;``None``(默认)表示返回【本会话全部】短期对话
                      (在 TTL 留存范围内),传正整数则只取最近 ``limit`` 条。
        :param since_seq: 游标高水位;只返回 seq > since_seq 的事件(Req1 游标互斥)。
                      默认 0 = 不设下界(兼容旧调用)。

        实现:以计数器当前 seq 为上界,倒序分批回读事件 hash(过程事件夹杂,故按批
        扫描),收够 ``limit`` 条或耗尽(到 since_seq/seq=1)即止,再按 seq 升序返回。
        Redis 不可用 / 无流水时返回 [](调用方降级,不阻断)。
        """
        client = self._client()
        max_raw = client.get(self._cnt_key(thread_id))
        if not max_raw:
            return []
        try:
            max_seq = int(max_raw)
        except (TypeError, ValueError):
            return []
        since_seq = max(0, int(since_seq or 0))

        want_all = not limit or limit <= 0
        collected: list[tuple[int, str, str]] = []  # (seq, role, content)
        cursor = max_seq
        BATCH = 60  # 每批回读的 seq 数(一轮问答夹杂若干 tool 事件,60 足够覆盖数轮)
        while cursor > since_seq and (want_all or len(collected) < limit):
            lo = max(since_seq + 1, cursor - BATCH + 1)
            seqs = list(range(lo, cursor + 1))
            pipe = client.pipeline(transaction=False)
            for s in seqs:
                pipe.hgetall(self._evt_key(thread_id, s))
            hashes = pipe.execute()
            for s, h in zip(seqs, hashes):
                if s <= since_seq or not h:
                    continue
                etype = h.get("type")
                if etype not in ("user_message", "assistant_message"):
                    continue
                try:
                    payload = json.loads(h.get("payload") or "{}")
                except (TypeError, ValueError):
                    payload = {}
                content = (payload.get("content") or "").strip()
                if not content:
                    continue
                role = "user" if etype == "user_message" else "assistant"
                collected.append((s, role, content))
            cursor = lo - 1

        collected.sort(key=lambda x: x[0])
        rows = collected if want_all else collected[-limit:]
        return [{"role": role, "content": content}
                for _s, role, content in rows]

    def list_user_threads(self, username: str) -> list[str]:
        """返回某用户名下(命名空间 username|)去重后的全部 thread 键。

        用于账号注销时定位工作记忆 checkpoint。
        """
        client = self._client()
        return list(client.smembers(self._user_key(username)))

    def stale_threads(self, older_than_days: int) -> list[tuple[str, float]]:
        """返回最后活动早于 N 天的 (thread_id, last_active_epoch) 列表,供滚动 TTL。

        以 mem:threads ZSET 的 score(最后事件 ts)判定;工作记忆 checkpoint 无时间列,
        同样以短期流水最后事件时间作为线程"最后活动时间"。
        """
        client = self._client()
        cutoff = time.time() - older_than_days * 86400
        items = client.zrangebyscore(_THREADS_INDEX, "-inf", cutoff, withscores=True)
        return [(tid, float(score)) for tid, score in items]

    def delete_threads_before(self, older_than_days: int) -> int:
        """删除所有最后活动早于 N 天的线程流水。返回删除事件总数。"""
        total = 0
        for tid, _score in self.stale_threads(older_than_days):
            try:
                total += self.delete_thread(tid)
            except Exception as e:
                logger.warning("delete_threads_before: 删除 %s 失败: %s: %s",
                               tid, type(e).__name__, e)
        return total


# 单例
short_term = ShortTermMemory()
