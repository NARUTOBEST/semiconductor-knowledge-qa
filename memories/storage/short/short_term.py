# -*- coding: utf-8 -*-
"""短期记忆层:session_events 会话流水(只追加)。

职责:审计、回放、溯源;作为升迁流水线的数据源。
不直接注入 LLM,不做语义召回。连接 SHORT_PG_URI。
"""
import json
from typing import Any, Optional

from ..connections import pg_conn


class ShortTermMemory:
    """append-only 会话事件写入与读取。"""

    def append_event(self, thread_id: str, event_type: str,
                     payload: dict[str, Any], *,
                     user_id: Optional[str] = None,
                     session_id: Optional[str] = None) -> int:
        """追加一条事件。seq 由该 thread 现有最大值 +1 自动分配。

        用 pg_advisory_xact_lock 按 thread_id 加事务级咨询锁,串行化同 thread 的
        "读 max seq + 插入",避免并发下两事务算出相同 seq 而撞 (thread_id, seq)
        唯一索引导致事件丢失。不同 thread 的锁互不阻塞。
        """
        with pg_conn("short") as (conn, cur):
            cur.execute(
                """
                SELECT pg_advisory_xact_lock(
                    hashtextextended(%s, 0)::bigint
                );
                INSERT INTO session_events
                    (thread_id, session_id, user_id, seq, event_type, payload)
                VALUES (%s, %s, %s,
                        COALESCE((SELECT max(seq) FROM session_events
                                  WHERE thread_id=%s), 0) + 1,
                        %s, %s)
                RETURNING seq;
                """,
                (thread_id,
                 thread_id, session_id, user_id, thread_id,
                 event_type, json.dumps(payload, ensure_ascii=False, default=str)),
            )
            return int(cur.fetchone()["seq"])

    # ---------- 升迁水位线(增量升迁) ----------
    def get_watermark(self, thread_id: str) -> int:
        """返回该 thread 已升迁到的最大 seq;无记录返回 0。"""
        with pg_conn("short") as (conn, cur):
            cur.execute(
                "SELECT last_seq FROM promotion_watermark WHERE thread_id = %s",
                (thread_id,),
            )
            row = cur.fetchone()
            return int(row["last_seq"]) if row else 0

    def advance_watermark(self, thread_id: str, last_seq: int) -> None:
        """把水位推进到 last_seq(只进不退,GREATEST 兜底并发)。

        成功推进即重置连续失败计数(fail_count=0)。
        """
        with pg_conn("short") as (conn, cur):
            cur.execute(
                """
                INSERT INTO promotion_watermark
                    (thread_id, last_seq, promoted_at, created_at, fail_count, last_attempt_at)
                VALUES (%s, %s, now(), now(), 0, now())
                ON CONFLICT (thread_id) DO UPDATE
                SET last_seq = GREATEST(promotion_watermark.last_seq, EXCLUDED.last_seq),
                    promoted_at = now(),
                    fail_count = 0,
                    last_attempt_at = now()
                """,
                (thread_id, int(last_seq)),
            )

    def get_watermark_state(self, thread_id: str) -> tuple[int, int]:
        """返回 (last_seq, fail_count);无记录返回 (0, 0)。"""
        with pg_conn("short") as (conn, cur):
            cur.execute(
                "SELECT last_seq, fail_count FROM promotion_watermark WHERE thread_id = %s",
                (thread_id,),
            )
            row = cur.fetchone()
            return (int(row["last_seq"]), int(row["fail_count"])) if row else (0, 0)

    def record_promotion_failure(self, thread_id: str, max_fail: int = 5) -> int:
        """升迁失败时累加 fail_count 并刷新 last_attempt_at。

        返回累加后的 fail_count。达到 max_fail 后调用方应强制推进水位(丢弃这批
        无法升迁的事件),避免永久性 LLM 故障导致 backlog 无限增长、每轮重放。
        """
        with pg_conn("short") as (conn, cur):
            cur.execute(
                """
                INSERT INTO promotion_watermark
                    (thread_id, last_seq, promoted_at, created_at, fail_count, last_attempt_at)
                VALUES (%s, 0, now(), now(), 1, now())
                ON CONFLICT (thread_id) DO UPDATE
                SET fail_count = promotion_watermark.fail_count + 1,
                    last_attempt_at = now()
                RETURNING fail_count
                """,
                (thread_id,),
            )
            return int(cur.fetchone()["fail_count"])

    def pending_promotion_threads(self, limit: int = 100) -> list[tuple[str, str]]:
        """返回有未升迁事件的 (thread_id, user_id) 列表,供启动补偿扫描。

        判定:session_events 的 max(seq) > promotion_watermark.last_seq
        (无水位记录时 last_seq 视为 0)。取该 thread 最近一条带 user_id 的事件
        作为 user_id;user_id 为空的(匿名)不返回,升迁层会跳过。
        """
        with pg_conn("short") as (conn, cur):
            cur.execute(
                """
                SELECT se.thread_id,
                       (SELECT user_id FROM session_events se2
                        WHERE se2.thread_id = se.thread_id
                          AND se2.user_id IS NOT NULL
                        ORDER BY se2.seq DESC LIMIT 1) AS user_id
                FROM (
                    SELECT thread_id, max(seq) AS max_seq
                    FROM session_events
                    GROUP BY thread_id
                ) se
                LEFT JOIN promotion_watermark pw ON pw.thread_id = se.thread_id
                WHERE se.max_seq > COALESCE(pw.last_seq, 0)
                  AND EXISTS (
                      SELECT 1 FROM session_events se3
                      WHERE se3.thread_id = se.thread_id AND se3.user_id IS NOT NULL
                  )
                ORDER BY COALESCE(pw.last_attempt_at, pw.promoted_at, to_timestamp(0)) ASC
                LIMIT %s
                """,
                (int(limit),),
            )
            return [(r["thread_id"], r["user_id"]) for r in cur.fetchall()
                    if r["user_id"]]

    def list_events_after(self, thread_id: str, after_seq: int,
                          event_types: Optional[list[str]] = None) -> list[dict]:
        """取 seq > after_seq 的事件(升序),供增量升迁。"""
        sql = "SELECT * FROM session_events WHERE thread_id=%s AND seq > %s"
        params: list[Any] = [thread_id, int(after_seq)]
        if event_types:
            sql += " AND event_type = ANY(%s)"
            params.append(event_types)
        sql += " ORDER BY seq ASC"
        with pg_conn("short") as (conn, cur):
            cur.execute(sql, params)
            return [dict(r) for r in cur.fetchall()]

    def delete_thread(self, thread_id: str) -> int:
        """删除某 thread 的全部流水 + 升迁水位(会话删除级联清理)。返回删除流水行数。"""
        with pg_conn("short") as (conn, cur):
            cur.execute(
                "DELETE FROM session_events WHERE thread_id = %s",
                (thread_id,),
            )
            deleted = cur.rowcount
            cur.execute(
                "DELETE FROM promotion_watermark WHERE thread_id = %s",
                (thread_id,),
            )
            return deleted

    def stale_threads(self, older_than_days: int) -> list[tuple[str, "datetime"]]:
        """返回最后活动早于 N 天的 (thread_id, last_active) 列表,供滚动 TTL 用。

        工作记忆 checkpoint 表无时间列,故以短期流水的最后事件时间作为线程
        "最后活动时间"(runner 每轮开始必写一条 user_message)。
        """
        with pg_conn("short") as (conn, cur):
            cur.execute(
                """
                SELECT thread_id, max(created_at) AS last_active
                FROM session_events
                GROUP BY thread_id
                HAVING max(created_at) < now() - (%s * interval '1 day')
                """,
                (int(older_than_days),),
            )
            return [(r["thread_id"], r["last_active"]) for r in cur.fetchall()]

    def delete_threads_before(self, older_than_days: int) -> int:
        """删除所有最后活动早于 N 天的流水行及其水位线。返回删除流水行数。"""
        with pg_conn("short") as (conn, cur):
            cur.execute(
                """
                DELETE FROM session_events
                WHERE thread_id IN (
                    SELECT thread_id FROM session_events
                    GROUP BY thread_id
                    HAVING max(created_at) < now() - (%s * interval '1 day')
                )
                """,
                (int(older_than_days),),
            )
            deleted = cur.rowcount
            # 清理无对应流水的孤儿水位线(含刚被删掉的过期 thread)
            cur.execute(
                """
                DELETE FROM promotion_watermark
                WHERE thread_id NOT IN (SELECT DISTINCT thread_id FROM session_events)
                """
            )
            return deleted


# 单例
short_term = ShortTermMemory()
