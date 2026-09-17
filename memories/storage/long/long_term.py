# -*- coding: utf-8 -*-
"""长期记忆 DAO:用户偏好的结构化存储 + pgvector 语义召回(按 user 哈希分表)。

设计要点:
- 分片:同一 username 恒定落 table_for(username) 这张 long_mem_XX,召回/去重/注销
  只打一张表(注销时兜底遍历全部分片,防止历史上 SHARD_COUNT 变更导致残留)。
- 去重:有 key 的结构化偏好用 UNIQUE(username,category,key) + ON CONFLICT 更新;
  无 key 的语义事实用向量余弦近邻(<=>)判重,相似度达 LONG_MEM_DUP_COSINE 视为同一条。
- 向量参数用 pgvector 文本字面量('[...]'::vector)传入,不依赖 numpy/register_vector
  的参数适配器;召回只读距离标量、不读向量列,故无需向量 loader。
- 全程 best-effort:PG 不可用 / 任何异常都吞掉记日志并返回安全默认值(记忆是旁路)。
"""
from __future__ import annotations

import logging
import os
import sys
import time
from typing import Optional

_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
for _p in (os.path.join(_PROJECT_ROOT, "config"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import config as C  # noqa: E402

from . import pg as _pg  # noqa: E402
from .sharding import table_for, all_shard_tables  # noqa: E402

logger = logging.getLogger("agent")

_SCHEMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema.sql")
_SHARD_MARKER = "-- @@SHARD_TABLE@@"


def _vec_literal(vec) -> str:
    """把 list[float] 向量转成 pgvector 文本字面量 '[v1,v2,...]'(配合 ::vector)。"""
    return "[" + ",".join(f"{float(x):.6f}" for x in vec) + "]"


class LongTermMemory:
    """长期偏好读写。所有方法失败降级,不抛异常。"""

    # ---------------- 建表 ----------------
    def setup(self) -> bool:
        """幂等建表(扩展 + user_profile + 全部分片表 + 索引)。成功返回 True。"""
        conn = _pg.get_pg(force=True)
        if conn is None:
            logger.warning("long-memory setup skipped: PG unavailable")
            return False
        try:
            with open(_SCHEMA_PATH, encoding="utf-8") as f:
                sql = f.read()
            static, _, template = sql.partition(_SHARD_MARKER)
            with conn.cursor() as cur:
                cur.execute(static)
                for name in all_shard_tables():
                    # 表名来自程序自身分片枚举(long_mem_XX),非用户输入,直接替换安全
                    cur.execute(template.format(table=name))
            logger.info("long-memory schema ready: %d shard tables", len(all_shard_tables()))
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning("long-memory setup failed: %s: %s",
                           type(e).__name__, str(e)[:200])
            return False

    # ---------------- 写入 ----------------
    def upsert_memory(self, username: str, *, category: str, key, content: str,
                      embedding=None, importance: float = 0.5,
                      source_thread: str = None) -> None:
        """写入/更新一条偏好。key 非空走结构化去重;key 为空走向量近邻去重。"""
        if not username or not content:
            return
        conn = _pg.get_pg()
        if conn is None:
            return
        table = table_for(username)
        try:
            with conn.cursor() as cur:
                if key:
                    cur.execute(
                        f"""
                        INSERT INTO {table}
                            (username, category, key, content, embedding,
                             importance, source_thread, status, updated_at)
                        VALUES (%s, %s, %s, %s, %s::vector, %s, %s, 'active', now())
                        ON CONFLICT (username, category, key) WHERE key IS NOT NULL
                        DO UPDATE SET
                            content = EXCLUDED.content,
                            embedding = COALESCE(EXCLUDED.embedding, {table}.embedding),
                            importance = EXCLUDED.importance,
                            source_thread = EXCLUDED.source_thread,
                            status = 'active', updated_at = now()
                        """,
                        (username, category, key, content,
                         _vec_literal(embedding) if embedding else None,
                         float(importance or 0.5), source_thread),
                    )
                else:
                    if not self._dedup_semantic(cur, table, username, category,
                                                content, embedding, importance,
                                                source_thread):
                        cur.execute(
                            f"""
                            INSERT INTO {table}
                                (username, category, key, content, embedding,
                                 importance, source_thread, status)
                            VALUES (%s, %s, NULL, %s, %s::vector, %s, %s, 'active')
                            """,
                            (username, category, content,
                             _vec_literal(embedding) if embedding else None,
                             float(importance or 0.5), source_thread),
                        )
        except Exception as e:  # noqa: BLE001
            logger.info("long-memory upsert skipped: %s: %s",
                        type(e).__name__, str(e)[:160])

    def _dedup_semantic(self, cur, table, username, category, content,
                        embedding, importance, source_thread) -> bool:
        """无 key 偏好:向量近邻判重。命中近邻则更新,返回 True;否则返回 False。"""
        if not embedding:
            return False
        threshold_dist = 1.0 - float(getattr(C, "LONG_MEM_DUP_COSINE", 0.9))
        cur.execute(
            f"""
            SELECT id FROM {table}
            WHERE username = %s AND category = %s AND status = 'active'
                  AND embedding IS NOT NULL
            ORDER BY embedding <=> %s::vector
            LIMIT 1
            """,
            (username, category, _vec_literal(embedding)),
        )
        row = cur.fetchone()
        if not row:
            return False
        # 近邻距离需复查(ORDER BY 取最近一条,再判阈值)
        cur.execute(
            f"SELECT embedding <=> %s::vector FROM {table} WHERE id = %s",
            (_vec_literal(embedding), row[0]),
        )
        dist = cur.fetchone()[0]
        if float(dist) > threshold_dist:
            return False  # 不够相似 -> 作为新条目插入
        cur.execute(
            f"""
            UPDATE {table}
            SET content = %s, embedding = %s::vector, importance = %s,
                source_thread = COALESCE(%s, source_thread),
                status = 'active', updated_at = now()
            WHERE id = %s
            """,
            (content, _vec_literal(embedding), float(importance or 0.5),
             source_thread, row[0]),
        )
        return True

    # ---------------- 召回 ----------------
    def search_relevant(self, username: str, query_embedding, k: int = None):
        """按问题向量语义召回该用户的 active 偏好(余弦近邻),返回 list[dict]。"""
        if not username or not query_embedding:
            return []
        conn = _pg.get_pg()
        if conn is None:
            return []
        table = table_for(username)
        k = int(k or getattr(C, "LONG_MEM_TOP_K", 5))
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT id, category, key, content, importance,
                           (embedding <=> %s::vector) AS distance
                    FROM {table}
                    WHERE username = %s AND status = 'active' AND embedding IS NOT NULL
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (_vec_literal(query_embedding), username,
                     _vec_literal(query_embedding), k),
                )
                rows = cur.fetchall()
                hits = [{"id": r[0], "category": r[1], "key": r[2],
                         "content": r[3], "importance": float(r[4] or 0.5),
                         "distance": float(r[5])} for r in rows]
                # 热度更新(best-effort,不影响召回结果)
                if hits:
                    ids = tuple(h["id"] for h in hits)
                    cur.execute(
                        f"UPDATE {table} SET hit_count = hit_count + 1, last_hit_at = now() "
                        f"WHERE id = ANY(%s)", (list(ids),))
                return hits
        except Exception as e:  # noqa: BLE001
            logger.info("long-memory search skipped: %s: %s",
                        type(e).__name__, str(e)[:160])
            return []

    # ---------------- 缺失向量回填(检索模型故障期 NULL 向量补救)----------------
    def backfill_missing_embeddings(self, embed_fn, *, limit: int = 8,
                                    time_budget_s: Optional[float] = None) -> dict:
        """把以 embedding IS NULL 落库的 active 语义事实重新向量化补回。

        背景:检索模型(8002)故障期,抽取写入 embed_texts 返回 None,无 key 的语义
        事实会以 embedding=NULL 插入,而召回 SQL 要求 embedding IS NOT NULL —— 这些
        条目"存了却永远召不回",模型恢复后也不会自动补。本方法由记忆链兜底节点
        每轮有界推进:扫 NULL -> 批量 embed -> UPDATE。

        :param embed_fn: texts(list[str]) -> list[vector];模型不可用/熔断冷却时返回 None。
        :param limit: 单次最多回填条数(有界,默认 8)。
        :returns: {"candidate","filled","remain","skipped"}。best-effort,不抛异常。
        """
        out = {"candidate": 0, "filled": 0, "remain": 0, "skipped": 0}
        conn = _pg.get_pg()
        if conn is None:
            return out
        start = time.time()
        # 1) 跨分片收集候选(表名为程序枚举的分片名,非外部输入,安全)
        candidates: list[tuple] = []
        try:
            for table in all_shard_tables():
                with conn.cursor() as cur:
                    cur.execute(
                        f"""
                        SELECT id, content FROM {table}
                        WHERE status='active' AND embedding IS NULL
                              AND content IS NOT NULL AND content <> ''
                        ORDER BY id LIMIT %s
                        """, (limit,))
                    for row in cur.fetchall():
                        candidates.append((table, row[0], row[1]))
                if len(candidates) >= limit:
                    break
        except Exception as e:  # noqa: BLE001
            logger.info("long-memory backfill scan skipped: %s: %s",
                        type(e).__name__, str(e)[:160])
            return out
        out["candidate"] = len(candidates)
        if not candidates:
            return out
        batch = candidates[:limit]
        # 2) 批量嵌入(一次调用);模型不可用/冷却 -> None,本轮跳过、保留 NULL 下轮再试
        try:
            vecs = embed_fn([c for _t, _i, c in batch])
        except Exception as e:  # noqa: BLE001
            logger.info("long-memory backfill embed skipped: %s: %s",
                        type(e).__name__, str(e)[:160])
            vecs = None
        if not vecs:
            out["remain"] = len(candidates)
            out["skipped"] = len(batch)
            return out
        # 3) 逐条更新(带 IS NULL 条件,不覆盖期间新写入的向量)
        for (table, mid, _content), vec in zip(batch, vecs):
            try:
                if time_budget_s and time.time() - start > time_budget_s:
                    out["skipped"] += 1
                    continue
                if not vec:
                    out["skipped"] += 1
                    continue
                with conn.cursor() as cur:
                    cur.execute(
                        f"""UPDATE {table} SET embedding=%s::vector, updated_at=now()
                            WHERE id=%s AND embedding IS NULL""",
                        (_vec_literal(vec), mid))
                    out["filled"] += 1 if cur.rowcount else 0
            except Exception as e:  # noqa: BLE001
                logger.info("long-memory backfill update skipped: %s: %s",
                            type(e).__name__, str(e)[:120])
                out["skipped"] += 1
        out["remain"] = max(0, out["candidate"] - out["filled"])
        if out["filled"]:
            logger.info("long-memory backfill filled %d (candidate=%d)",
                        out["filled"], out["candidate"])
        return out

    # ---------------- 画像 ----------------
    def get_profile(self, username: str):
        """返回用户画像 dict(display_prefs/summary/top_interests/fact_count),无则 None。"""
        if not username:
            return None
        conn = _pg.get_pg()
        if conn is None:
            return None
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT display_prefs, summary, top_interests, fact_count
                    FROM user_profile WHERE username = %s
                    """,
                    (username,),
                )
                row = cur.fetchone()
                if not row:
                    return None
                return {"display_prefs": row[0] or {}, "summary": row[1] or "",
                        "top_interests": row[2] or [], "fact_count": int(row[3] or 0)}
        except Exception as e:  # noqa: BLE001
            logger.info("long-memory get_profile skipped: %s: %s",
                        type(e).__name__, str(e)[:160])
            return None

    def upsert_profile(self, username: str, *, display_prefs=None, summary=None,
                       top_interests=None):
        """upsert 用户画像;传入 None 的字段保持原值不变,并按分片表活跃数刷新 fact_count。"""
        if not username:
            return
        conn = _pg.get_pg()
        if conn is None:
            return
        try:
            from psycopg.types.json import Jsonb
            table = table_for(username)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO user_profile (username, display_prefs, summary,
                                              top_interests, fact_count, updated_at)
                    VALUES (%s, COALESCE(%s, '{}'::jsonb), %s, COALESCE(%s, '[]'::jsonb), 0, now())
                    ON CONFLICT (username) DO UPDATE SET
                        display_prefs = CASE WHEN EXCLUDED.display_prefs = '{}'::jsonb
                                             THEN user_profile.display_prefs
                                             ELSE EXCLUDED.display_prefs END,
                        summary       = COALESCE(NULLIF(EXCLUDED.summary, ''), user_profile.summary),
                        top_interests = CASE WHEN EXCLUDED.top_interests = '[]'::jsonb
                                             THEN user_profile.top_interests
                                             ELSE EXCLUDED.top_interests END,
                        updated_at    = now()
                    """,
                    (username,
                     Jsonb(display_prefs) if display_prefs is not None else None,
                     summary or "",
                     Jsonb(top_interests) if top_interests is not None else None),
                )
                # fact_count 以分片表实际活跃条目数为准
                cur.execute(
                    f"SELECT count(*) FROM {table} WHERE username = %s AND status = 'active'",
                    (username,),
                )
                cnt = int(cur.fetchone()[0] or 0)
                cur.execute(
                    "UPDATE user_profile SET fact_count = %s, updated_at = now() "
                    "WHERE username = %s",
                    (cnt, username),
                )
        except Exception as e:  # noqa: BLE001
            logger.info("long-memory upsert_profile skipped: %s: %s",
                        type(e).__name__, str(e)[:160])

    # ---------------- 注销级联 ----------------
    def delete_user(self, username: str) -> int:
        """删除该用户全部长期记忆(画像 + 分片条目)。返回删除的条目数。"""
        if not username:
            return 0
        conn = _pg.get_pg()
        if conn is None:
            return 0
        deleted = 0
        try:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM user_profile WHERE username = %s", (username,))
                # 主分片直打;兜底遍历全部分片(防 SHARD_COUNT 历史变更残留)
                for name in all_shard_tables():
                    cur.execute(f"DELETE FROM {name} WHERE username = %s", (username,))
                    deleted += cur.rowcount or 0
            logger.info("long-memory deleted user=%s rows=%d", username, deleted)
        except Exception as e:  # noqa: BLE001
            logger.warning("long-memory delete_user failed: %s: %s",
                           type(e).__name__, str(e)[:160])
        return deleted


# 单例
long_term = LongTermMemory()
