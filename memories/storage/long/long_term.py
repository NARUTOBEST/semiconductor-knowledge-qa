# -*- coding: utf-8 -*-
"""长期记忆层:long_term_memories(结构化 + pgvector 向量)。

职责:跨 thread 沉淀事实/偏好/实体,语义可召回。
不存原始流水、不存进行中状态。连接 LONG_PG_URI。

向量通过 RAG/embed.py 的 BGE-m3 文本编码器生成(dense 1024 维,已 L2 归一化),
与后端主进程 retriever_warmup 复用同一单例,不重复加载。
"""
from typing import Any, Optional
import os
import sys

import psycopg2
from pgvector.psycopg2 import register_vector

# 本文件位于 memories/storage/long/;项目根在上三级
_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
if os.path.join(_ROOT, "config") not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "config"))
import config as C  # noqa: E402


def _embed_text(text: str):
    """返回 BGE-m3 dense 向量(np.ndarray, shape=(1024,), float32)。

    延迟导入 embed,避免无模型环境(如纯写库/测试)在 import 时加载 torch。
    """
    for p in (os.path.join(_ROOT, "RAG"), _ROOT, os.path.join(_ROOT, "config")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import embed  # type: ignore
    dense, _sparse = embed.get_text_encoder().encode([text])
    return dense[0]


class LongTermMemory:
    """长期记忆读写。"""

    def _conn(self):
        """从 long 库连接池取一个注册了 pgvector 的连接(池满时等待)。"""
        from ..connections import pool_getconn, pool_putconn
        conn = pool_getconn("long")
        try:
            register_vector(conn)
        except Exception:
            pool_putconn("long", conn, close=True)
            raise
        return conn

    @staticmethod
    def _release(conn):
        """归还连接到池。"""
        from ..connections import pool_putconn
        pool_putconn("long", conn)

    def find_similar(self, user_id: str, vec, *,
                     memory_type: Optional[str] = None,
                     threshold: float = 0.92,
                     k: int = 5) -> Optional[dict[str, Any]]:
        """在同用户(可选同类型)既有记忆中,找与 vec 余弦相似度 >= threshold 的最近邻。

        用于写入前语义去重。vec 为已算好的 BGE-m3 向量(避免重复编码)。
        返回命中的记忆 dict(含 id/content/score),无命中返回 None。
        """
        conn = self._conn()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                sql = (
                    "SELECT id, content, 1 - (embedding <=> %s) AS score "
                    "FROM long_term_memories "
                    "WHERE user_id = %s AND embedding IS NOT NULL"
                )
                params: list[Any] = [vec, user_id]
                if memory_type is not None:
                    sql += " AND memory_type = %s"
                    params.append(memory_type)
                sql += " ORDER BY embedding <=> %s LIMIT %s"
                params.extend([vec, int(k)])
                cur.execute(sql, params)
                for r in cur.fetchall():
                    if float(r["score"]) >= threshold:
                        return dict(r)
                return None
        finally:
            self._release(conn)

    def add_memory(self, user_id: str, content: str, *,
                   memory_type: str = "fact",
                   thread_id: Optional[str] = None,
                   meta: Optional[dict[str, Any]] = None,
                   embed: bool = True,
                   dedup: bool = True) -> Optional[int]:
        """写入一条长期记忆(双层去重)。

        1. 语义去重:同 user 同 type 下,与既有记忆余弦相似度 >=
           LONG_MEMORY_DEDUP_THRESHOLD 则跳过,返回已存在记忆的 id。
        2. 精确去重:唯一索引 (user_id, md5(content)) 兜底,相同 content 返回已存在 id。
        语义去重仅在 embed=True 且 dedup=True 时执行。
        embed=False 时不计算向量(embedding 置 NULL),用于无法加载模型时的兜底写入。
        新增语义重复时返回命中的已有记忆 id(正数);无写入但也无命中不会发生。
        """
        vec = _embed_text(content) if embed else None
        # 语义去重:同用户同类型的近似记忆已存在则跳过
        if dedup and vec is not None:
            hit = self.find_similar(
                user_id, vec, memory_type=memory_type,
                threshold=C.LONG_MEMORY_DEDUP_THRESHOLD,
            )
            if hit is not None:
                return int(hit["id"])
        conn = self._conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO long_term_memories
                        (user_id, thread_id, memory_type, content, embedding, meta)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (user_id, md5(content)) DO UPDATE
                        SET updated_at = now()
                    RETURNING id;
                    """,
                    (user_id, thread_id, memory_type, content, vec,
                     psycopg2.extras.Json(meta or {})),
                )
                row = cur.fetchone()
            conn.commit()
            return int(row[0])
        finally:
            self._release(conn)

    def vector_search(self, user_id: str, query: str,
                      k: int = 5,
                      memory_types: Optional[list[str]] = None
                      ) -> list[dict[str, Any]]:
        """向量召回:余弦相似度 top-k。"""
        qv = _embed_text(query)
        conn = self._conn()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                sql = (
                    "SELECT id, user_id, thread_id, memory_type, content, meta, "
                    "       1 - (embedding <=> %s) AS score "
                    "FROM long_term_memories "
                    "WHERE user_id = %s AND embedding IS NOT NULL"
                )
                params: list[Any] = [qv, user_id]
                if memory_types:
                    sql += " AND memory_type = ANY(%s)"
                    params.append(memory_types)
                sql += " ORDER BY embedding <=> %s LIMIT %s"
                params.extend([qv, int(k)])
                cur.execute(sql, params)
                return [dict(r) for r in cur.fetchall()]
        finally:
            self._release(conn)

    def get_by_user(self, user_id: str, limit: int = 100) -> list[dict[str, Any]]:
        """结构化拉取某用户全部记忆(管理/审计用)。"""
        conn = self._conn()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT id, thread_id, memory_type, content, meta, created_at, updated_at "
                    "FROM long_term_memories WHERE user_id=%s "
                    "ORDER BY created_at DESC LIMIT %s",
                    (user_id, int(limit)),
                )
                return [dict(r) for r in cur.fetchall()]
        finally:
            self._release(conn)


long_term = LongTermMemory()
