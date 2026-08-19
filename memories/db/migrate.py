# -*- coding: utf-8 -*-
"""轻量 schema 迁移 runner(不引入 Alembic)。

设计:
- 迁移文件位于 ``memories/db/migrations/``,命名 ``NNN_description.sql``,
  按三位数字前缀升序执行。
- 每个文件首行必须含注释 ``-- database: <db_name>`` 声明目标库
  (db_name 是 init_all 传入 uris 的键,如 agent_short_db)。
- 每个业务库各自维护一张 ``schema_migrations(version PRIMARY KEY, applied_at)``,
  已记录的版本跳过。
- 每个迁移在单个事务中执行:SQL 成功才写入版本记录;失败回滚,不写记录。
- 迁移文件本身应尽量幂等(IF NOT EXISTS 等),便于人工重放;但 runner 以
  schema_migrations 记录为准,正常流程不会重复执行。

被 init_all.py 在全量 DDL 之后调用(全量 DDL 是 fresh-install 的最新快照,
迁移用于把存量库增量升到最新)。
"""
from __future__ import annotations

import os
import re
from typing import Iterable

import psycopg2

_HERE = os.path.dirname(os.path.abspath(__file__))
MIGRATIONS_DIR = os.path.join(_HERE, "migrations")
_DB_RE = re.compile(r"--\s*database:\s*(\S+)", re.IGNORECASE)
_VER_RE = re.compile(r"^(\d{3})_")


def _ensure_version_table(uri: str) -> None:
    conn = psycopg2.connect(uri)
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version    TEXT PRIMARY KEY,
                    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                """
            )
    finally:
        conn.close()


def _applied_versions(uri: str) -> set[str]:
    conn = psycopg2.connect(uri)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT version FROM schema_migrations;")
            return {r[0] for r in cur.fetchall()}
    finally:
        conn.close()


def _parse_migration(path: str) -> tuple[str, str, str]:
    """返回 (version, db_name, sql)。"""
    fname = os.path.basename(path)
    m = _VER_RE.match(fname)
    if not m:
        raise ValueError(f"迁移文件名需以 NNN_ 开头: {fname}")
    version = m.group(1)
    with open(path, encoding="utf-8") as f:
        sql = f.read()
    dm = _DB_RE.search(sql)
    if not dm:
        raise ValueError(f"迁移文件缺少 '-- database: <db>' 声明: {fname}")
    return version, dm.group(1), sql


def discover_migrations(directory: str = MIGRATIONS_DIR) -> list[tuple[str, str, str]]:
    """扫描目录,返回按版本号升序的 (version, db_name, sql) 列表。"""
    if not os.path.isdir(directory):
        return []
    items = []
    for fname in os.listdir(directory):
        if not fname.endswith(".sql"):
            continue
        items.append(_parse_migration(os.path.join(directory, fname)))
    items.sort(key=lambda x: x[0])
    return items


def run_migrations(uris: dict[str, str], *,
                   directory: str = MIGRATIONS_DIR,
                   printer=print) -> list[str]:
    """对所有业务库执行未应用的迁移。返回本次新应用的版本列表。"""
    for uri in uris.values():
        _ensure_version_table(uri)

    applied_per_db: dict[str, set[str]] = {
        name: _applied_versions(uri) for name, uri in uris.items()
    }

    newly_applied: list[str] = []
    for version, db_name, sql in discover_migrations(directory):
        uri = uris.get(db_name)
        if uri is None:
            raise KeyError(f"迁移 {version} 声明的库 '{db_name}' 不在 uris 中")
        if version in applied_per_db[db_name]:
            continue

        conn = psycopg2.connect(uri)
        try:
            # 关闭 autocommit,让 SQL + 版本写入在同一事务内原子提交
            conn.autocommit = False
            with conn.cursor() as cur:
                cur.execute(sql)
                cur.execute(
                    "INSERT INTO schema_migrations (version) VALUES (%s);",
                    (version,),
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

        applied_per_db[db_name].add(version)
        newly_applied.append(version)
        printer(f"  [{db_name}] 已应用迁移 {version}")

    if not newly_applied:
        printer("  无待执行迁移(schema 已是最新)。")
    return newly_applied
