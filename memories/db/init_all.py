# -*- coding: utf-8 -*-
"""
Agent 记忆系统一键初始化(幂等)。

执行顺序对应需求第 4 节:
  1. [手动,在 WSL PostgreSQL 中] CREATE DATABASE agent_working_db / agent_short_db / agent_long_db
  2. 三个库分别 CREATE EXTENSION IF NOT EXISTS vector  (本脚本)
  3. agent_working_db: PostgresSaver.setup() 生成 checkpoint 系列表 (本脚本)
  4. agent_short_db:   执行 01_short_session_events.sql
                        + 03_short_promotion_watermark.sql (升迁水位线) (本脚本)
  5. agent_long_db:    执行 02_long_term_memories.sql + 索引 (本脚本)

用法:
    python memories/db/init_all.py            # 初始化全部
    python memories/db/init_all.py --check    # 只检查连接与扩展,不建表

连接参数全部从 env/env.env 读取,禁止硬编码。
重复执行安全(IF NOT EXISTS / setup() 幂等)。
"""
import os
import sys
import argparse

# 把 config/ 加入 path 以读取 .env(本文件位于 <root>/memories/db/,需回退两级)
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, "config"))
import config as C  # noqa: E402

import psycopg2  # noqa: E402
from langgraph.checkpoint.postgres import PostgresSaver  # noqa: E402

from migrate import run_migrations  # noqa: E402  (同目录,_HERE 已在 path)

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def _connect(uri: str):
    return psycopg2.connect(uri)


def _ensure_vector(uri: str, db_name: str) -> None:
    conn = _connect(uri)
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute("SELECT extversion FROM pg_extension WHERE extname='vector';")
            row = cur.fetchone()
            print(f"  [{db_name}] pgvector 扩展就绪: version={row[0] if row else '?'}")
    finally:
        conn.close()


def _exec_sql_file(uri: str, sql_file: str, db_name: str) -> None:
    with open(sql_file, encoding="utf-8") as f:
        sql = f.read()
    conn = _connect(uri)
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
        print(f"  [{db_name}] 已执行 {os.path.basename(sql_file)}")
    finally:
        conn.close()


def _setup_working(uri: str) -> None:
    # from_conn_string 是上下文管理器,内部用独立 connection;setup() 幂等
    with PostgresSaver.from_conn_string(uri) as saver:
        saver.setup()
    print("  [agent_working_db] LangGraph checkpoint 表已就绪 (PostgresSaver.setup)")


def main() -> int:
    parser = argparse.ArgumentParser(description="初始化 Agent 三库记忆系统")
    parser.add_argument("--check", action="store_true", help="只检查连接+扩展,不建表")
    args = parser.parse_args()

    uris = {
        "agent_working_db": C.WORKING_PG_URI,
        "agent_short_db": C.SHORT_PG_URI,
        "agent_long_db": C.LONG_PG_URI,
    }
    missing = [k for k, v in uris.items() if not v]
    if missing:
        print(f"[错误] 以下连接串未配置: {missing};请检查 env/env.env", file=sys.stderr)
        return 2

    print("[1/3] 检查三库连接并安装 pgvector 扩展 ...")
    for name, uri in uris.items():
        try:
            _ensure_vector(uri, name)
        except Exception as e:
            print(f"  [{name}] 连接/装扩展失败: {e}", file=sys.stderr)
            print("       请确认已在 WSL PostgreSQL 中手动 CREATE DATABASE 该库。", file=sys.stderr)
            return 1

    if args.check:
        print("\n[check] 连接与扩展正常。")
        return 0

    print("\n[2/3] agent_working_db: 生成 LangGraph checkpoint 表 ...")
    _setup_working(uris["agent_working_db"])

    print("\n[3/3] agent_short_db / agent_long_db: 执行业务 DDL ...")
    _exec_sql_file(uris["agent_short_db"],
                   os.path.join(_HERE, "01_short_session_events.sql"),
                   "agent_short_db")
    _exec_sql_file(uris["agent_short_db"],
                   os.path.join(_HERE, "03_short_promotion_watermark.sql"),
                   "agent_short_db")
    _exec_sql_file(uris["agent_long_db"],
                   os.path.join(_HERE, "02_long_term_memories.sql"),
                   "agent_long_db")

    print("\n[4/3] 执行增量迁移(migrations/)...")
    run_migrations(uris)

    print("\n全部初始化完成。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
