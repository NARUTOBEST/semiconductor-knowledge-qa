# -*- coding: utf-8 -*-
"""
Agent 三层记忆系统一键初始化 / 健康检查(幂等)。

  1. Redis 探测(PING):工作记忆 checkpoint + 短期记忆流水共用同一实例;
  2. 工作记忆:RedisSaver.setup() 幂等创建 checkpoint 索引(短期 mem:* 键运行期自动创建);
  3. 长期记忆(可选):PostgreSQL + pgvector,幂等建 user_profile + long_mem_00..N-1 分片表。

长期记忆是旁路增强:PG 不可用时仅告警、不阻断(Redis 两层是主功能)。

用法:
    python -m memories.db.init_all            # 初始化(checkpoint 索引 + PG 分片表)
    python -m memories.db.init_all --check    # 只探测连通性,不建索引/表

连接参数全部从 env 读取(config.REDIS_* / LONG_PG_URI / POSTGRES_*),禁止硬编码。
"""
import os
import sys
import argparse

# 项目根(本文件位于 <root>/memories/db/):加入 sys.path 以便 import config 与 memories 包
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (_ROOT, os.path.join(_ROOT, "config")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config as C  # noqa: E402


def _check_redis() -> bool:
    """探测 Redis 连通性,打印结果。返回是否可用。"""
    from memories.storage.connections import get_redis
    client = get_redis()
    if client is None:
        print("  [redis] 客户端构建失败(未安装 redis 包?)", file=sys.stderr)
        return False
    try:
        pong = client.ping()
        info = client.dbsize()
        print(f"  [redis] 连接 OK: {C.REDIS_HOST}:{C.REDIS_PORT}/db{C.REDIS_DB}"
              f"  ping={pong}  keys={info}")
        return True
    except Exception as e:
        print(f"  [redis] 连接失败: {type(e).__name__}: {e}", file=sys.stderr)
        print("        本地开发请先在 WSL2 启动 redis(见 env/env.example);"
              "部署环境请确认 compose 的 redis 服务已起。", file=sys.stderr)
        return False


def _setup_working() -> None:
    """RedisSaver.setup():幂等创建工作记忆 checkpoint 索引。"""
    from langgraph.checkpoint.redis import RedisSaver
    with RedisSaver.from_conn_string(C.REDIS_URL) as saver:
        saver.setup()
    print("  [working] LangGraph checkpoint 索引已就绪 (RedisSaver.setup)")


def _check_pg() -> bool:
    """探测长期记忆 PG 连通性。返回是否可用。"""
    from memories.storage.long import ping_pg
    if not getattr(C, "LONG_MEM_ENABLED", True):
        print("  [long] LONG_MEM_ENABLED=0,已禁用长期记忆,跳过。")
        return False
    if ping_pg():
        print(f"  [long] PostgreSQL 连接 OK,分片数={getattr(C, 'LONG_MEM_SHARD_COUNT', 16)}")
        return True
    print("  [long] PostgreSQL 不可用(长期记忆为旁路,将自动降级,不影响聊天)。",
          file=sys.stderr)
    print("        起一个带 pgvector 的 Postgres 并建库,例如:\n"
          "        docker run -d --name pg-mem -e POSTGRES_PASSWORD=<pwd> "
          "-e POSTGRES_DB=memory_long -p 5432:5432 pgvector/pgvector:pg16",
          file=sys.stderr)
    return False


def _setup_long() -> bool:
    """长期记忆:幂等建 pgvector 扩展 + user_profile + 分片表。返回是否成功。"""
    from memories.storage.long import long_term
    if long_term.setup():
        print(f"  [long] 长期记忆 schema 已就绪(user_profile + "
              f"{getattr(C, 'LONG_MEM_SHARD_COUNT', 16)} 张 long_mem_XX 分片表)")
        return True
    print("  [long] 长期记忆建表失败(见上方日志);PG 不可用时长期记忆降级。",
          file=sys.stderr)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="初始化 / 检查 Agent 三层记忆系统")
    parser.add_argument("--check", action="store_true",
                        help="只探测 Redis/PG 连通性,不建索引/表")
    args = parser.parse_args()

    # ---- Redis(工作 + 短期,必需)----
    print("[1/3] 检查 Redis 连接(工作记忆 + 短期记忆)...")
    redis_ok = _check_redis()
    if args.check:
        print("\n[3/3] 检查 PostgreSQL 连接(长期记忆)...")
        _check_pg()
        if redis_ok:
            print("\n[check] 完成。Redis 正常;短期流水键运行期自动创建。")
            return 0
        return 1

    if not redis_ok:
        return 1

    print("\n[2/3] 创建工作记忆 checkpoint 索引 ...")
    try:
        _setup_working()
    except Exception as e:
        print(f"  [working] RedisSaver.setup 失败: {type(e).__name__}: {e}",
              file=sys.stderr)
        return 1

    # ---- 长期记忆 PG(可选,失败不阻断)----
    print("\n[3/3] 初始化长期记忆(PostgreSQL + pgvector 分片表)...")
    if _check_pg():
        _setup_long()
    else:
        print("  [long] 跳过建表(PG 未就绪)。配好后重跑本脚本即可幂等补建。")

    print("\n全部初始化完成(工作记忆 checkpoint 就绪;短期流水 mem:* 运行期自动创建;"
          "长期记忆 PG 就绪后偏好抽取/召回自动生效)。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
