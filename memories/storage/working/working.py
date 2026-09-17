# -*- coding: utf-8 -*-
"""工作记忆层。

LangGraph checkpoint:RedisSaver(langgraph-checkpoint-redis),连 config.REDIS_URL,
存会话级 state 快照、跨轮续跑。

设计约束(对照 memory-system-design,第二种语义:工作记忆=会话级状态):
  - thread_id = 前端 conversation.id,同一对话跨轮复用 checkpoint(messages 跨轮累积)。
  - 长对话由 react/summarize.py 做摘要压缩(旧轮次 RemoveMessage + summary)。
  - 不长期保留:会话删除时 lifecycle.delete_thread_artifacts 级联清工作 checkpoint
    与短期流水;30 天未活动线程由 prune_inactive 守护线程滚动清理。
  - 不做语义召回(两级记忆中工作记忆只管会话级 state)。
  - checkpoint 索引由 RedisSaver.setup() 幂等创建(见 memories/db/init_all.py),不手写。
  - 状态结构由 agent_reasoning/ReAct/core/state.py 的 AgentState 定义;图运行时每个节点
    执行完,LangGraph 自动把 state 快照写 Redis,无需业务代码手动存。
  - checkpoint 键默认前缀 "checkpoint"/"checkpoint_write",与短期流水 mem:* 前缀互不冲突。
"""
import contextlib
import logging
import os
import sys

from ..connections import _PROJECT_ROOT  # noqa: F401

# config 路径(项目根/config)
_CONFIG_DIR = os.path.join(_PROJECT_ROOT, "config")

logger = logging.getLogger("agent")


@contextlib.contextmanager
def working_saver():
    """同步 RedisSaver 上下文管理器,用于编译 LangGraph graph。

    Redis 不可用(未启动/连不上)时按三级降级:SQLite 文件 checkpointer(本地
    文件,跨轮续跑保留,单机多 worker 共享)→ 内存 checkpointer(InMemorySaver,
    仅本轮,保底)——记忆是旁路,不能让 Redis 故障拖垮主对话。降级时记 warning,
    不抛异常。降级路径可用 WORKING_DEGRADE_DB env 改 SQLite 文件位置。

    用法:
        from memories.storage import working_saver
        with working_saver() as cp:
            graph = builder.compile(checkpointer=cp)
            ...
    退出时自动归还连接。setup() 幂等,这里再调一次确保索引就绪。
    """
    if _CONFIG_DIR not in sys.path:
        sys.path.insert(0, _CONFIG_DIR)
    import config as C

    _redis_cm = None
    try:
        from langgraph.checkpoint.redis import RedisSaver
        _redis_cm = RedisSaver.from_conn_string(C.REDIS_URL)
    except Exception as e:
        logger.warning(
            "working_saver RedisSaver 初始化失败,降级到 InMemorySaver"
            "(本轮不持久化 checkpoint): %s: %s",
            type(e).__name__, e,
        )

    if _redis_cm is not None:
        try:
            saver = _redis_cm.__enter__()
            saver.setup()
        except Exception as e:
            logger.warning(
                "working_saver Redis 不可用,降级到 InMemorySaver"
                "(本轮不持久化 checkpoint): %s: %s",
                type(e).__name__, e,
            )
            try:
                _redis_cm.__exit__(None, None, None)
            except Exception:
                pass
            _redis_cm = None

    if _redis_cm is None:
        # 降级链:SQLite 文件 checkpointer → InMemorySaver(保底)。
        # SQLite 落本地文件:单机多 worker 部署时各进程共享同一文件,会话状态不丢、
        # 跨轮续跑保留(InMemorySaver 只在进程内,多 worker 下各 worker 各一份,
        # 会话轮换到另一 worker 即"失忆");单 worker 下也优于内存(重启不丢)。
        db_path = os.getenv(
            "WORKING_DEGRADE_DB",
            os.path.join(_PROJECT_ROOT, "data", "working_degrade.sqlite3"))
        conn = None
        saver = None
        try:
            import sqlite3
            from langgraph.checkpoint.sqlite import SqliteSaver
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            conn = sqlite3.connect(db_path, check_same_thread=False,
                                   timeout=5)
            # WAL:多进程并发读写(每 worker 一个连接)互不阻塞;busy_timeout 兜底写竞争
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            saver = SqliteSaver(conn)
            saver.setup()
            logger.warning(
                "working_saver: Redis 不可用,降级 SQLite checkpointer(%s)——"
                "跨轮续跑保留,Redis 恢复后新 checkpoint 回到 Redis"
                "(降级窗口的 checkpoint 不回迁)", db_path)
        except Exception as e:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass
            saver = None
            logger.warning(
                "working_saver: SQLite 降级不可用,进一步降级 InMemorySaver"
                "(仅本轮有效,不跨轮续跑): %s: %s", type(e).__name__, e)
        if saver is not None:
            try:
                yield saver
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
            return
        # 保底:内存 checkpointer(无 setup,进程内有效)
        from langgraph.checkpoint.memory import InMemorySaver
        with InMemorySaver() as saver:
            yield saver
        return

    # yield 必须在任何 except 之外:否则 with 体内调用方抛出的普通异常会被
    # 上面的 except 捕获,被误判为"Redis 不可用"并二次 yield(RuntimeError),
    # 掩盖真实根因。finally 中无论正常/异常退出都归还连接。
    try:
        yield saver
    finally:
        _redis_cm.__exit__(*sys.exc_info())


@contextlib.asynccontextmanager
async def working_saver_async():
    """异步 AsyncRedisSaver 上下文管理器(用于 async LangGraph)。"""
    from langgraph.checkpoint.redis.aio import AsyncRedisSaver
    if _CONFIG_DIR not in sys.path:
        sys.path.insert(0, _CONFIG_DIR)
    import config as C
    async with AsyncRedisSaver.from_conn_string(C.REDIS_URL) as saver:
        await saver.setup()
        yield saver
