# -*- coding: utf-8 -*-
"""工作记忆层。

LangGraph checkpoint:PostgresSaver(WORKING_PG_URI),存会话级 state 快照、跨轮续跑。

设计约束(对照 memory-system-design,第二种语义:工作记忆=会话级状态):
  - thread_id = 前端 conversation.id,同一对话跨轮复用 checkpoint(messages 跨轮累积)。
  - 长对话由 react/summarize.py 做摘要压缩(旧轮次 RemoveMessage + summary)。
  - 不长期保留:会话删除时 react/cleanup.delete_thread_artifacts 级联删三表;
    30 天未活动线程由 cleanup.prune_inactive 守护线程滚动清理。
  - 不做语义召回(那是 long 层的事)。
  - checkpoint 表由 PostgresSaver.setup() 自动创建(见 memories/db/init_all.py),不手写 DDL。
  - 状态结构由 server/chat/react/state.py 的 AgentState 定义;图运行时每个节点
    执行完,LangGraph 自动把 state 快照写库,无需业务代码手动存。
"""
import contextlib
import logging
import os
import sys

from ..connections import _PROJECT_ROOT  # noqa: F401

# config 路径(项目根/config)
_CONFIG_DIR = os.path.join(_PROJECT_ROOT, "config")

logger = logging.getLogger("agent")


def _pg_uri_with_timeout(uri: str) -> str:
    """给 PG URI 追加 connect_timeout(秒),避免故障时长时间挂起。已存在则不覆盖。"""
    if "connect_timeout" in uri:
        return uri
    sep = "&" if "?" in uri else "?"
    return f"{uri}{sep}connect_timeout=3"


@contextlib.contextmanager
def working_saver():
    """同步 PostgresSaver 上下文管理器,用于编译 LangGraph graph。

    PG 不可用(宕机/重启/连接打满)时降级为内存 checkpointer(InMemorySaver):
    本轮对话仍能正常完成,只是不跨轮续跑、不写 checkpoint——记忆是旁路,不能让
    PG 故障拖垮主对话。降级时记 warning,不抛异常。

    用法:
        from memories.storage import working_saver
        with working_saver() as cp:
            graph = builder.compile(checkpointer=cp)
            ...
    退出时自动归还连接。setup() 已在 db/init_all.py 执行,这里幂等再调一次更稳。
    """
    if _CONFIG_DIR not in sys.path:
        sys.path.insert(0, _CONFIG_DIR)
    import config as C

    _pg_cm = None
    try:
        from langgraph.checkpoint.postgres import PostgresSaver
        _pg_cm = PostgresSaver.from_conn_string(
            _pg_uri_with_timeout(C.WORKING_PG_URI)
        )
    except Exception as e:
        logger.warning(
            "working_saver PG 不可用,降级到 InMemorySaver(本轮不持久化 checkpoint): %s: %s",
            type(e).__name__, e,
        )

    if _pg_cm is not None:
        try:
            saver = _pg_cm.__enter__()
            saver.setup()
        except Exception as e:
            logger.warning(
                "working_saver PG 不可用,降级到 InMemorySaver(本轮不持久化 checkpoint): %s: %s",
                type(e).__name__, e,
            )
            try:
                _pg_cm.__exit__(None, None, None)
            except Exception:
                pass
            _pg_cm = None

    if _pg_cm is None:
        # 降级:内存 checkpointer(无 setup,进程内有效)
        from langgraph.checkpoint.memory import InMemorySaver
        with InMemorySaver() as saver:
            yield saver
        return

    # yield 必须在任何 except 之外:否则 with 体内调用方抛出的普通异常会被
    # 上面的 except 捕获,被误判为"PG 不可用"并二次 yield(RuntimeError),
    # 掩盖真实根因。finally 中无论正常/异常退出都归还连接。
    try:
        yield saver
    finally:
        _pg_cm.__exit__(*sys.exc_info())


@contextlib.asynccontextmanager
async def working_saver_async():
    """异步 AsyncPostgresSaver 上下文管理器(用于 async LangGraph)。"""
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    if _CONFIG_DIR not in sys.path:
        sys.path.insert(0, _CONFIG_DIR)
    import config as C
    async with AsyncPostgresSaver.from_conn_string(C.WORKING_PG_URI) as saver:
        await saver.setup()
        yield saver
