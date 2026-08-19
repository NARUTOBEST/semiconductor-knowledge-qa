# -*- coding: utf-8 -*-
"""验证:工作记忆 + PostgresSaver checkpoint 的"断点续跑"真的生效。

关键设计:用两个**独立 Python 进程**分别执行 run / resume,模拟"程序关掉再重新运行"。
工具调用次数通过磁盘文件 .weather_tool_calls.json 跨进程计数,作为权威证据。

用法:
    python tests/test_working_memory.py            # 跑完整两阶段(自动起子进程)
    python tests/test_working_memory.py run        # 仅首轮(新进程)
    python tests/test_working_memory.py resume     # 仅恢复(新进程)

成功标志:
    - 首轮:工具调用 1 次,AI 回复与 token 写入 checkpoint。
    - 恢复进程:能从 PG 读回首轮状态(回复、token);空输入再次 invoke 时工具调用次数
      仍为 1(不重复调外部工具)。
"""
import json
import os
import subprocess
import sys

from checkpoint_demo_graph import (
    SIDE_EFFECT_FILE,
    _read_calls,
    build_graph,
    reset_calls,
)
from memories.storage import pg_conn

THREAD_ID = "task-001"
thread_config = {"configurable": {"thread_id": THREAD_ID}}

# LangGraph PostgresSaver 的 checkpoint 表
_CHECKPOINT_TABLES = ["checkpoint_writes", "checkpoints", "checkpoint_blobs"]


def _purge_thread(thread_id: str) -> None:
    """从 working_db 删除某 thread 的全部 checkpoint,保证测试可重复运行。"""
    with pg_conn("working", dict_row=False) as (conn, cur):
        for tbl in _CHECKPOINT_TABLES:
            cur.execute(f"DELETE FROM {tbl} WHERE thread_id = %s", (thread_id,))

_PY = sys.executable
_HERE = os.path.dirname(os.path.abspath(__file__))


def _phase_run():
    """第一轮:新进程执行,调用一次工具并写 checkpoint。"""
    reset_calls()
    _purge_thread(THREAD_ID)
    print("=== 第一轮执行(全新进程) ===")
    g, cm = build_graph()
    try:
        result = g.invoke({"question": "帮我查一下北京的天气"}, config=thread_config)
        ai = result["messages"][-1].content
        print("AI回复:", ai)
        print("已用token:", result.get("usage"))
        print("工具实际调用次数:", _read_calls())
    finally:
        cm.__exit__(None, None, None)
    assert _read_calls() == 1, "首轮工具应恰好调用 1 次"


def _phase_resume():
    """第二轮:模拟重启后的全新进程,只凭 thread_id 恢复。"""
    print("\n=== 模拟重启后恢复任务(全新进程) ===")
    assert _read_calls() == 1, "恢复前磁盘上的工具调用次数应仍为 1(首轮遗留)"
    g, cm = build_graph()
    try:
        # 1) 直接从 PG checkpoint 读回状态,证明状态已持久化
        snapshot = g.get_state(config=thread_config)
        recovered = snapshot.values
        print("恢复的任务目标 question:", recovered.get("question"))
        last_msg = recovered["messages"][-1].content if recovered.get("messages") else None
        print("恢复的上一轮AI回复:", last_msg)
        print("恢复的已用token:", recovered.get("usage"))
        assert recovered.get("usage", {}).get("total_tokens") == 30, "应恢复出 token 统计"
        assert "北京天气" in (last_msg or ""), "应恢复出首轮 AI 回复"

        # 2) 不传新消息,空输入再 invoke —— 已到 END 的图不应重跑 weather_tool
        print("\n--- 不传新消息再次 invoke,验证工具不会被重复调用 ---")
        g.invoke(None, config=thread_config)
        print("再次 invoke 后工具实际调用次数:", _read_calls())
        assert _read_calls() == 1, "已完成的任务恢复后不应再次调用工具!checkpoint 未生效"
    finally:
        cm.__exit__(None, None, None)
    print("\n✅ 验证成功:状态从 PostgreSQL checkpoint 恢复,工具未重复执行。")


def main():
    phase = sys.argv[1] if len(sys.argv) > 1 else "all"
    if phase == "run":
        _phase_run()
    elif phase == "resume":
        _phase_resume()
    elif phase == "all":
        # 关键:用两个独立子进程,真正模拟"关掉程序再重开"
        for sub in ("run", "resume"):
            r = subprocess.run([_PY, os.path.join(_HERE, "test_working_memory.py"), sub],
                               cwd=_HERE)
            if r.returncode != 0:
                sys.exit(r.returncode)
        # 清理跨进程证据文件
        if os.path.exists(SIDE_EFFECT_FILE):
            os.remove(SIDE_EFFECT_FILE)
    else:
        print(f"unknown phase: {phase}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
