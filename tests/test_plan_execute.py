# -*- coding: utf-8 -*-
"""阶段 5.9:Plan-and-Execute 路径测试。

mock 掉 react_loop(每步子循环)、synthesizer 的 LLM 与 grounding_check,验证:
  - 多步按顺序执行,每步把 step_instruction 透传给 react_loop(步骤隔离)
  - 发 plan 事件 + synthesis_start + token + assistant_message
  - 某步无结果 -> 重试一次 -> 仍无则标记 missing 跳过,不中断
  - synthesizer 失败 -> 拼接各步结果降级
  - 总预算不足 -> 跳过剩余步骤直接 synthesizer
  - planner 失败 -> runner.run_plan_execute 降级普通 ReAct
不调真实 LLM / Qdrant / PG。
"""
import os
import sys
import time
import types
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_reasoning.ReAct.paths import plan_execute as pe_mod
from agent_reasoning.ReAct.trace import TraceRecorder
import agent_reasoning.ReAct.support.runner as runner_mod


def _fake_step(answer="该步答案", sources=None, found=True):
    """返回一个假 react_loop 生成器工厂;产出若干事件后 return 终态。"""
    result = {
        "full_reply": answer if found else "",
        "collected_sources": {} if not found else {
            "d__t1": {"chunk_id": "d__t1", "source_stem": "manual",
                      "page": "12", "score": 0.9,
                      "content": (sources or "资料内容")},
        },
        "search_count": 1 if found else 0,
        "final_reason": "answer",
    }

    def _gen(*a, **k):
        yield {"type": "step_start", "step": 1}
        yield {"type": "token", "delta": answer[:1]}
        return result
    return _gen


def _synth_stream(text="综合答案"):
    """假 synthesizer LLM 流。"""
    for ch in text:
        yield types.SimpleNamespace(
            choices=[types.SimpleNamespace(
                finish_reason=None,
                delta=types.SimpleNamespace(content=ch))],
            usage=None)
    yield types.SimpleNamespace(
        choices=[types.SimpleNamespace(
            finish_reason="stop", delta=types.SimpleNamespace(content=None))],
        usage=types.SimpleNamespace(prompt_tokens=5, completion_tokens=4,
                                    total_tokens=9))


def _run_path(steps, *, react_factories=None, synth_text="综合答案",
              max_total_seconds=60, synth_raises=False):
    """跑 plan_execute_stream,收集事件。react_factories 是每步 react_loop 工厂列表。"""
    recorder = TraceRecorder("pe", time.time(), "q")
    t0 = time.time()
    if react_factories is None:
        react_factories = [_fake_step(answer=f"步骤{i}答案")
                           for i in range(1, len(steps) + 1)]

    # side_effect 为可调用对象:每次 react_loop(...) 调用取下一个工厂并"调用"它,
    # 返回一个全新的生成器迭代器(模拟生成器函数语义)。
    queue = list(react_factories)

    def _react_dispatch(*a, **k):
        factory = queue.pop(0)
        return factory(*a, **k)

    def _collect_stream(*a, **k):
        if synth_raises:
            return None, RuntimeError("synth down")
        return _synth_stream(synth_text), None

    with patch.object(pe_mod, "react_loop", side_effect=_react_dispatch) as rl, \
         patch.object(pe_mod, "llm_create_with_retry",
                      side_effect=_collect_stream), \
         patch.object(pe_mod, "grounding_check",
                      return_value={"passed": True, "warnings": []}):
        events = list(pe_mod.plan_execute_stream(
            "复杂问题", [], steps,
            recorder=recorder, trace_id="pe", t0=t0,
            thread_id="t1", username="alice",
            max_total_seconds=max_total_seconds,
        ))
    return events, recorder, rl


class TestPlanExecute:
    def test_plan_event_and_sequential_steps(self):
        steps = ["查 ALD", "查 CVD", "对比"]
        events, recorder, rl = _run_path(steps)
        types = [e["type"] for e in events]
        # plan 事件在最前
        assert types[0] == "plan"
        plan = events[0]
        assert plan["steps"] == steps
        # 每步都调了 react_loop,且 step_instruction 依次透传
        assert rl.call_count == 3
        for i, c in enumerate(rl.call_args_list):
            assert c.kwargs["step_instruction"] == steps[i]
            # 每步独立:不共享 messages(每次都新建 SystemMessage)
            assert c.kwargs["skip_recall"] is True
        # synthesizer 与收尾事件齐全
        assert "synthesis_start" in types
        assert types[-1] == "done"
        assert any(e["type"] == "assistant_message" for e in events)
        assert recorder.plan is not None

    def test_step_isolation_distinct_state(self):
        """每步 react_loop 接收独立的 initial_messages(不复用前一步 messages)。"""
        captured = []

        def _capture(*a, **k):
            captured.append([m for m in a[0]])  # initial_messages 快照
            yield {"type": "step_start"}
            return {"full_reply": "x",
                    "collected_sources": {"d1": {"chunk_id": "d1"}},
                    "search_count": 1, "final_reason": "answer"}

        _run_path(["s1", "s2"], react_factories=[_capture, _capture])
        # 两次调用的初始消息列表是不同对象(隔离),且都只含一条 system
        assert captured[0] is not captured[1]
        assert len(captured[0]) == 1 and len(captured[1]) == 1

    def test_missing_step_retried_then_skipped(self):
        # 第一步:第一次没结果 -> 第二次仍没结果 -> 标记 missing
        not_found = _fake_step(found=False)
        ok = _fake_step(answer="第二步答案")
        events, recorder, rl = _run_path(
            ["缺资料的步", "正常步"],
            react_factories=[not_found, not_found, ok],
        )
        # 第1步调用 2 次(原 + 重试),第2步 1 次 = 3
        assert rl.call_count == 3
        # 重试指令含"重试"提示
        assert "重试" in rl.call_args_list[1].kwargs["step_instruction"]
        # 仍产出最终答案(第二步行 + synthesizer),未中断
        assert any(e["type"] == "done" for e in events)
        # 缺失步骤写入 coverage.uncovered_steps
        cov = recorder.coverage
        assert any("缺资料的步" in u for u in cov["uncovered_steps"])

    def test_missing_step_retry_succeeds(self):
        # 第一次没结果,重试有结果 -> 不标记 missing
        not_found = _fake_step(found=False)
        found = _fake_step(answer="补到的答案")
        events, recorder, rl = _run_path(
            ["s1"], react_factories=[not_found, found])
        assert rl.call_count == 2
        assert recorder.coverage["uncovered_steps"] == []

    def test_synthesizer_failure_concatenates_steps(self):
        ok = _fake_step(answer="步骤A的检索结果")
        events, recorder, rl = _run_path(
            ["s1"], react_factories=[ok], synth_raises=True)
        msgs = [e for e in events if e["type"] == "assistant_message"]
        # 降级拼接:assistant_message 含该步结果
        assert msgs and "步骤A的检索结果" in msgs[-1]["content"]
        assert any(e["type"] == "done" for e in events)

    def test_budget_cutoff_skips_remaining_steps(self):
        # max_total_seconds 给得极小 -> 第一步之后预算耗尽,后续步骤直接 missing
        ok = _fake_step(answer="x")
        events, recorder, rl = _run_path(
            ["s1", "s2", "s3"],
            react_factories=[ok, ok, ok],
            max_total_seconds=0,   # 立即耗尽
        )
        # 只真正执行了第一步
        assert rl.call_count == 1
        assert any("跳过剩余" in e.get("message", "")
                   for e in events if e["type"] == "status")
        cov = recorder.coverage
        # s2,s3 标记为缺失
        assert len(cov["uncovered_steps"]) == 2

    def test_nested_trace_structure(self):
        """8.3:每步用独立子 recorder,嵌套进 plan_execute.step_results;synthesizer 独立段。"""
        def _recording_step(answer, tool_name):
            def _gen(*a, **k):
                child = k["configurable"]["trace_recorder"]
                sd = child.new_step(1)
                child.record_llm(sd, finish_reason="tool_calls",
                                 usage={"prompt_tokens": 3, "completion_tokens": 2,
                                        "total_tokens": 5},
                                 thought="思考", tool_calls=[{"id": "c1", "name": tool_name}])
                child.record_tool(sd, tool_call_id="c1", name=tool_name,
                                  args={"q": answer}, ok=True, duration_ms=10,
                                  result={"hits": [answer]})
                child.finish_step(sd, "tool_calls", new_sources_count=1)
                yield {"type": "step_start", "step": 1}
                return {"full_reply": answer,
                        "collected_sources": {"d1": {"chunk_id": "d1"}},
                        "search_count": 1, "final_reason": "answer"}
            return _gen

        events, recorder, rl = _run_path(
            ["查 ALD", "查 CVD"],
            react_factories=[_recording_step("ALD答案", "search"),
                             _recording_step("CVD答案", "search")],
        )
        pe = recorder.plan_execute
        assert pe is not None
        assert pe["planned_steps"] == ["查 ALD", "查 CVD"]
        # planner 段已记录
        assert pe["planner"] and pe["planner"]["steps"] == ["查 ALD", "查 CVD"]
        # 每步独立嵌套
        results = pe["step_results"]
        assert [r["instruction"] for r in results] == ["查 ALD", "查 CVD"]
        assert all(not r["missing"] for r in results)
        # 子步骤的 tool/llm 折叠进了嵌套 steps,且不污染父 recorder.steps
        assert results[0]["steps"][0]["tools"][0]["name"] == "search"
        assert results[0]["tokens"]["total"] == 5
        # 父 steps 不混入子循环记录(只有 synthesizer 也已移走,故为空)
        assert recorder.steps == []
        # token 累加进父(每步 5 + synth 9)
        assert recorder.total_tokens["total"] == 5 + 5 + 9
        # synthesizer 独立段
        synth = pe["synthesizer"]
        assert synth is not None
        assert synth["answer_len"] > 0
        assert synth["usage"]["total_tokens"] == 9
        # to_dict 携带 plan_execute
        d = recorder.to_dict()
        assert d["plan_execute"] is pe


class TestPlannerDegradation:
    def test_planner_failure_falls_back_to_react(self):
        """runner.run_plan_execute:planner 返回空 -> 降级 run_agent_graph。"""
        called = {"react": False}

        def _fake_react(*a, **k):
            called["react"] = True
            yield {"type": "status", "message": "reacting"}
            yield {"type": "done", "trace": {}}

        from memories.storage.short import short_term as _st
        with patch.object(runner_mod, "generate_plan",
                          return_value=([], "planner down")), \
             patch.object(runner_mod, "run_agent_graph", side_effect=_fake_react), \
             patch.object(_st, "append_event", lambda *a, **k: None), \
             patch.object(runner_mod, "after_stream", lambda *a, **k: None), \
             patch.object(runner_mod, "persist_event", lambda *a, **k: None):
            events = list(runner_mod.run_plan_execute(
                "复杂问题", [], thread_id="t1"))
        assert called["react"] is True
        assert any("规划不可用" in e.get("message", "") for e in events)
        assert events[-1]["type"] == "done"


class TestGeneratePlanShared:
    def test_generate_plan_force_returns_steps(self):
        from agent_reasoning.ReAct.support.planning import generate_plan
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = ('{"need_plan": true, '
                                           '"steps": ["查 ALD", "查 CVD"]}')
        with patch("agent_reasoning.ReAct.support.planning.get_client",
                   return_value=MagicMock()), \
             patch("agent_reasoning.ReAct.support.planning.llm_create_with_retry",
                   return_value=(resp, None)):
            steps, err = generate_plan("对比 ALD 和 CVD", force=True)
        assert steps == ["查 ALD", "查 CVD"]
        assert err is None

    def test_generate_plan_failure_returns_empty_with_error(self):
        from agent_reasoning.ReAct.support.planning import generate_plan
        with patch("agent_reasoning.ReAct.support.planning.get_client",
                   return_value=MagicMock()), \
             patch("agent_reasoning.ReAct.support.planning.llm_create_with_retry",
                   return_value=(None, RuntimeError("down"))):
            steps, err = generate_plan("q", force=True)
        assert steps == []
        assert err and "down" in err
