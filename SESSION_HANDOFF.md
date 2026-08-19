# 会话交接：三级范式架构

> 半导体知识 RAG Agent 已从「单体 ReAct 图」改造为「复杂度路由 + 三种推理范式 +
> 共享质检门」。本文档记录新架构、关键文件与运行/验证方式，供交接使用。
> 改造任务清单见 [`REFACTOR_LOOP.md`](REFACTOR_LOOP.md)（阶段 0–9 已完成，阶段 10 为验证与文档）。

## 一、架构总览

```
用户问题
  │
  ▼
classify_complexity(question, history)        # agent_reasoning/router.py
  │  规则预筛(超短/问候→simple, 复杂标记→complex) + lite LLM 分类
  │  失败/低置信度 → fail-open medium
  ▼
react_stream(service.py) 先发 {"type":"tier",...}
  │
  ├── simple  → run_simple()           单轮直答, 无工具, TIER_MODEL_SIMPLE(lite)
  ├── medium  → run_agent_graph()      ReAct 循环, plan 条件触发(软引导)
  └── complex → run_plan_execute()     Plan-and-Execute(必经规划 + 逐步 react_loop + synthesizer)
  │
  ▼  答案产出后、done 之前
quality_gate.check(answer, context, tier)     # agent_reasoning/quality_gate.py
  │  passed            → 放行
  │  failed            → 同 tier 重做一次(最多 1 次)
  │  needs_escalation  → simple→medium→complex(整条请求最多 1 次)
  ▼
done(含结构化 trace)
```

所有旁路功能（记忆、规划、grounding、重排）**失败一律 fail-open + 可见 status
警示，不阻断主对话**。

## 二、关键模块

| 文件 | 职责 |
|------|------|
| `agent_reasoning/router.py` | `classify_complexity()`：规则预筛 + lite LLM 分类，返回 `{tier, confidence, source}` |
| `agent_reasoning/quality_gate.py` | `check()`：按 tier 深浅返回 `passed/failed/needs_escalation`；复用 medium 图已算好的 grounding，避免重复 LLM |
| `server/chat/service.py` | `react_stream()`：路由→分发→扣留 `done`→质检→重做/升级；按 tier 取 `TIER_CONFIG` 预算 |
| `server/chat/router.py` | HTTP `/api/chat` → SSE；观测 `tier/escalation/grounding` 事件写 per-tier metrics |
| `agent_reasoning/ReAct/support/runner.py` | `run_path()` 通用包装器：短期流水落库 + `on_event` + 异常兜底 + `finally` 触发 `after_stream`；三个入口 `run_simple/run_agent_graph/run_plan_execute` |
| `agent_reasoning/ReAct/paths/simple.py` | simple 路径：一次 LLM 调用，不绑工具，保留 recall，跳过 rewrite |
| `agent_reasoning/ReAct/paths/plan_execute.py` | complex P&E：planner→逐步隔离 `react_loop`（缺资料换词重试 1 次）→synthesizer→grounding；每步独立子 `TraceRecorder` |
| `agent_reasoning/ReAct/support/planning.py` | `generate_plan(force=…)` 共享规划，返回 `(steps, error)`；medium 条件触发、complex force=True |
| `agent_reasoning/ReAct/trace.py` | `TraceRecorder`；P&E 嵌套结构 `plan_execute{planner, step_results[], synthesizer}` |
| `server/support/metrics.py` | 内存指标；新增 `record_tier_result/record_escalation`，`get_stats()["by_tier"]` 含请求量/错误率/升级数/延迟 p50/p95/grounding 通过率/token |
| `config/config.py` | `TIER_CONFIG`（每 tier 模型/`max_steps`/`max_total_seconds`/`plan_enabled`/`quality_depth`）、`ROUTER_*`，全部可经环境变量覆盖 |

## 三、SSE 事件契约

- 每个流**第一个事件固定是 `tier`**。
- 升级发 `escalation`（`from_tier/to_tier/reason`），前端与 `reflect` 一样清空当前输出。
- 完整序列与所有事件类型见 [`docs/sse_events.md`](docs/sse_events.md)。
- 前端已适配：`web/lib/api.ts` 解析 `tier/escalation`；`web/app/page.tsx` 管状态；
  `web/components/ChatBox.tsx` 显示 tier 徽标（快速直答/检索推理/计划执行）。

## 四、配置

所有 tier 参数见 `config/config.py` 的 `TIER_CONFIG`，可在 `env/env.env` 覆盖
（模板见 `env/env.example`）：

- `TIER_MODEL_SIMPLE/MEDIUM/COMPLEX`
- `TIER_<TIER>_MAX_STEPS`、`TIER_<TIER>_MAX_TOTAL_SECONDS`
- `TIER_MEDIUM_PLAN_ENABLED`（1/0）
- `ROUTER_TIMEOUT`、`ROUTER_CONFIDENCE_MIN`、`ROUTER_SHORT_LEN`

## 五、运行与验证

```bash
# 后端（Windows）
启动配置/run_agent.ps1
# 前端
cd web && npm run dev
```

手动验证清单（10.2/10.3）：

1. 闲聊（如「你好」「你能做什么」）→ `tier=simple`，无工具调用，响应快。
2. 领域事实题（如「ALD 基本原理」）→ `tier=medium`，有 `tool_call/tool_result`。
3. 多步/对比题（如「对比 ALD 和 CVD 的优缺点与适用场景」）→ `tier=complex`，
   先 `plan` 再逐 `step_start`，最后 `synthesis_start`。
4. 故意让 simple 命中领域内容或 medium grounding 未过 → 收到 `escalation` 升级重跑，
   前端清空旧输出显示「正在深入分析…」。
5. `GET /metrics` 返回 `by_tier` 统计；`/health` 正常；后端日志无异常 traceback。

## 六、测试

```bash
python -m pytest            # 267 passed（无需 -p no:langsmith 兜底）
```

测试分层与 mock 边界见 [`TESTING.md`](TESTING.md)「三级范式测试要点」。新增的
相关测试文件：`test_router.py`、`test_routing.py`、`test_quality_gate.py`、
`test_escalation.py`、`test_plan_execute.py`、`test_persistence_three_paths.py`、
`test_metrics.py`、`test_tier_config.py`、`test_simple_path.py`。

## 七、已知遗留

- `memories/storage/working/summarize.py:263` 偶发 `InvalidStateError: FINISHED`
  线程告警（改造前既有，不影响结果，可单独修）。
- 阶段 10.2/10.3 为手动验证项，需在配好 LLM/Qdrant/PG 的真实环境执行；自动化测试
  已覆盖三条路径的事件序列、质检升级、流水升迁与指标记账。
