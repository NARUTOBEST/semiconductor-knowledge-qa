# 测试指南

> **架构（2026-08 改造后）**：单体 ReAct 图已改为「复杂度路由 + 三种推理范式 +
> 共享质检门」。请求先由路由器（lite LLM + 规则预筛）判定 `simple` / `medium` /
> `complex`，分别走单轮直答、ReAct 检索循环、Plan-and-Execute；答案产出后过共享
> 质检门，可同 tier 重做一次或升级一次（simple→medium→complex）。
> 详见 [`docs/sse_events.md`](docs/sse_events.md) 与 `agent_reasoning/`。

## 运行测试

```bash
# 安装测试依赖
pip install -r requirements-test.txt

# 运行全部测试（环境已修复，直接用 pytest 即可，无需 -p no:langsmith）
python -m pytest

# 若 langsmith 插件在某些环境导致收集失败，可临时禁用
python -m pytest -p no:langsmith

# 运行单个文件
python -m pytest tests/test_chunker.py

# 运行单个测试类
python -m pytest tests/test_auth.py::TestJWT

# 运行单条用例
python -m pytest tests/test_sse.py::TestSSEEventFormat::test_basic_stream

# 详细输出 + 完整回溯
python -m pytest -v --tb=long
```

> 基线（2026-08-19）：**28 个 `test_*.py` 文件，其中 27 个共收集 267 条用例，全部通过**。
> `test_working_memory.py` 是需要真实 PostgreSQL + 子进程的手动集成脚本，不含 pytest 用例（收集 0 条），见下文「手动集成脚本」。

## 测试清单

| 文件 | 用例数 | 覆盖模块 | 说明 |
|------|--------|----------|------|
| `test_chunker.py` | 28 | `RAG/chunker.py` | 分块装箱、标题切段、过滤项、图片/表格/公式、合并与超大切分、页码范围 |
| `test_auth.py` | 25 | `server/auth/` | 密码哈希(bcrypt)、JWT(签发/校验/过期/篡改)、注册/登录/me API、DB 操作 |
| `test_grounding.py` | 21 | `agent_reasoning/ReAct/support/answer_grounding.py` | 引用校验 + 忠实度检查，均 mock LLM，覆盖 grounding 全链路 |
| `test_router.py` | 20 | `agent_reasoning/router.py` | 复杂度路由:规则预筛、LLM 三分类、解析失败/低置信度/异常兜底 medium |
| `test_summarize_and_cleanup.py` | 18 | `memories/storage/working/` | 工作记忆摘要压缩与 TTL 清理 |
| `test_sse.py` | 15 | `server/chat/router.py` | SSE 事件格式、错误处理、鉴权、参数校验 |
| `test_quality_gate.py` | 14 | `agent_reasoning/quality_gate.py` | 三态判定(passed/failed/needs_escalation)、各 tier 深度、fail-open |
| `test_query_rewrite.py` | 13 | `context management/query_rewrite.py` | LLM 查询改写成功/回退/历史处理/markdown 清洗等 |
| `test_trace.py` | 12 | `agent_reasoning/ReAct/trace.py`、`support/runner.py` | ReAct 结构化事件(step_start/tool_call/tool_result/grounding/error/done)与完整 trace |
| `test_ratelimit.py` | 10 | `server/support/ratelimit.py` | 单用户限流、全局限流、并发模拟、release_all |
| `test_plan_execute.py` | 10 | `ReAct/paths/plan_execute.py` | complex P&E:多步顺序、步骤隔离、缺失重试、synthesizer、预算截断、planner 降级、嵌套 trace |
| `test_agent_graph_stream.py` | 9 | `agent_reasoning/ReAct/core/` | 图流式执行、节点序列、状态流转 |
| `test_context_management.py` | 8 | `context management/` | 工具结果截断：错误透传、长文本截断并保留元数据、短文本不截断 |
| `test_escalation.py` | 7 | `server/chat/service.py` | 升级链:simple→medium→complex、最多一次、complex 不可升、失败重做、异常跳过质检 |
| `test_promotion_watermark.py` | 7 | `memories/orchestration/` | 短期→长期升迁水位线与失败计数 |
| `test_recall_gateway.py` | 7 | `memories/storage/long/recall_gateway.py` | 长期记忆召回网关、降级、过滤 |
| `test_metrics.py` | 5 | `server/support/metrics.py` | 按 tier 分维度(请求量/升级率/grounding/延迟/token)统计 |
| `test_tier_config.py` | 5 | `config/config.py`、`server/chat/service.py` | tier 配置结构、env 覆盖、service 按 tier 取预算 |
| `test_react_loop.py` | 3 | `agent_reasoning/ReAct/core/loop.py` | 抽取后的 react_loop 行为 |
| `test_long_term_dedup.py` | 5 | `memories/storage/long/` | 长期记忆去重 |
| `test_routing.py` | 4 | `server/chat/service.py` | 按 tier 分发到 simple/medium/complex、tier 首事件 |
| `test_persistence_three_paths.py` | 4 | `support/runner.py` | 三条路径都落短期流水 + 触发 after_stream(含异常路径) |
| `test_migrate.py` | 4 | `memories/db/migrate.py` | DB 迁移脚本 |
| `test_tier_flags.py` | 3 | `support/plan_grounding.py` 等 | tier 相关开关 |
| `test_simple_path.py` | 3 | `ReAct/paths/simple.py` | simple 单轮直答:无工具、lite 模型、流水/升迁 |
| `test_runner_wrapper.py` | 3 | `support/runner.py` | run_path 通用包装器:透传/落库/升迁/异常兜底 |
| `test_working_saver_fallback.py` | 2 | `memories/storage/working/` | PG checkpoint 不可用时回退内存 saver |
| `test_working_memory.py` | 0 | `tests/checkpoint_demo_graph.py` | **手动集成脚本**，见下文，不参与自动收集 |
| **合计（自动收集）** | **267** | **27 个文件** | |

辅助文件（非用例）：

- `tests/conftest.py`：sys.path 注入、环境隔离、共享 fixtures。
- `tests/checkpoint_demo_graph.py`：`test_working_memory.py` 使用的演示图与跨进程计数辅助。

### 手动集成脚本：test_working_memory.py

该文件验证「工作记忆 + PostgresSaver checkpoint 断点续跑」，用两个**独立 Python 进程**模拟程序关掉再重启，需要真实 PostgreSQL，因此不写 pytest 用例：

```bash
python tests/test_working_memory.py           # 跑完整两阶段（自动起子进程）
python tests/test_working_memory.py run       # 仅首轮
python tests/test_working_memory.py resume    # 仅恢复
```

## 测试架构

### conftest.py 做的事

- **sys.path 注入**：把项目根目录及 `config/`、`RAG/`、`server/`、`context management/` 加入 import 路径（在任何项目 import 之前执行）。
- **HF 镜像**：设置 `HF_ENDPOINT=https://hf-mirror.com`。
- **AUTH_DB 重定向**：导入后立即把 `config.AUTH_DB_PATH` 改到临时目录的 `semi_agent_pytest_auth.db`，避免污染开发库。
- **Fixtures**：
  - `tmp_auth_db`：每个用例独立的临时 SQLite auth 库，自动 `init_db`、用后清理。
  - `_reset_login_rate_limit`（autouse）：每个用例前清空 `auth.router._login_attempts` 登录限流状态。
  - `isolated_ratelimit`：重置全局限流状态（新信号量=2、空用户锁、队列超时 0.5s）。
  - `make_llm_response(content)`：构造 mock OpenAI ChatCompletion 响应对象。

### Mock 策略

| 目标 | Mock 方式 | 目的 |
|----------|----------|------|
| LLM (ReAct 主链路) | `@patch("agent_reasoning.ReAct.support.llm.get_client")` + `make_llm_response()` | 不产生真实 API 调用（LLM 客户端单例位于 `support/llm.py`） |
| LLM (grounding) | `@patch("agent_reasoning.ReAct.support.answer_grounding.get_client")` 或 `check_faithfulness` | 隔离忠实度/引用校验的外部调用 |
| SQLite | `tmp_path` / `tmp_auth_db` fixture | 用临时库，互不干扰 |
| Qdrant | 测试中 lazy import，按需 patch | 不连接真实向量库 |
| react_stream | `patch.object(chat_router_mod, "react_stream")` | SSE 测试中隔离 ReAct 内部逻辑 |
| 流式 client | `patch` 内部 client，用 `_chunk()` 构造流式 chunk | trace 测试验证事件序列 |

### 关键约定

1. **不调真实 LLM**：涉及 `verify_citations`、`check_faithfulness`、查询改写等外部调用的地方一律 mock。
2. **Lazy import 优先**：embed/LLM 等重资源在函数内部 import，避免收集阶段就加载模型。
3. **FastAPI TestClient**：SSE 与 Auth 测试统一用 `TestClient`。
4. **422 -> 400 转换**：`server/main.py` 注册了 `RequestValidationError` handler，参数校验失败返回 400。
5. **JWT 时间**：签发 token 用 `jwt.encode`；过期类用例需 mock 时间（`datetime.now` / `time.time`）。

## 新增测试约定

1. 在 `tests/` 下新建 `test_<module>.py`，测试类以 `Test` 开头。
2. 不要自己处理 import 路径，统一依赖 conftest.py 注入的 sys.path。
3. 需要 mock ReAct 主链路 LLM 时，用 `@patch("agent_reasoning.ReAct.support.llm.get_client")` + `make_llm_response()`。
4. 需要临时 auth DB 时，用 `tmp_auth_db` fixture。
5. 涉及限流时，用 `isolated_ratelimit` fixture；登录/注册限流由 autouse fixture 自动重置。

## 三级范式测试要点

新增/改动路由、质检、P&E 相关测试时遵循以下 mock 边界（均不调真实 LLM/Qdrant/PG）：

- **复杂度路由**（`test_router.py`）：patch `router.get_client` + `llm_create_with_retry`
  返回带 JSON content 的假响应，验证三分类与各类兜底（解析失败/低置信/超时/异常→medium）。
- **按 tier 分发**（`test_routing.py`）：patch `service` 的 `run_simple` /
  `run_agent_graph` / `run_plan_execute` 为桩生成器，`quality_check` 返回 passed，
  断言 `tier` 是首个事件且调用了正确路径。
- **质检门**（`test_quality_gate.py`）：直接调纯函数 `quality_gate.check`，按 tier
  传不同 `context`（`grounding` / `coverage`），验证三态与深度差异。
- **升级/重做**（`test_escalation.py`）：patch 路径桩记录调用顺序，用 `_gate(verdicts)`
  让 `quality_check` 依次返回不同 verdict，断言升级链、只升一次、complex 不升。
- **P&E**（`test_plan_execute.py`）：patch `plan_execute.react_loop`（side_effect
  按序调用工厂返回新生成器）、`llm_create_with_retry`（synthesizer 流）、`grounding_check`，
  验证步骤顺序、`step_instruction` 透传、`skip_recall=True`、缺失重试、预算截断、
  planner 降级以及 `trace["plan_execute"]` 嵌套结构。
- **三路径接线**（`test_persistence_three_paths.py`）：patch `persist_event` /
  `after_stream` / `short_term.append_event`，分别跑三条路径，断言 user_message
  落库、关键事件经白名单落库、结束后升迁恰好一次（异常路径也升迁）。
- **per-tier metrics**（`test_metrics.py`）：直接对 `Metrics` 单例记账并断言
  `get_stats()["by_tier"]` 与 `escalation_rate`。

### 已知非致命告警

`memories/storage/working/summarize.py:263` 在测试收尾时偶发
`PytestUnhandledThreadExceptionWarning: InvalidStateError: FINISHED`（守护线程对已
完成 Future 调 `set_exception`）。这是改造前就存在的问题，与三级范式无关，不影响
测试结果，可单独修复。
