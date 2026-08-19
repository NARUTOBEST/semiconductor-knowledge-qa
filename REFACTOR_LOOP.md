# 三级范式架构改造 — Claude Code 任务循环

> **用法**：对 Claude Code 说「读 REFACTOR_LOOP.md，开始任务循环」。
> Claude Code 每次只做一个任务，做完验证、打勾、提交，再读下一个。

---

## 你的角色

你是本项目（半导体知识 RAG Agent）的开发者。你正在把现有的单体 ReAct 图改造成**复杂度路由 + 三种推理范式**的架构。你严格按照本文件的任务清单顺序执行，每完成一个任务就更新本文件的勾选状态。

**不要一次性做多个任务。不要跳任务。遇到设计决策问题，停下来问用户，不要自己拍板。**

---

## 项目上下文

### 技术栈

- Python 3.14.6，FastAPI，LangGraph 1.2.11，langchain-core 1.5.4，langchain-openai 1.5.0
- 向量库 Qdrant（本地文件模式），嵌入 BGE-m3 + CLIP，重排 BGE-reranker-v2-m3
- 记忆系统：PostgreSQL 三库（working/short/long）+ pgvector + 可选 Redis
- 前端 Next.js 14（`web/` 目录）
- 测试 pytest，17 个测试文件

### 关键目录结构

```
config/config.py                         # 全局配置（路径、模型、密钥、限流参数）
env/env.env                              # 密钥（gitignore）；env.example 是模板
agent_reasoning/ReAct/
  core/graph.py                          # LangGraph 图定义（当前单体图）
  core/state.py                          # AgentState TypedDict + reducers
  core/nodes.py                          # 所有节点（43KB，主要改造对象）
  support/runner.py                      # run_agent_graph 入口（trace/checkpoint/流水/升迁）
  support/llm.py                         # LLM 客户端单例 + 重试 + 主备模型切换
  support/answer_grounding.py            # 引用校验 + 忠实度检测
  support/plan_grounding.py              # CoverageTracker 异步覆盖度追踪
  trace.py                               # TraceRecorder
memories/                                # 三层记忆（working/short/long）
  storage/                               # 存储访问层（PG 连接池、长期记忆、升迁、摘要压缩）
  orchestration/                         # 编排层（事件落库、升迁触发、TTL 清理）
RAG/                                     # 切块、嵌入、入库、检索
tools/
  dispatch.py                            # 工具分发
  search_tools.py                        # search_text/search_image/get_chunk schema + HTTP 调用
  _common.py                             # 路径注入 + payload 转换
server/
  main.py                                # FastAPI 应用入口
  chat/router.py                         # POST /api/chat → SSE
  chat/service.py                        # react_stream 薄封装
  chat/conversation/                     # 会话 CRUD（SQLite）
  auth/                                  # JWT + bcrypt + SQLite
  support/ratelimit.py                   # 双层限流
  support/metrics.py                     # 内存指标
  retrieval_service.py                   # 检索微服务（:8002）
  admin/                                 # PDF 上传 + 处理流水线
context management/                      # 消息构建、查询改写、工具结果截断、system prompt
tests/                                   # pytest 测试
  conftest.py                            # sys.path 注入、mock fixtures
web/                                     # Next.js 前端
```

### 当前图拓扑（改造前）

```
START → setup → recall → rewrite → plan → build_messages → agent
                                              ┌─ tools ─┘
                                              └─ reflect（coverage + grounding）
                                              → finalize → END
```

- `plan_node`：`_looks_complex()` 粗筛命中才调 LLM 生成 steps，塞进 system prompt 当软引导
- `reflect_node`：同时做覆盖度回退 + grounding 校验 + 反思重生成（职责过载，要拆）
- 所有问题都走这一张图，模型用 `OPENAI_TEXT_MODEL`

### 目标架构

```
用户问题
  ↓
[复杂度路由器]  doubao-seed-2.0-lite 分类
  ├── simple  → 单轮直答（无工具，lite 模型）
  ├── medium  → ReAct 循环（deepseek-v4-flash，plan 软引导条件触发）
  └── complex → Plan-and-Execute（deepseek-v4-flash，plan 硬驱动 + 逐步 react_loop + synthesizer）
  ↓
[共享质检门]  按 tier 深浅不同；返回 passed / failed / needs_escalation
  ├── passed → 返回
  ├── failed → 退回当前路径重做（有界）
  └── needs_escalation → 升级到更高 tier（simple→medium→complex，最多一次）
  ↓
答案
```

### 编码约定（必须遵守）

1. **路径注入**：项目用裸 import（`import config`、`import embed`），靠 `sys.path.insert` 把 `config/`、`RAG/`、`server/`、`context management/` 加入路径。新文件放在哪个目录，参考同目录现有文件的路径注入写法。**不要改成相对 import 或装包**，保持与现有代码一致。
2. **注释用中文**，文件头用 `# -*- coding: utf-8 -*-` + docstring 说明职责。
3. **配置走 `config.py` + 环境变量**，不要硬编码模型名、超时、阈值。
4. **LLM 调用统一走 `support/llm.py` 的 `llm_create_with_retry`**（重试 + 主备切换），不要直接调 `client.chat.completions.create`。记忆后台任务走 `memories/storage/_llm.py`。
5. **SSE 事件是 dict**，通过 `get_stream_writer()`（图内）或 `yield`（runner）发出，格式与现有事件保持一致。
6. **降级原则**：所有旁路功能（记忆、规划、grounding、重排）失败都不能阻断主对话，catch 后降级 + 可见警示。
7. **有界执行**：所有循环都有步数/时长/次数上限，防止死循环。
8. **日志用 `logging.getLogger("agent")`**，不要用 print（启动脚本和 `__main__` 自测除外）。

### 测试约定

- 测试在 `tests/` 下，文件名 `test_*.py`，统一靠 `conftest.py` 注入 sys.path，不要自己处理路径。
- Mock LLM 用 `@patch` + `conftest.make_llm_response()`；临时 auth DB 用 `tmp_auth_db` fixture；限流用 `isolated_ratelimit` fixture。
- 不调真实 LLM / Qdrant / PG。
- 跑测试命令：`C:\python\python.exe -m pytest -p no:langsmith`（环境缺 typing_extensions，langsmith 插件会导致收集失败；若已修复则直接 `python -m pytest`）。
- **每个任务完成后必须跑全量测试，全绿才算完成。**

### 已知环境问题

- 当前 Python 环境缺 `typing_extensions` 包，导致 langchain_core 无法 import、8 个测试文件无法收集。第一个任务应先修复（`pip install typing_extensions` 或加入 requirements）。
- 项目根无 `.git`，改造开始前应先 `git init` 并做首次提交。

---

## 工作循环（LOOP）

每一轮严格按以下步骤执行：

1. **读本文档**，找到第一个 `- [ ]` 未完成任务。
2. **检查前置任务**：该任务依赖的前序任务是否都已 `[x]`？没有则先做前序。
3. **理解任务**：阅读任务涉及的现有代码，弄清改造点。如果任务描述不够具体，先在代码中定位再动手。
4. **实现**：只改这个任务需要改的代码，不要顺手重构无关部分。遵循编码约定。
5. **写/改测试**：新功能必须有对应测试；改了行为要更新旧测试。
6. **验证**：
   - 跑相关测试文件，确认新测试通过
   - 跑全量 `python -m pytest -p no:langsmith`，确认无回归
   - 如果任务涉及 SSE 事件，检查事件格式与现有契约一致
7. **更新本文档**：把该任务的 `- [ ]` 改成 `- [x]`。
8. **提交**：`git add -A && git commit -m "<阶段号>: <任务简述>"`（如果 git 已初始化）。
9. **回到第 1 步**，继续下一个任务。

### 停止条件

遇到以下情况，**停下来问用户，不要继续**：

- 任务涉及「需用户拍板的设计决策」（见下方章节）且用户尚未决定
- 发现任务清单有遗漏或错误，需要调整
- 全量测试无法通过且原因不是本任务引入的（环境问题等）
- 需要引入新的第三方依赖
- 需要修改 `.env` 或密钥配置
- 任何不确定的架构取舍

---

## 任务清单

### 阶段 0：环境与基线

- [x] **0.1** 修复环境：安装缺失的 `typing_extensions`；把它加入 `requirements.txt`（langchain-core 的依赖，应显式锁定）。验证 `python -m pytest -p no:langsmith` 能正常收集并全绿（基线 132+ passed）。
- [x] **0.2** `git init`，配置 `.gitignore`（已存在，确认覆盖新产物），首次提交当前代码作为基线。
- [x] **0.3** 更新 `TESTING.md`：同步实际测试文件数（当前 17 个）和用例数；修正文档中过时的路径描述。

### 阶段 1：重构准备 — 抽取可复用核心

- [x] **1.1** 把 `agent_reasoning/ReAct/core/nodes.py` 中 `agent_node ↔ tools_node` 的循环抽成独立的 `react_loop()` 函数（放在 `agent_reasoning/ReAct/core/loop.py`）。它接受初始 messages、step 指令（可选，P&E 每步传入）、max_steps、max_total_seconds、checkpointer、configurable，yield SSE 事件 dict 并返回最终 state。现有图改为调用这个函数，行为不变。
- [x] **1.2** 拆分 `reflect_node` 为两个独立节点：
  - `coverage_check_node`：计划覆盖度判定 + 回退逻辑（原 `_maybe_coverage_rollback` 相关）
  - `grounding_node`：引用校验 + 忠实度检测 + 反思重生成
  - 更新 `graph.py` 的边：agent → tools → agent → coverage_check →（回退 | grounding）→ finalize。
- [x] **1.3** 把 `setup / recall / rewrite / build_messages` 等前置节点做成可按 tier 开关：`react_loop()` 接受参数控制是否跳过 rewrite、是否跳过 recall、是否绑定 tools。simple 路径需要跳过 rewrite 和 tools。
- [x] **1.4** 泛化 `support/runner.py`：把 trace 初始化、checkpoint、短期流水落库、长期升迁触发、异常兜底等横切逻辑抽成一个通用的 `run_path(path_name, event_iterable, thread_id, username, session_id, trace_id)` 包装器，能包裹 simple / react / P&E 任意一条事件流。`run_agent_graph` 改为调用它。
- [x] **1.5** 全量测试通过，现有测试无需大改（如果测试断言了具体事件序列，按拆分后的新序列更新）。

### 阶段 2：简单路径（single-shot, no tools）

- [x] **2.1** 新建 `agent_reasoning/ReAct/paths/simple.py`：一次 LLM 调用，不绑定 `tools` 参数，模型用 `config.TIER_MODEL_SIMPLE`（新增配置，默认 `doubao-seed-2.0-lite`）。yield `status → token* → assistant_message` 事件。
- [x] **2.2** 裁剪 system prompt：新建 `context management/system_prompt_simple.py`（或在 `system_prompt.py` 加 `SIMPLE_SYSTEM_PROMPT`），只保留角色定义 + 安全规则，去掉工具说明和引用规则。
- [x] **2.3** simple 路径跳过 `rewrite_node`（不调改写 LLM）；保留 `recall`（长期记忆召回，便宜的向量检索，用于个性化）。
- [x] **2.4** simple 路径接入阶段 1.4 的通用 runner 包装器（trace、流水、升迁、checkpoint 都要有）。
- [x] **2.5** 写测试：`tests/test_simple_path.py`，mock LLM，验证不调工具、事件序列正确、模型用 lite、流水落库。

### 阶段 3：复杂度路由器

- [x] **3.1** 新建 `agent_reasoning/router.py`：`classify_complexity(question, history) -> {tier, confidence}`，用 `config.TIER_MODEL_SIMPLE`（lite 模型）做一次短调用，prompt 只输出 `simple`/`medium`/`complex`，超时 5-10s，失败默认 medium。
- [x] **3.2** 分类 prompt 设计：simple 仅限闲聊/元问题/明确不需要领域知识；**领域事实题哪怕很短也走 medium**；complex 为多子问题、多维度对比、含"对比/分别/优缺点/流程/步骤/综合"等特征。参考现有 `_COMPLEX_MARKERS` 和 `_looks_complex` 逻辑。
- [x] **3.3** 置信度阈值：LLM 输出解析不出或置信度低时兜底 `medium`。加规则预筛：纯问候/超短无领域术语可直接判 simple，省一次调用（可选优化）。
- [x] **3.4** `server/chat/service.py` 的 `react_stream` 改为：先调路由器 → 按 tier 分发到 simple / react / P&E 路径。thread_id/username/session_id 透传。
- [x] **3.5** 流开始时发 `{"type": "tier", "tier": "simple|medium|complex"}` 事件，告知前端。
- [x] **3.6** 写测试：`tests/test_router.py`，mock LLM 输出，验证三分类、解析失败兜底 medium、低置信度兜底 medium；`tests/test_routing.py` 验证分发到正确路径。

### 阶段 4：共享质检门 + 升级通道

> 这是安全网，必须在 P&E（阶段 5）之前完成。

- [ ] **4.1** 新建 `agent_reasoning/quality_gate.py`：`check(answer, context, tier) -> {verdict, feedback}`，verdict ∈ `passed` / `failed` / `needs_escalation`。
- [ ] **4.2** 按 tier 实现检查深度：
  - simple：安全/格式检查；若答案涉及领域事实内容（启发式：含半导体术语/型号/参数）→ `needs_escalation`
  - medium：调用现有 `grounding_check`（引用 + 忠实度）
  - complex：引用 + 忠实度 + 覆盖度（复用阶段 1.2 拆出的覆盖度检查）
- [ ] **4.3** 在 runner 包装器中接入质检门：答案产出后、`done` 事件前调用。
  - `failed`：带 feedback 退回当前路径重做（重做次数有界，建议最多 1 次，复用现有 reflect 机制）
  - `needs_escalation`：升级 tier 重跑（见 4.4）
- [ ] **4.4** 实现升级链：simple→medium，medium→complex，**最多升级一次**。升级时带上已有对话历史和失败反馈，发 `{"type": "escalation", "from_tier": ..., "to_tier": ...}` 事件，前端据此重置流式输出区。complex 无法再升，质检失败就带警示放行。
- [ ] **4.5** 质检 LLM 失败时 fail-open 带可见警示（沿用现有 `grounding_check` 异常处理逻辑）。
- [ ] **4.6** 写测试：`tests/test_quality_gate.py`（三态判定、各 tier 深度）；`tests/test_escalation.py`（simple 领域题升级 medium、medium 多子问题升级 complex、只升一次、升级事件格式）。

### 阶段 5：Plan-and-Execute 路径

- [ ] **5.1** 抽 `_generate_plan(question)` 共享函数（从现有 `plan_node` 提取 LLM 调用 + JSON 解析逻辑），放 `agent_reasoning/ReAct/support/planning.py`，medium 和 complex 都能调。
- [ ] **5.2** 新建 `agent_reasoning/ReAct/paths/plan_execute.py`：
  - 第一步调 `_generate_plan()` 生成 steps（必经，非条件触发）
  - 发 `plan` 事件
  - 对每个 step 调阶段 1.1 的 `react_loop(step_instruction=step)`，每步独立 messages/state，步骤间隔离
  - 收集每步结果
- [ ] **5.3** 步骤失败处理：某步搜不到结果 → 换词重试 1 次 → 仍失败则跳过，标记该步缺失，不中断整体。
- [ ] **5.4** 新建 synthesizer：一次 LLM 调用，输入 = 原问题 + 各步结果（含缺失标记），输出最终整合答案。yield `synthesis_start → token* → assistant_message`。
- [ ] **5.5** 跨步骤总预算：P&E 整体受 `max_total_seconds` 和总 token 预算约束，N 个子循环不能各算各的导致乘爆。预算不足时跳过剩余步骤直接 synthesizer。
- [ ] **5.6** P&E 的 grounding 在 synthesizer 之后做（质检门 complex 深度），校验最终答案是否忠于各步结果。
- [ ] **5.7** 评估 `CoverageTracker` 在 P&E 中的去留：结构上每步已执行，覆盖度由步骤完成度保证。决定保留（检查 synthesizer 是否漏用某步结果）或移除，并更新代码。
- [ ] **5.8** 降级：planner 失败 → 降级普通 ReAct（发 status 告知用户）；synthesizer 失败 → 拼接各步结果返回。
- [ ] **5.9** 写测试：`tests/test_plan_execute.py`，mock LLM 和 react_loop，验证多步执行顺序、步骤隔离、步骤失败跳过、synthesizer 调用、总预算截断、planner 失败降级。

### 阶段 6：plan_node 双路径逻辑区分

- [ ] **6.1** medium ReAct 路径：plan 条件触发（保留 `_looks_complex` 粗筛），steps 作为 system prompt 软引导（现有行为不变），但调用的是阶段 5.1 抽出的共享 `_generate_plan()`。
- [ ] **6.2** complex P&E 路径：plan 必经，steps 驱动阶段 5.2 的 executor 循环（硬约束）。
- [ ] **6.3** 删除 `nodes.py` 中旧的 `plan_node` 内联实现，改为调用共享函数；确认两条路径的 plan 行为差异正确。
- [ ] **6.4** 全量测试通过。

### 阶段 7：SSE 事件契约 + 前端适配

- [ ] **7.1** 确认三条路径的事件序列文档化（写在 `agent_reasoning/ReAct/__init__.py` 或单独的 `docs/sse_events.md`）：
  - simple：`tier → status → token* → assistant_message → grounding? → done`
  - medium：`tier → status(setup) → ...现有序列... → done`
  - complex：`tier → plan → step_start*(每步) → tool_call/tool_result/token* → step_end* → synthesis_start → token* → assistant_message → grounding → done`
- [ ] **7.2** 新增 `escalation` 事件格式：`{"type": "escalation", "from_tier": str, "to_tier": str, "reason": str}`。
- [ ] **7.3** 改 `web/` 前端 SSE 解析：
  - 收到 `tier` 事件可展示对应 UI（如 complex 展示计划步骤进度）
  - 收到 `escalation` 事件重置流式输出区，展示"正在深入分析..."
  - 现有 `reflect` 事件的重置逻辑复用于 escalation
  - simple 路径无 step/tool 事件时正常渲染
- [ ] **7.4** 前端手动测试三条路径 + 升级场景（mock 或真实后端）。

### 阶段 8：记忆 / Trace / Metrics 全路径接线

- [ ] **8.1** 确认三条路径都通过阶段 1.4 的通用 runner 写短期流水（白名单：user_message / assistant_message / tool_call / tool_result / grounding / error / done）。
- [ ] **8.2** 确认三条路径结束后都触发 `after_stream` 长期升迁。
- [ ] **8.3** `TraceRecorder` 支持 P&E 嵌套结构：trace 中包含 planner 步骤 + 每步独立的 tools/llm 记录 + synthesizer 记录。更新 `trace.py` 的数据结构和 `to_dict()`。
- [ ] **8.4** `server/support/metrics.py` 按 tier 分维度：请求量、升级率、grounding 通过率、p50/p95 延迟、token 消耗。`/metrics` 端点返回 per-tier 统计。
- [ ] **8.5** 写测试验证三条路径的流水落库和升迁触发。

### 阶段 9：配置

- [ ] **9.1** `config/config.py` 新增 tier 配置：
  - `TIER_MODEL_SIMPLE`（默认 `doubao-seed-2.0-lite`）
  - `TIER_MODEL_MEDIUM`（默认 `deepseek-v4-flash`，即现有 `OPENAI_TEXT_MODEL`）
  - `TIER_MODEL_COMPLEX`（默认 `deepseek-v4-flash`）
  - 各 tier 的 `max_steps`、`max_total_seconds`、是否启用 plan、质检深度
  - 路由器超时、置信度阈值
- [ ] **9.2** `env/env.example` 同步更新所有新配置项，带注释。
- [ ] **9.3** 所有 tier 参数可通过环境变量覆盖。

### 阶段 10：最终验证

- [ ] **10.1** 全量测试通过。
- [ ] **10.2** 启动完整服务（`启动配置/run_agent.ps1`），手动验证：闲聊走 simple 且快、领域问题走 medium、多步对比走 complex、分错时能升级。
- [ ] **10.3** 检查日志无异常、metrics 端点正常、健康检查正常。
- [ ] **10.4** 更新 `TESTING.md` 和 `SESSION_HANDOFF.md`（或新建 `README.md`）记录新架构。

---

## 需用户拍板的设计决策

以下决策在对应任务开始前必须确认，**不要自行决定**：

1. **simple 路径是否保留长期记忆召回（recall）？**
   - 建议：保留（向量检索便宜、能个性化），但多一次 Qdrant 查询。
2. **medium 的 plan_node 是条件触发还是完全关掉？**
   - 建议：条件触发（保留 `_looks_complex`），单一意图问题不规划。
3. **P&E 步骤间是否共享检索来源？**
   - 建议：隔离（每步独立上下文），最后全量传 synthesizer；避免步骤间干扰。
4. **路由器用 LLM 分类还是纯规则？**
   - 建议：LLM 分类（lite 模型）+ 规则预筛兜底；纯规则对"短但需检索"的题判断不准。
5. **simple 路径发现领域内容时，是直接升级还是让 lite 模型拒答？**
   - 建议：直接升级 medium（用户体验更好，不中断）。

开始阶段 2 之前问 1、5；开始阶段 3 之前问 4；开始阶段 5 之前问 3；开始阶段 6 之前问 2。

---

## 完成标准

- 所有任务 `[x]`
- `python -m pytest` 全绿（无 `-p no:langsmith` 兜底，环境问题已修）
- 三条路径手动验证通过
- 升级通道端到端验证通过
- 前端适配三种事件形态
- 无新增 lint/语法警告
- 每个任务有独立 commit
