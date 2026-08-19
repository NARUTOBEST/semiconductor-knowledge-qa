# 测试指南

## 运行测试

```bash
# 安装测试依赖
pip install -r requirements-test.txt

# 运行全部测试
python -m pytest

# 运行单个文件
python -m pytest tests/test_chunker.py

# 运行单个测试类
python -m pytest tests/test_auth.py::TestJWT

# 运行单条用例
python -m pytest tests/test_sse.py::TestSSEEventFormat::test_basic_stream

# 详细输出 + 完整回溯
python -m pytest -v --tb=long
```

## 测试清单

| 文件 | 用例数 | 覆盖模块 | 说明 |
|------|--------|----------|------|
| `test_chunker.py` | 28 | `RAG/chunker.py` | 分块装箱、标题切段、过滤项、图片/表格/公式、合并与超大切分、页码范围 |
| `test_grounding.py` | 21 | `chat/grounding/check.py` | 引用校验 + 忠实度检查,均 mock LLM,覆盖 grounding 全链路 |
| `test_auth.py` | 25 | `auth/` 全套 | 密码哈希(bcrypt)、JWT(签发/校验/过期/篡改)、注册/登录/me API、DB 操作 |
| `test_ratelimit.py` | 10 | `server/support/ratelimit.py` | 单用户限流、全局限流、并发模拟、release_all |
| `test_query_rewrite.py` | 13 | `chat/query/rewrite.py` | LLM 查询改写成功/回退/历史处理/markdown 清洗等 |
| `test_sse.py` | 15 | `chat/router.py` | SSE 事件格式、错误处理、鉴权、参数校验 |
| `test_trace.py` | 12 | `chat/service.py` | ReAct 结构化事件(step_start/tool_call/tool_result/grounding/error/done)与完整 trace |
| `test_context_management.py` | 8 | `context_management/` | 工具结果截断:错误透传、长文本截断并保留元数据、短文本不截断 |
| **合计** | **132** | **8 个模块** | |

## 测试架构

### conftest.py 做的事

- **sys.path 注入**:把 `config/`、`RAG/`、`server/`、`context management/` 及项目根目录加入 import 路径(在任何项目 import 之前执行)。
- **HF 镜像**:设置 `HF_ENDPOINT=https://hf-mirror.com`。
- **AUTH_DB 重定向**:导入后立即把 `C.AUTH_DB_PATH` 改到临时目录的 `semi_agent_pytest_auth.db`,避免污染开发库。
- **Fixtures**:
  - `tmp_auth_db`:每个用例独立的临时 SQLite auth 库,自动 `init_db`、用后清理。
  - `_reset_login_rate_limit`(autouse):每个用例前清空 `auth.router._login_attempts` 登录限流状态。
  - `isolated_ratelimit`:重置全局限流状态(新信号量=2、空用户锁、队列超时 0.5s)。
  - `make_llm_response(content)`:构造 mock OpenAI ChatCompletion 响应对象。

### Mock 策略

| 目标 | Mock 方式 | 目的 |
|----------|----------|------|
| LLM (OpenAI API) | `@patch("chat.service.get_client")` + `make_llm_response()` | 不产生真实 API 调用(get_client 现位于 `chat.service`) |
| SQLite | `tmp_path` / `tmp_auth_db` fixture | 用临时库,互不干扰 |
| Qdrant | 测试中 lazy import,按需 patch | 不连接真实向量库 |
| react_stream | `patch.object(chat_router_mod, "react_stream")` | SSE 测试中隔离 ReAct 内部逻辑 |
| chat.service 流式客户端 | `patch` 内部 client,用 `_chunk()` 构造流式 chunk | trace 测试验证事件序列 |

### 关键约定

1. **不调真实 LLM**:涉及 `verify_citations`、`chunk_content_list` 等外部调用的地方一律 mock。
2. **Lazy import 优先**:embed/LLM 等重资源在函数内部 import,避免收集阶段就加载模型。
3. **FastAPI TestClient**:SSE 与 Auth 测试统一用 `TestClient`。
4. **422 -> 400 转换**:`main.py` 注册了 `RequestValidationError` handler,参数校验失败返回 400。
5. **JWT 时间**:签发 token 用 `jwt.encode`;过期类用例需 mock 时间(`datetime.now` / `time.time`)。

## 新增测试约定

1. 在 `tests/` 下新建 `test_<module>.py`,测试类以 `Test` 开头。
2. 不要自己处理 import 路径,统一依赖 conftest.py 注入的 sys.path。
3. 需要 mock LLM 时,用 `@patch("chat.service.get_client")` + `make_llm_response()`。
4. 需要临时 auth DB 时,用 `tmp_auth_db` fixture。
5. 涉及限流时,用 `isolated_ratelimit` fixture;登录/注册限流由 autouse fixture 自动重置。
