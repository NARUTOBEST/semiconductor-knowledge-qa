# SSE 事件契约

后端 `/api/chat` 以 `text/event-stream` 返回,每个事件为一行
`data: {"type": "...", ...}\n\n`(JSON,UTF-8)。前端按 `type` 分发。

阶段 3–7 改造后,所有请求先发 `tier` 事件告知所选推理范式,再按路径产出不同
序列;质检不通过时可能发 `escalation` 升级到更高 tier 并重跑(最多一次)。

## 三条路径的事件序列

### simple(单轮直答,lite 模型,不绑工具)

```
tier(tier=simple)
status("思考中…")
token*
assistant_message
meta
done
```

无 `step_start` / `tool_call` / `tool_result`。simple 路径不做检索,因此正常
情况下不发 `grounding`;若质检门判定答案涉及领域事实,会发 `escalation` 升级
medium 而非直接放行。

### medium(ReAct 检索循环)

```
tier(tier=medium)
status("理解问题中…")
[ step_start → token* → llm_response
  → tool_call → tool_result → step_end ]*   (0~N 轮,有界)
[ status("验证答案来源…") → grounding ]?      (有检索来源时)
[ plan / status("已生成 N 步检索计划") ]?      (复杂问题条件触发规划)
[ reflect ]?                                  (grounding 未过、反思重生成时)
assistant_message
meta
done
```

`reflect` 表示旧回答作废、正在重生成;前端收到应**清空当前正在流式输出的内容**,
后续 `token` 属于新回答。

### complex(Plan-and-Execute)

```
tier(tier=complex)
plan(steps, question)                       (必经规划)
# 对每个 step:
status("执行计划第 i/N 步…")
step_start → token* → [tool_call → tool_result]* → step_end
[status("第 i 步未检索到结果,更换关键词重试…") → 再来一轮]?
synthesis_start
token*
assistant_message
grounding
meta
done
```

每步独立 messages / state / 检索来源(步骤间隔离);某步两次都搜不到结果则标记
缺失并继续。synthesizer 失败时降级为拼接各步结果并发 `status` 提示。planner
失败则整个 complex 路径降级为普通 ReAct(发 `status("检索规划不可用…")`)。

## 跨路径 / 控制类事件

| type | 字段 | 说明 |
|------|------|------|
| `tier` | `tier`(simple/medium/complex)、`confidence`、`source`(rule/llm/fallback) | 流的第一个事件,告知前端本次所选范式 |
| `escalation` | `from_tier`、`to_tier`、`reason` | 质检判定当前 tier 不足,升级重跑;前端应**清空已输出内容**并展示"正在深入分析…"。整条请求最多一次 |
| `status` | `message` | 状态提示文案(检索中、重试、降级、质检警示等) |
| `error` | `message`、`trace_id?` | 面向用户的错误;流仍会以 `done` 收尾 |
| `error_trace` | `phase`、`error_type`、`message`、`traceback_preview` | 调试用结构化错误(前端一般不展示) |
| `meta` | `elapsed_ms`、`steps`、`tokens`、`tools_count`、`grounding_passed`、`sources_count` | 成本/性能元数据 |
| `done` | `trace_id`、`trace?` | 流结束。`trace` 含完整结构化追踪 |
| `token` | `delta`、`step?` | 流式文本增量 |
| `assistant_message` | `content`、`trace_id?` | 最终答案全文(token 已累积产出,此处为确认) |
| `step_start` / `step_end` | `step`、`elapsed_ms?`、`decision?` | ReAct / P&E 步骤边界 |
| `tool_call` | `name`、`args`、`args_parse_error?` | 工具调用 |
| `tool_result` | `name`、`ok`、`duration_ms`、`result_preview`、`error?` | 工具结果 |
| `sources` | `items` | 聚合后的引用来源(若路径显式发送) |
| `plan` | `steps`、`question` | P&E / medium 生成的检索计划 |
| `synthesis_start` | `trace_id` | P&E 开始综合各步结果 |
| `grounding` | `passed`、`warnings` | 引用/忠实度校验结果 |
| `reflect` | `feedback?` | 反思重生成(清空旧回答) |
| `llm_response` | `finish_reason`、`usage`、`has_tool_calls`、… | 一轮 LLM 响应元数据 |

## 前端处理要点

1. **`tier`** 永远是第一个事件,可据此切换 UI(如 complex 展示计划/步骤进度)。
2. **`escalation` 与 `reflect` 都要清空当前流式输出区**:`escalation` 表示整段
   答案将由更高 tier 重新生成;`reflect` 表示同 tier 内重生成。两者都把正在拼接
   的 assistant 消息 `content` 置空,并展示状态文案。
3. simple 路径没有 step/tool 事件,不要依赖它们才能渲染。
4. 任何路径出错都会最终发 `done`(或 HTTP 层补一个),前端在 `done` 时复位
   streaming 态;即使中途收到 `error`,也应继续读到 `done`。
5. 未知 `type` 应**静默忽略**(前向兼容),不要报错。
