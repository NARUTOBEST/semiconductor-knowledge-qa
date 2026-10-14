# 20261015 席位绑定 + 冲突源限定反问 + 检索超时放宽(10u 全量 0.8821/0.9712/4.35% 三项历史最好)

## 改动点

1. **复合题子检索席位绑定**(用户口径:拆子问题后每个子问题各自的 top-4 绑定进上下文,
   不照单问题总 top-4):
   - service `_speculative_search`:子路 top-`RETRIEVAL_SUBQUERY_PIN_K`(=4) 打 `sub_pin=True`,
     去重追加,`RETRIEVAL_SUBQUERY_MERGE_CAP` 不截席位块;
   - 下游保席位:raglite `_pin_select` / compressor `compress_chunks`(返回列表与输入等长对齐)/
     react `_pin_fill` + 两个 source extractor 透传,保证 blocks/grounding/sources(评测召回口径)
     都含席位块。
2. **20261013 版子查询合并死代码修复**:`call_with_resilience` 返回 `(result, err)` 元组,
   旧代码当 dict 判定恒 False——拆分合并自创建起从未执行。主/子路全部改元组解包 +
   回归钉子 test_tuple_unpack_regression;MCP 工具 `-> dict` 注解 + 裸列表返回触发
   FastMCP output schema 校验失败(structured_content 变 {})→ tools.py 包一层
   `{"items": out}`。
3. **冲突即源限定反问**(20261015 产品规则,用户拍板):`RETRIEVAL_CONFLICT_CLARIFY` 默认开;
   投机检索冲突信号(top-4 跨文档族)经 `_conflict_objects` specific 前置判定后,非具体问题
   直接发源限定反问句(`_SOURCE_CLARIFY_QUESTION`,与轮次21 Path B 共用),应答轮
   `_scope_resolve` → infer 锁 stem → scope_hard 硬过滤。题面无限定词也触发(dev_043 类)。
4. **检索超时放宽**(10u 全量 68% srcs=0 治理):search_text/image 策略超时 30→90s;
   mcp_bridge 读超时 60s 写死改 `MCP_READ_TIMEOUT_S` 默认 180s;breaker threshold 2→4、
   cooldown 30→20s(env.env 定格)。

## 根因

- 复合题金块分布在多个子问题的检索结果里,单问题总 top-4 装不下 → 主题漂移挤掉金块
  (dev_018 池拥挤悬崖);20261013 误判为"索引缺口",实为排序+死代码。
- 10u 全量 srcs=0 68%:新架构(席位绑定+冲突探测)每题检索调用 ~3 倍,F40(4080 vGPU)
  rerank 排队偶发 >30s → 30s 策略超时连环重试失败 → 熔断开闸 30s 内全员秒失败。
  GPU 侧全程 961 次 200/零 5xx、10 隧道全通,排除链路故障。

## 对照数据(dev200 全量,judge=ARK coding plan doubao-seed-2-0-lite,含 40 题补答)

| 指标 | 本轮 (runs/20261015_v2_judged) | cascade_v2 基线 | 变化 |
|---|---|---|---|
| context_recall | **0.8821** | 0.8195 | **+6.3pp** |
| faithfulness | **0.9712** | 0.9587 | +1.3pp |
| 幻觉率 | **4.35%**(8/184) | 6.04% | -1.7pp |
| answer_relevancy | 0.9823 | — | — |
| context_precision | 0.5267 | 0.6690 | **-14.2pp** |

- **recall/faith/幻觉率三项历史最好**;badcase 清账:dev_018(金块 t00256 席位 #1)、
  dev_176/069(srcs 4→8/9)、dev_043(原"无解",冲突反问→magnatran_71→0.1mm TIR 正确)、
  dev_124(源限定"未涉及"口径正确)。
- **precision -14.2pp 为席位绑定的结构性代价**:每题上下文从 4 块扩到 ~12 块
  (主路 cap 8 + 子路席位 4×N),不相关块占比上升。recall +6.3pp 与生成侧 faith/幻觉
  双改善说明扩上下文净收益为正;precision 目标(≥0.80)与"多块上下文"的结构性冲突
  进一步加剧,口径需重校准(见 README plateau 结论)。

## 运维注

- **F40机(西北B区,克隆自E70)**:qdrant 二进制本次直启成功(旧 glibc loader 步骤未触发,
  克隆带全 /root/glibc238 与 /root/autodl-tmp);deploy_gpu.sh 会幂等误启 vLLM :8003,
  部署后必杀(`pgrep -f "[v]llm serve"`);VM 侧 deploy_vm 后须 `sudo docker compose restart backend`
  (up -d 不会重挂 bind mount 内的代码变化)且隧道目标文件要 cp 到 ~/projectdocker/。
- **并发容量**:F40 实测 20u 超容(150-225s/题),10u 为上限(与 cascade_v2 一致);
  20×10 分片形态需 user11..20.json,qa_sets 默认只有 1..10,勿直接 --users 20。
- **judge 三个坑**:① --results-dir 与 --out-dir 不可同目录(边读边写污染 eval*.jsonl,
  total 会虚增到 356);② 判分源文件必须命名 eval*.jsonl 且含 user 字段;③ --meta 缺失时
  拒答题不单列(静默),跑前确认 meta.json 在 results-dir。
- **评测补答流程**:空答案(answer='')/无源(srcs=0)题用 run_http.py --ids 单用户补跑
  新目录,合并时按条件覆盖(空答案用补答1、无源用补答2),judge 前断言 empty_left=0 且
  srcs0_left=0。本轮 10u 主跑空答案 24 题(~12%,coding plan 偶发空回,与 GPU 无关)+ 无源
  27 题,补答后全部恢复。
- 超时回退:env 置 `MCP_READ_TIMEOUT_S=60` + mcp_policies timeout 恢复 30 即回旧口径。
