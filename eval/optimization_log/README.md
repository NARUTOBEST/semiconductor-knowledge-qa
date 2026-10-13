# dev 指标优化记录

每轮优化一个条目:方法、动机、结果、结论(保留/回退)。当前生产配置与历史见
`HISTORY.md`;单轮详情见对应 `2026MMDD_<slug>.md`。

## 指标口径(ragas,judge=ARK ep-20260917012501-44sxb,dev-200 全量)

| 指标 | 当前目标 | 说明 |
|---|---|---|
| faithfulness 忠实度 | ≥ 0.90 | 答案被来源支撑的程度 |
| 幻觉率 | < 0.10 | 1 - faithfulness 口径(run 报告) |
| context_recall 检索率 | ≥ 0.80 | 金块/facts 覆盖 |
| context_precision 精确率 | ≥ 0.80 | 来源排序质量 |

## 轮次索引

| 日期 | 方案 | faith | halluc | recall | precision | 结论 |
|---|---|---|---|---|---|---|
| 2026-09-18 | 基线(dev200_baseline) | 0.8133 | 25.52% | 0.6116 | 0.5297 | — |
| 2026-10-04 | 方案A 术语表双路 + top-5 | 0.8494 | 21.55% | 0.6719 | 0.5547 | 保留 |
| 2026-10-05 | top-5 → top-4 | 0.8339 | 23.3% | 0.6535 | 0.6179 | 保留(用户拍板) |
| 2026-10-05 | 去精确码兜底 + 方案一补录席位 | — | — | — | — | 保留(subset16 A/B 不劣且省一路) |
| 2026-10-05 | 检索域软收窄 + 负证据守门 prompt | 0.9446 | 7.82% | 0.6956 | 0.6020 | 保留(生成侧达标;检索侧差 0.10/0.20,详见 20261005_source_scope_gate.md) |
| 2026-10-05 | SOURCE_SCOPE_BOOST 1.35→1.6 | 0.9334 | 9.29% | 0.6772 | 0.5862 | 回退(四项均不及 boost=1.35,判为噪声内无改善) |
| 2026-10-05 | RERANK_ROW_CONTEXT=1(行上下文) | 0.9457 | 7.30% | 0.6716 | 0.5994 | 回退(recall 反降,恢复 boost=1.35 + ROW_CONTEXT=0 即轮次1配置) |
| 2026-10-05 | 邻块扩展(top-2 带后接块) | 0.9079 | 14.77% | 0.6931 | 0.5983 | 回退(幻觉率恶化超标;额外文本诱发越界混写,flag 置 0 代码保留) |
| 2026-10-05 | 查询改写多路召回 | 0.9556* | 4.44%* | 0.7712* | 0.6328* | 回退(同口径 judge 下四项均不及轮次1且在噪声带内;flag 置 0 代码保留。* 为 coding 裁判口径,与上一行不可比,详见 20261005_query_rewrite.md) |
| 2026-10-05 | reranker v2-m3→v2-gemma | 0.9473* | 6.94%* | **0.7986*** | **0.6575*** | 待定(recall/precision 首次同向越噪声带上行,生成侧仍达标;待用户拍板,详见 20261005_reranker_gemma.md) |
| 2026-10-06 | τ/CLIFF/FIXED_K 重扫 + 池 48→96→192 扫描 | — | — | — | — | 保留现状(τ 为死杠杆、FIXED_K=4 已最优、池 96 是拐点;192 无增益弃;详见 20261006_tau_pool_scan.md,确认评测待跑) |
| 2026-10-06 | 双路重排 + 术语表扩容(+29条) + 删 LLM 查询改写 + 守门微调(推导豁免/查表前置) | 0.9476 | 6.29% | 0.7728 | 0.6659 | 回退双路重排(延迟×2 无收益,recall -0.026);其余保留(详见 20261006_round8_result.md,待用户拍板) |
| 2026-10-07 | 轮次9 badcase 三修(双路重排回退确认 / 检索冲突反问 / 翻译双路) | — | — | — | — | 反问机制上线(产品侧验证);翻译双路保留(详见 20261007_round9_badcase.md) |
| 2026-10-07 | 轮次10 20u200(并发引入:auth 限流 + 429 退避) | — | — | — | — | 并发协议定型 20u;详见 20261007_round10_20u200.md |
| 2026-10-08 | 轮次11-12:新架构接入 + 反问重构(对象化选项,scope 退役) | — | — | — | — | 反问题 recall 0.105→0.589 追平直查;详见 20261008_round11_newarch_dev200.md / 20261008_round12_clarify_subset.md |
| 2026-10-09 | 轮次13:qdrant 上 GPU + 翻译路关思考换 coding plan 通道 | 0.9364† | 10.56%† | 0.7915† | 0.6831† | 保留(检索不可用硬答 32→9;† 为 round13d 20u200 口径;详见 20261009_round13_20u200_link_fixes.md) |
| 2026-10-09 | MCP 隧道池(K 隧道分摊 transport 流控窗口) | — | — | — | — | 保留(治单隧道队头阻塞;详见 20261009_mcp_tunnel_pool.md) |
| 2026-10-09 | 轮次14-17 并发治理四连(qdrant 上 GPU / LLM 三状态熔断 / gate10 / 会话级限流槽)+ 600s 挂死根治(Next.js rewrites→nginx 反代 + SSE 心跳 + host 网络) | 0.9375† | 7.14%† | 0.7068† | 0.5456† | 保留全部(27 题挂死→0;† 为 20u×200 口径,与 20u×10 行不可比;详见 20261013_backfill_rounds14_17_concurrency.md) |
| 2026-10-12 | 重排吞吐治理:长度分桶批处理 + vLLM score 服务证伪回退 + 混合双路重排定型(4B+m3,W1=0.6) | 0.9393 | 8.74% | 0.7549 | 0.6714 | 长度分桶/混合双路保留;vLLM 回退(短块可修长块错乱,"未找到"51.5%);幻觉率历史最好(详见 20261012_rerank_throughput_vllm_falsified.md) |
| 2026-10-13 | 级联重排:m3 粗筛全量召回池(153-192)→24,混合双路精排;跳过 ratio 压池 | 0.9241 | 11.48% | **0.7907** | 0.6776 | 保留(recall +3.6pp 历史最高,faith -1.5pp/幻觉 +2.7pp 为代价;200 题 3h→15min 0 挂死;含 26 题三关漏损定位,详见 20261013_cascade_rerank.md) |
| 2026-10-13 | cascade_v2:复合题子检索(整句主路+子问题逐路去重追加)+ 重排前 content 哈希去重 + KEEP 24→32 | 0.9587 | 6.04% | **0.8195** | 0.6690 | 保留全部(recall/faith/幻觉率三项历史最好,级联双代价完全反转;含 NO_MATCH 五题分块排查定论:仅 dev_018 类表格推理链属索引缺口,详见 20261013_cascade_v2_subquery_dedup_keep32.md) |

## 当前生产配置(2026-10-13 定格)

- 检索:RETRIEVAL_SOURCE_SCOPE=1 + SOURCE_SCOPE_BOOST=1.35,ROW_CONTEXT=0,
  NEIGHBOR_EXPAND=0,FIXED_K=4,RESCUE_SEAT=1,EXACT_CODE=0,
  RERANK_RECALL_K=96,RERANK_BATCH_SIZE=64(长度分桶),
  混合双路重排 RERANK_MODEL=Qwen/Qwen3-Reranker-4B + RERANK2_MODEL=BAAI/bge-reranker-v2-m3(W1=0.6),
  **级联 RERANK_CASCADE=1(KEEP=32,粗筛吃全量召回池、跳过 ratio 压池,重排前 content 哈希去重)**;
  **复合题子检索 RETRIEVAL_SUBQUERY_SPLIT=1(整句主路+子问题逐路检索去重追加,cap 8)**;
  翻译双路 QUERY_TRANSLATE=1(coding plan 接入点,关思考)
- LLM:火山 coding plan(主 deepseek-v4-flash-ga-260731 / 副 doubao-seed-2-0-lite);
  LLM 三状态熔断 + 会话级限流槽(TTL120)
- 链路:VM semi-nginx(:3000)反代 /api→backend:8001(/→semi-web:3005);
  SSE 心跳 5s;backend host 网络;评测口径 --base 192.168.88.138:3000 且
  NO_PROXY 必带(本机系统代理劫持 LAN 请求)
- judge:火山 coding plan doubao-seed-2-0-lite-260428
- 检索服务:AutoDL 4080S 32G,VM→GPU 隧道池(8002-8011 多端口 watchdog);
  qdrant 副本在 GPU 本机(重入库后须重做快照同步)

## 检索侧 plateau 结论

调参轮(boost/行上下文/邻块/查询改写/τ/CLIFF/FIXED_K/池扩容)全部落在
judge 噪声带(±0.02)内、恶化或无增益。轮次6 换 gemma reranker 后
recall 0.7986 / precision 0.6575(coding 裁判口径),recall 距 0.80 目标仅 0.002,
已在噪声带边缘;**precision 0.80 目标与"每次给 3-4 块上下文"结构性冲突**
(4 块中相关块通常 1-2 个,数学上限 ~0.5-0.6),需重新校准口径。

已证伪/已完成,勿再试:
- ~~查询改写多路召回~~(轮次5 无效回退)
- ~~语义分块~~(金块审计无截断证据,见 20261005_chunk_audit.md)
- ~~更强 reranker~~(轮次6 已落地 gemma,金块在池时首现中位排名=2,不再是短板)
- ~~候选池扩容 96→192~~(轮次7 在池率仅 +1%,翻倍延迟白付;96 是拐点)
- ~~τ/CLIFF/FIXED_K 重扫~~(轮次7:τ 在 raw logits 口径下是死杠杆,FIXED_K=4 已最优)
- ~~重排输入语言(英文翻译 query / 按块语言分区打分)~~(2026-10-07 离线三臂
  均无增益,ZH/EN/MIX 各救 13/25,详见 20261007_inpool_rerank_test.md)

剩余候选(成本递增,先审计再动手):
1. ~~剩余低 recall 题再审计~~(已完成,见 20261006_lowrecall_reaudit.md:
   检索侧真实损失仅 12/185 题,43.5% 是生成/判分侧)
2. **查询英文翻译双路**(审计推荐,~1 天):中文 query 附加英文翻译路 RRF 合并,
   同时攻击池外 7 题 + top-4 边缘 5 题;预期 recall +0.03~0.05
3. 数据表/报警章节 chunk 特化(视翻译路效果):表格块文本稀疏 embedding 差,
   只对表格块做桥接,比全库方案便宜一个量级
4. 入库侧中文摘要桥接(全库重构):chunk 入库时拼中文摘要进 embedding 文本,
   需全库 40 万点重嵌入+重建 collection,仅在 2 证明语言不匹配是真桶后考虑

### 2026-10-05 subset16 异常处理子集(coding plan LLM + 新检索卡)

- before(无收窄/旧 prompt): good=5 partial=8 bad=3 → good+partial 81.25%
- after(收窄+新 prompt): good=5 partial=7 bad=4 → good+partial 75.00%
- 结论:16 题样本方差大,以 dev-200 ragas 全量为准(运行中);
  收窄实现已含 stem_registry.json 盘缓存修复(全库 40 万点扫过隧道会堵死首查)。

## 已知问题(暂不阻塞)

- tests/test_grounding.py 等 8 个单测失败:会话早前 grounding/nodes/llm 未完成改动
  的遗留,backend 运行时表现正常(200/200 评测 0 错误),待回补实现或修正测试。
