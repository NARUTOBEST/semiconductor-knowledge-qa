# 2026-10-05 金块审计:低召回 30 题卡点定位(②语义分块 vs ③更强 reranker 决策依据)

数据:`eval/optimization_log/20261005_chunk_audit.json`(脚本 `chunk_audit.py`,锚点法 v3——
金块文档为英文手册、facts 为中文提炼句,跨语言用数字+英文术语做不变锚点)。
输入:dev200_afterscope(coding 裁判口径)judge_context_recall<0.5 的 30 题。

## 卡点分布

| verdict | n | 占比 | 含义 |
|---|---|---|---|
| ranking_fail | 17 | 56.7% | 金块文档已检回、内容确在库(best_ratio≥0.83 者占 13/17,≥0.9 者 9/17),但金块没进 top-4 |
| doc_missed | 9 | 30% | 金块文档整篇没被检回(其中 8 题文档确在库、锚点命中率 0.57~1.0;1 题 dev_070 疑未入库,待确认) |
| anchor_missing | 2 | 6.7% | 最优块锚点命中仅 ~0.41(弱锚点或事实跨文档) |
| in_topk | 2 | 6.7% | 金块进了 top-4 裁判仍低(判分严/块内粘连) |

## 解读

1. **主因是排序,不是分块**:56.7% 的题金块内容完好在库、文档也进了候选,
   只是 reranker 没把它顶进 top-4(RERANK_FIXED_K=4)。没有发现截断/切断证据
   (最优块长度分布 114~781 字,集中在 380~550,与 CHUNK_TARGET=500 吻合;
   v2 shingle 版曾误报 81% content_lost,系跨语言字面匹配假象,已弃用)。
2. 表格块占比不高(ranking_fail 里 4/17),单纯"表格特殊处理"收益有限。
3. doc_missed 9 题(30%)是召回侧问题,reranker 帮不上(它只能重排已检回的);
   这部分与跨语言(中文问英文文档)dense 检索弱有关。

## 结论:优先 ③ 更强 reranker

- ③ 直接攻击 56.7% 的 ranking_fail 主桶,且**无需重入库**——换模型+重测即得结论;
- ② 语义分块(重入库,影响面大)对主桶帮助证据不足,且审计未发现分块截断证据;
- 候选模型:bge-reranker-v2-gemma(9.2B,跨语言强,慢)或 bge-reranker-v2-m3
  (与现库同族但更强);QPS 换质量,先 subset A/B 再全量。
