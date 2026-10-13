# 2026-10-13 cascade_v2:复合题子检索 + 重排去重 + KEEP 32 — dev200 20u×10 全量复测

## 变更(三项,均当轮部署验证)

1. **复合题子检索**(`server/chat/service.py` `_speculative_search` + `_split_composite_query`,
   `RETRIEVAL_SUBQUERY_SPLIT=1`):整句主路不动,按 ？/;拆子问题(去序号前缀,标点强制)
   逐路检索,chunk_id 去重后追加到尾部(cap 8)。治多 fact 复合题金块互相挤占同一 top-4。
2. **重排前 content 哈希去重**(`mcp_servers/retrieval/engine_api.py` `_rerank`):入库重复块
   规范化文本去重保留首现。dev_184 实锤 top-6 两组一模一样,K=4 实际只有 ~3 席有效。
3. **RERANK_CASCADE_KEEP 24→32**(`ssh_helper/ret.env`,不入库):26 题三关定位中 GATE2
   边缘漏杀(dev_101 金块 m3 排 30/167 差 6 名被筛),32 与旧 ratio 池容量对齐。

## 结果(dev200 20u×10,198 判 16 免答,~13 分钟,0 挂死;1 题 RemoteDisconnected 单用户补答)

| 指标 | proc 基线 | cascade(上轮) | **cascade_v2(本轮)** |
|---|---|---|---|
| faithfulness | 0.9393 | 0.9241 | **0.9587** |
| context_precision | 0.6714 | 0.6776 | 0.6690 |
| context_recall | 0.7549 | 0.7907 | **0.8195**(历史最高) |
| answer_relevancy | — | — | 0.9768 |
| 幻觉率 | 8.74% | 11.48% | **6.04%**(历史最低) |

- 冒烟:dev_002 top-4 含 2 金块(#1 安装位置、#3 Barksdale)。
- 级联重排的 faithfulness/recall 双代价被本轮三项修复完全反转:recall 再 +2.9pp,
  faithfulness 反超 proc 基线 (+1.9pp),幻觉率减半。

## 分块质量排查(NO_MATCH 五题单 fact 探针,chunk_audit_nomatch.py)

- **dev_018 确认分块缺失**:单查也找不到含"12.5nm 短波截止"事实的完整块,该推理链
  只存于专利表格碎片——属索引资产问题,修分块(表格/PDF 解析)才有救,重排无责。
- **dev_176/dev_069**:单查能召回同主题块(SiN 热磷酸刻蚀、lockout/tagout 程序),
  主因是复合 query 污染+事实跨块割裂——子检索(本轮已上线)应能救,等复测分项验证。
- **dev_196/dev_070**:召回块为公式/术语块(LaTeX 归一化后 cov 天然失真),匹配误差
  可能性大,非检索失败。
- 结论:**无需重建分块**;唯一确定的索引缺口是 dev_018 类"表格中的推理链"。

## 运维注

- `run_concurrent.py --users 20` 会因 qa_sets 只有 10 个分片直接崩(main 加载 user11.json
  FileNotFoundError);实际口径是 **10 用户 × 20 题**,与基线一致。
- deploy_gpu.sh 会幂等拉起已证伪的 vLLM rerank(:8003),部署后手动 kill(白占 ~8G 显存)。
- GPU 实例今晨两次开机均遇欠费停机,充值后 SSH 端点不变(connect.westc.seetacloud.com:49157)。
