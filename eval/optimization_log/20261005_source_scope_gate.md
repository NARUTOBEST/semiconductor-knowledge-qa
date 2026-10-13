# 2026-10-05 检索域软收窄 + 负证据守门 prompt + 火山 coding plan LLM

## 变更

1. **检索域软收窄**(`mcp_servers/retrieval/engine_api.py`):从 query 推断目标文档
   (标题词元打分,≥SOURCE_SCOPE_MIN_SCORE=4 且 ≥次高×1.5 才确信),重排后对目标文档
   块加权 SOURCE_SCOPE_BOOST=1.35;不确信则零操作。语料注册表
   `stem_registry.json`(22841 base stems)盘上缓存——40 万点全量扫描过隧道会
   堵死首查,必须走盘缓存。
2. **负证据守门 + 同源纪律 prompt**(`context management/system_prompt.py`):
   SYSTEM_PROMPT 增"负证据守门"(问数值而资料无该数值→强制拒答)与"同源纪律"
   (禁止跨文档挪用参数);SIMPLE/RAGLITE prompt 相应补忠实性条款。
3. **LLM 切换**:主=deepseek-v4-flash-ga-260731,副/轻=doubao-seed-2-0-lite-260428
   (火山 coding plan,ARK coding endpoint)。
4. **检索服务迁至新小显存卡**(vGPU-32GB ¥1.58/h),VM→GPU 隧道改用密钥认证
   (AutoDL sshd 拒绝无密钥密码转发的老链路)。

## 结果(dev-200,judge 同口径)

| 指标 | 上轮(方案A top-5) | 本轮 | 目标 | 判定 |
|---|---|---|---|---|
| faithfulness | 0.8494 | **0.9446** | ≥0.90 | ✅ 达标 |
| 幻觉率 | 21.55% | **7.82%** | <0.10 | ✅ 达标 |
| context_recall | 0.6719 | 0.6956 | ≥0.80 | ❌ |
| context_precision | 0.5547 | 0.6020 | ≥0.80 | ❌ |

n=189 计分,refusal 单列 10 题,0 请求错误,200/200 完成。运行目录
`eval/pipeline/dev/pipeline_50x5/runs/dev200_afterscope/`(judge 带
`--meta eval/qa_sets/v2_20261001/dev_200.json`,拒答题剔除口径与历史一致)。

subset16(异常处理 16 题)A/B:before good+partial 81.25% → after 75.00%,
16 题样本方差大,以 dev-200 全量为准。

## 结论

- 保留。生成侧两项指标一次达标,忠实度 +0.081、幻觉率 -12pp,为历史最大单项改善。
- 剩余缺口全在检索侧:recall +0.11、precision +0.22 才达标。
- 注意:faithfulness 提升部分来自新 LLM 更保守,与收窄/守门共同作用,未单独消融。

## 下一步候选

- precision:SCORE 阈值硬截断(低分块直接不返回)、SOURCE_SCOPE_BOOST 1.35→1.6
- recall:查询改写多路召回、glossary 扩展调参
- k 再降会伤 recall,不再动 k
