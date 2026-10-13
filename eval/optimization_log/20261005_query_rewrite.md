# 2026-10-05 轮次5:查询改写多路召回(recall 攻坚)

## 方法

- `server/chat/service.py`:`_speculative_search` 改走 `_search_with_rewrite`——
  首查 top1 < `QUERY_REWRITE_TRIGGER_SCORE`(0.5)或空结果且 `RETRIEVAL_QUERY_REWRITE=1`
  时,轻量 LLM(`MODEL_LIGHT`,temp 0,max_tokens 80)出 2 个改写查询(换手册术语视角:
  设备型号/英文术语/报警代码/部件名),逐个补检,`_merge_search_results` 按 chunk_id
  去重取分高合并,cap 至 RAGLITE_SEARCH_K(上下文规模不变)。
- 单测 `tests/test_query_rewrite.py` 6 项全过(高置信直通/低置信合并+cap/开关直通/空结果触发)。

## 结果(dev-200,judge 口径变更见下节)

judge 切火山 coding plan 后同口径对比(doubao-seed-2-0-lite-260428):

| 配置 | faith | 幻觉率 | recall | precision | 拒答剔除 |
|---|---|---|---|---|---|
| 轮次1 收窄(dev200_afterscope) | **0.9696** | **3.91%** | **0.7772** | **0.6340** | 10 |
| 轮次5 查询改写(dev200_qrewrite2) | 0.9556 | 4.44% | 0.7712 | 0.6328 | 15 |

backend 触发改写 13/200 次;触发题多为零召回题,补检后合并未把金块顶进 top-4。

## 结论:回退

- 四项指标全部落在 judge 噪声带(±0.02)内,recall 反而 -0.006;
- 每次触发多一次 LLM 调用,无对价收益;
- VM `.env` `RETRIEVAL_QUERY_REWRITE=1→0`,代码与单测保留(与邻块扩展同待遇)。
- 检索侧仍停在 plateau:recall ~0.78 / precision ~0.63,离 0.80 差 0.02 / 0.17。
  下一候选仍是 ②语义分块 ③更强 reranker(成本递增,需用户决策)。

## judge 口径变更(2026-10-05 起)

- 旧裁判 `ep-20260917012501-44sxb`(按量 /api/v3)——ARK 按量账户欠费(AccountOverdueError)弃用;
- 新裁判:火山 coding plan `doubao-seed-2-0-lite-260428`(/api/coding/v1,订阅制),
  `judge_50x5.py` 的 ARK_BASE/JUDGE_MODEL 支持环境变量 `ARK_JUDGE_BASE`/`ARK_JUDGE_MODEL` 覆盖;
- **与 ep 裁判的历史数值不可直接比**(同份 dev200_afterscope 结果:ep faith 0.9446/halluc 7.82%
  vs coding faith 0.9696/halluc 3.91%;coding 裁判 recall 普遍偏高、faith 更宽松);
- 已同口径重判:dev200_afterscope、dev200_baseline(2026-09-18 老基线,coding 口径
  faith 0.831/halluc 25.52%/recall 0.7453)、dev200_qrewrite2,基线文件在各自 `judge_coding/`。

## 附:今晚基础设施修复(非指标轮次)

1. 新 GPU 实例缺 HF 离线缓存(模型加载 LocalEntryNotFoundError→检索全挂):
   `start_gpu.sh` 软链段幂等补跑,`~/.cache/huggingface/hub_local/BAAI__* → /root/autodl-fs/models/*`。
2. backend 常驻 MCP 会话被 GPU 服务端 idle timeout 掐死后不重连(报 "SSE stream ended
   without a response" 不在 `_is_session_dead` 匹配内):`tools/mcp_bridge.py` 补
   `"sse stream ended"` 匹配 + 重建重试。
3. eval 断点续跑陷阱:`run_concurrent.py` 的 `run_user` 跳过旧输出已完成 id——
   复用旧 --out 目录时首轮失败记录会让后续"重跑"全部空转。重跑必须换新目录或先清空。
4. judge 依赖本地 `forward_6333.py`(块原文),进程被杀会导致幻觉率虚高(65%);
   重跑 judge 前先确认 6333 在监听。
