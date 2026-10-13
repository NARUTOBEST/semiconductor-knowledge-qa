# 2026-10-06 轮次7: τ/CLIFF/FIXED_K 离线重扫 + 候选池 48→96→192 扫描(raw logits 口径)

## 动机

轮次6 切 gemma 后分数改为 raw logits(解决 sigmoid 饱和),生产 τ=0.3 变 stale;
金块审计指向排序主桶。用户指令"先做1和2":
① 重扫 RERANK_TAU / RERANK_CLIFF_RATIO / RERANK_FIXED_K;② 候选池 48→96 实证。

## 方法

- 基建:`engine_api.py search_text` 加 `with_ranked`(重排截断前全量候选分数,
  `rank_k = len(filtered)` 绕开 MAX_K×2 截断);`service.py` 透传。
- 采集:`collect_ranked.py` 对 dev200 全部 200 题 POST /search_text
  (k=8, score_ratio=0.01, with_ranked=true),落盘每题完整 gemma 分列表。
- gold 定义(与 chunk_audit 一致):金块文档内锚点(数字+英文术语)命中率 ≥0.5 的 chunk。
- 扫描:`scan_tau.py` 模拟生产选择链(score≥τ 且 score≥cliff×s1 → 截 fixed_k),
  网格 τ∈{None,-3..6} × cliff∈{0,0.4,0.5,0.6,0.7} × fixed_k∈{3,4,5,6}。

## 结果

### ① τ/CLIFF/FIXED_K 扫描(199 可判题,锚点金块严格口径)

- **τ 是死杠杆**:池内候选分数普遍偏高,top-4 均在 +5~+11;τ≤5 完全不起作用,
  τ=6 才开始截断且仅噪声级(F1 0.264 vs 基线 0.246,recall -0.015)。
  raw 口径下 τ=0.3 在 FIXED_K=4 绑定时完全惰性,无需回改。
- cliff=0.7:precision proxy +0.023 / recall -0.015,噪声内,不值得加旋钮。
- FIXED_K=4 已是最优(k=3 recall -0.05,k=5 仅 +0.005)。
- **结论:①不改动,维持 FIXED_K=4、τ/cliff 不启用。**

### ② 候选池扫描(真正的发现)

| 池大小 | 金块在池内 | 金块在池外但文档在 | 文档缺 | 金块进 top-4 |
|---|---|---|---|---|
| 96 | 117 (58.8%) | **79 (39.7%)** | 3 | 81 (40.5%) |
| 192 | 118 (60.5%) | 74 (37.9%) | 2 | 81 (41.5%) |

- 96→192 几乎不动(+1% 在池率,top-4 命中 +1%),翻倍重排延迟白付。
- **96 是拐点;主瓶颈不是池大小,是第一路向量召回对跨语言难例
  (中文 query ↔ 英文文档)的固有盲区**——金块 chunk 的向量相似度排不进全局前 192。
- 金块在池内时 gemma 排序很好(首现排名中位数 = 2)。
- **结论:②维持 96(已部署);池上限已打满,再往上要改召回侧结构:
  ③查询英文翻译双路 或 ④入库侧中文摘要桥接。**

## 生产状态(2026-10-06)

- GPU ret.env:RERANK_RECALL_K=96(本地已回退;GPU 侧一度切 192 做实验,
  **实验后因机器失联未能推回,下次开机需确认 ret.env 为 96 并重启检索**)。
- 配置不变项:FIXED_K=4、SOURCE_SCOPE_BOOST=1.35、ROW_CONTEXT=0、RERANK_MODEL=gemma。

## 待办

- #40 轮次7 确认评测(gemma + RECALL_K=96,dev-200 新目录 + coding 裁判同口径,
  对比轮次6 的 0.7986/0.6575)——因 GPU 机失联未跑。
- 产物:`runs/dev200_gemma_ranked.jsonl`(k96)、`runs/dev200_gemma_ranked_k192.jsonl`、
  `runs/scan_tau.json`、`runs/gold_cache.json`。

## 坑实录

1. collect_ranked 断点续跑把带 error 的记录当已完成(补采需删行重跑)。
2. nohup+& 经 ssh_run.py 非交互会话起服务会被会话回收杀掉;用
   /root/start_retrieval_only.sh(日志在 /root/autodl-tmp/logs/retrieval.log)。
3. GPU 机 SSH 突发失联(AutoDL 网关 38986 拒连 >5 分钟),期间隧道进程本地
   还活着但远端已断;VM:3000 前端不受影响(前端不直接依赖检索)。
4. 首查 500=qdrant 经隧道超时(冷启动),重试即恢复;k192 稳态 4.3s/题。
