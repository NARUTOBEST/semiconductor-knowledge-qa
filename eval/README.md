# 离线 RAGAS 式评估(eval/)

**不进在线链路、不进镜像**(`.dockerignore` 已忽略 `eval/`)。仅在开发机/有服务时手动跑,
用来量化 RAG 质量与改动前后对比(如自适应检索/低置信重进是否救回 bad case、p50/p95 延迟变化)。

## 目录结构

```
eval/
├── pipeline/                         # 同类型测试的全套流水线脚本(可复用,服务所有版本)
│   ├── pipeline_full100/             # 全量 100 题白盒评估流水线(RAGAS 式)
│   │   └── run_eval.py / metrics.py / report.py / make_groundtruth.py / judge.py
│   └── pipeline_50x5/                # 50x5 并发回归流水线(SSE 真链路)
│       ├── run_concurrent.py         #   ① 跑测(SSE 真链路,断点续跑)
│       ├── judge_50x5.py             #   ② LLM 裁判(忠实度/幻觉率/召回/切题)
│       ├── concurrent_metrics.py     #   ③ 检索硬指标(P@5/R@5/NDCG@5/MRR)
│       ├── regression_50x5.py / .bat #   总指挥:串联①②③ + 基线对比
│       └── regression_baseline.json  #   回归基线(勿随运行覆盖)
├── tools/        # 通用调试探针(登录+SSE 冒烟、记忆链路测试),不属任何一次评测
└── runs/         # 每次评测一个文件夹,放该次的题集/结果/日志/专用脚本
    ├── 2026-09-18_regress20x50/  # 现行题集 qa_sets_20x50/(100 题库均衡拆成 20 用户 x 50 题,每题恰好被 10 名用户作答)
    └── 50x5_20260918_190804/     # 最新一轮 20 并发 x 50 题公网压测产物(1000 条记录 + judge + metrics_report)
```

同类测试共用 `eval/pipeline/` 下的一套流水线脚本(服务该类型测试的所有版本);每次评测的
产物/题集/专属脚本放 `runs/<名称或时间戳>/`(regression_50x5 默认自动建
`runs/50x5_<时间戳>/`)。

## 指标

- **RAGAS 式(LLM 裁判 + BGE-m3,离线跑)**:
  - `faithfulness` 忠实度:答案陈述能被检索资料支撑的比例(防幻觉核心);
  - `context_precision` 上下文精确率:检索块中真正有用的比例;
  - `answer_relevancy` 答案切题:BGE-m3 余弦(问↔答)与 LLM 判定平均;
  - `context_recall`(可选,需参考答案):参考答案要点能被资料支撑的比例。
- **确定性硬指标(不调 LLM)**:facts 关键词组命中率、期望来源书命中、tier 路由准确率、
  延迟 avg/p50/p95、`final_reason` 分布、redo/升级率、平均检索次数。
- **bad case 自动标注**:检索低置信/检索无来源/拒答/不忠实/事实覆盖不足/路由错/运行异常等。

裁判 LLM 默认走 `MODEL_LIGHT`(网关 `light` 或云端 lite),可用 `EVAL_JUDGE_MODEL` 覆盖。

## 前置服务

白盒直调 `chat.service.react_stream`(不走 HTTP/JWT/限流),但检索与裁判仍依赖:

- 检索微服务 `:8002`(BGE-m3/rerank;answer_relevancy 的嵌入也用它)
- LLM 网关 `:4000` 或云端 ARK
- Qdrant `:6333`

主后端 `:8001`、Next `:3000` **不需要**。

## 运行(项目根;先旁路 Clash 代理,否则 127.0.0.1 被劫持)

```bash
export NO_PROXY=127.0.0.1,localhost,.volces.com,.hf-mirror.com
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy   # Windows: set HTTP_PROXY=

python -m eval.pipeline.pipeline_full100.run_eval                # 全量 100 题 + 裁判
python -m eval.pipeline.pipeline_full100.run_eval --no-judge     # 只跑硬指标,最省(不调裁判 LLM/不依赖 :8002 嵌入)
python -m eval.pipeline.pipeline_full100.run_eval --limit 10     # 前 10 题(冒烟)
python -m eval.pipeline.pipeline_full100.run_eval --ids 1,27,60  # 指定 bad case
python -m eval.pipeline.pipeline_full100.run_eval --tier react   # 只跑 react 题

python -m eval.pipeline.pipeline_full100.make_groundtruth --limit 20   # 生成参考答案草稿供人工校对
```

报告写到 `eval/results/`:`eval-detail-<ts>.json`(每题明细 + bad_tags)、
`eval-summary-<ts>.json`(汇总),并打印控制台汇总。

## 题库

`web/public/eval-qa.json`,字段 `{id, q, tier, facts: [[近义词组]...], source, answer?}`。
`facts` 是"事实组"列表,组内任一关键词命中即该事实点覆盖;`answer` 为可选参考答案,
填入后启用 context_recall(用 `make_groundtruth.py` 生成草稿、人工校对后抄回)。

## 对比改动前后

各跑一次 `run_eval`,比 `eval-summary-*.json`:预期自适应检索上线后
faithfulness/context_precision/通过率不降或升、此前检索未命中类 bad case 被救回一部分;
p50 基本不变(常见题 1 检索 + 1 答),仅低置信题 p95 略升且受 40s 预算封顶。

## 20x50 并发回归测试(一键)

真实 HTTP(SSE)链路 20 并发用户 x 50 题(100 题库均衡拆分,`runs/2026-09-18_regress20x50/qa_sets_20x50/`,
每题恰好被 10 名用户作答,共 1000 条记录)→ RAGAS 式 LLM 裁判 → 确定性硬指标 →
与基线 `eval/pipeline/pipeline_50x5/regression_baseline.json` 对比,任一指标退化超过 ±0.05
退出码 1(可挂 CI 或改动前后各跑一次)。

```bash
eval\pipeline\pipeline_50x5\regression_50x5.bat                    # 全量(跑测+裁判约 2.5 小时)
eval\pipeline\pipeline_50x5\regression_50x5.bat --users 5          # 冒烟
eval\pipeline\pipeline_50x5\regression_50x5.bat --skip-run         # 只重判/重比对现有结果
eval\pipeline\pipeline_50x5\regression_50x5.bat --update-baseline --skip-run # 刷新基线(写回 pipeline_50x5/)
```

流程 = `run_concurrent.py`(SSE 真链路,429/断流自动退避重试)→ `judge_50x5.py`(faithfulness/
context_precision/context_recall/answer_relevancy,幻觉率=faithfulness<0.75 占比)
→ `concurrent_metrics.py`(文档级 P@5/R@5/F1@5/NDCG@5/MRR/tier 准确率)。
对比口径:judge 5 项 + run 3 项(完成率/来源命中率/错误率)+ hard 6 项,共 14 项。

**最新一轮实测(2026-09-18 晚,20 并发 x 1000 题,公网 cloudflared 隧道 → VM Docker → AutoDL GPU 检索 + 方舟云端 LLM)**:

| 类别 | 指标 | 数值 |
|---|---|---|
| 运行 | 完成率 / 错误率 | 99.4% / 2.5% |
| 运行 | 延迟 p50 / p95 | 16.9s / 99.9s |
| 运行 | 来源命中率 | 95.5% |
| 检索 | MRR / P@5 / R@5 / NDCG@5 | 0.732 / 0.492 / 0.785 / 0.737 |
| 路由 | tier 路由准确率 | 97.5% |

(LLM 裁判 5 项质量指标因当晚方舟账户欠费未出,充值后 `--skip-run` 补判;晚高峰方舟限流下的
错误为 per-user 在途锁级联,已由客户端退避重试吸收,链路本身——隧道/检索/GPU/DB——零错误。)

基线(2026-09-17,方舟按量 ep-8vvns 主 / ep-44sxb 副,AutoDL 3080Ti 检索,50 用户 x 5 题口径):
忠实度 0.872 / 检索精确率 0.538 / 召回率 0.882 / 切题 0.963 / 幻觉率 14.7% /
完成率 100% / 来源命中 95.2% / 错误率 3.6% / R@5 0.775 / NDCG@5 0.729 / tier 准确 96.4%。
