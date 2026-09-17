# 离线 RAGAS 式评估(eval/)

**不进在线链路、不进镜像**(`.dockerignore` 已忽略 `eval/`)。仅在开发机/有服务时手动跑,
用来量化 RAG 质量与改动前后对比(如自适应检索/低置信重进是否救回 bad case、p50/p95 延迟变化)。

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

python -m eval.run_eval                # 全量 100 题 + 裁判
python -m eval.run_eval --no-judge     # 只跑硬指标,最省(不调裁判 LLM/不依赖 :8002 嵌入)
python -m eval.run_eval --limit 10     # 前 10 题(冒烟)
python -m eval.run_eval --ids 1,27,60  # 指定 bad case
python -m eval.run_eval --tier react   # 只跑 react 题

python -m eval.make_groundtruth --limit 20   # 生成参考答案草稿供人工校对
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

## 50x5 并发回归测试(一键)

真实 HTTP 链路 50 用户 x 5 题(报警码类事实题,`qa_sets_50x5/`)→ RAGAS 式 LLM 裁判 →
确定性硬指标 → 与基线 `eval/regression_baseline.json` 对比,任一指标退化超过 ±0.05
退出码 1(可挂 CI 或改动前后各跑一次)。

```bash
eval\regression_50x5.bat                    # 全量(约 40 分钟,含裁判 LLM 调用)
eval\regression_50x5.bat --users 5          # 冒烟
eval\regression_50x5.bat --skip-run         # 只重判/重比对现有结果
eval\regression_50x5.bat --update-baseline --skip-run --out eval/results_50x5  # 刷新基线
```

流程 = `run_concurrent.py`(SSE 真链路)→ `judge_50x5.py`(faithfulness/
context_precision/context_recall/answer_relevancy,幻觉率=faithfulness<0.75 占比)
→ `concurrent_metrics.py`(文档级 P@5/R@5/F1@5/NDCG@5/MRR/tier 准确率)。
对比口径:judge 5 项 + run 3 项(完成率/来源命中率/错误率)+ hard 6 项,共 14 项。

基线(2026-09-17,方舟按量 ep-8vvns 主 / ep-44sxb 副,AutoDL 3080Ti 检索):
忠实度 0.872 / 检索精确率 0.538 / 召回率 0.882 / 切题 0.963 / 幻觉率 14.7% /
完成率 100% / 来源命中 95.2% / 错误率 3.6% / R@5 0.775 / NDCG@5 0.729 / tier 准确 96.4%。
