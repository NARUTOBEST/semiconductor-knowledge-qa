# 离线 RAGAS 式评估(eval/)

**不进在线链路、不进镜像**(`.dockerignore` 已忽略 `eval/`)。仅在开发机/有服务时手动跑,
用来量化 RAG 质量与改动前后对比(如自适应检索/低置信重进是否救回 bad case、p50/p95 延迟变化)。

## 目录结构

```
eval/
├── pipeline/                         # 评测流水线脚本,按用途分 dev / heldout 两类
│   ├── dev/                          # 开发迭代用流水线(题集参与调参,指标只作回归对比)
│   │   └── pipeline_50x5/            # 50x5 并发回归流水线(SSE 真链路;50x5 为初创时口径的历史命名)
│   │       ├── run_concurrent.py     #   ① 跑测(SSE 真链路,断点续跑)
│   │       ├── judge_50x5.py         #   ② LLM 裁判(忠实度/精确率/召回/切题,RAGAS 口径)
│   │       ├── regression_50x5.py / .bat #   总指挥:串联①② + 基线对比
│   │       └── regression_baseline.json  #   回归基线(勿随运行覆盖)
│   └── heldout/                      # 独立留出集评测流水线(协议见 heldout/README.md)
├── tools/        # 通用调试探针(登录+SSE 冒烟、记忆链路测试),不属任何一次评测
└── runs/         # 每次评测一个文件夹,按 dev / heldout 分区存放
    ├── dev/      # 开发集产物(题集参与迭代调参,指标为开发集口径)
    │   ├── 20261002_dev200_baseline/ # dev-200 首轮基线报告(REPORT.md + metrics_report.json)
    │   └── history/  # 各题集历史指标自动记录(<题集名>_history.jsonl,每次重判追加一行)
    └── heldout/  # 独立留出集产物(出题来源与迭代调参隔离,配置冻结后一次定版)
```

同类测试共用 `eval/pipeline/` 下的一套流水线脚本(服务该类型测试的所有版本);每次评测的
产物/题集/专属脚本放 `runs/<dev|heldout>/<名称或时间戳>/`(regression_50x5 默认自动建
`runs/dev/50x5_<时间戳>/`)。**口径纪律**:dev/ 下的题集与产物参与了迭代调参,其指标只能
用于版本间回归对比,不得对外宣称为独立测试指标;对外报告泛化效果必须用 heldout/ 下
满足独立性协议(见 `pipeline/heldout/README.md`)的题集与产物。

## 指标

质量评判**严格按 RAGAS 定义**(LLM 裁判,语义判分):

- `faithfulness` 忠实度:答案陈述能被检索资料支撑的比例(防幻觉核心);
- `context_precision` 上下文精确率:检索块中真正有用的比例;
- `answer_relevancy` 答案切题:BGE-m3 余弦(问↔答)与 LLM 判定平均;
- `context_recall`(可选,需参考答案):参考答案要点能被资料支撑的比例。

派生口径:幻觉率 = faithfulness<0.75 的题占比(闲聊题/拒答题剔除分母,拒答题单列
`refusal_bucket`)。运行健康度(完成率/错误率/延迟/来源命中率)仅作参考,不算质量评判。
历史上曾用的确定性硬指标(facts 关键词命中 Rfact@k、文档级 P/R/F1/NDCG/MRR、tier
路由准确率)已于 2026-10-02 移除,不再作为评判标准。
- **bad case 自动标注**:检索低置信/检索无来源/拒答/不忠实/事实覆盖不足/路由错/运行异常等。

裁判 LLM 默认走 `MODEL_LIGHT`(网关 `light` 或云端 lite),可用 `EVAL_JUDGE_MODEL` 覆盖。

## 前置服务

- Next `:3000`(登录 + SSE 提问;run_concurrent 走真实 HTTP 链路)
- VM Qdrant `:6333`(judge 按 chunk_id 直查取全文;`ssh_helper/forward_6333.py` 转发,
  取不到全文裁判会退化到 160 字截断片段,faithfulness 假性崩塌)
- 方舟 ARK 裁判 LLM(`OPENAI_API_KEY`,本地 `env/env.env`;403 多为欠费/配额)

## 运行(项目根;先旁路 Clash 代理,否则 127.0.0.1 被劫持)

```bash
export NO_PROXY=127.0.0.1,localhost,.volces.com,192.168.88.138
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy   # Windows: set HTTP_PROXY=

# 一键回归(跑测 + RAGAS 裁判 + 基线对比,见下节)
eval\pipeline\dev\pipeline_50x5\regression_50x5.bat --users 5     # 冒烟

# dev-200 单独跑(生成并发题集 → 跑测 → 裁判 → 报告)
python -m eval.pipeline.dev.pipeline_50x5.make_qasets_dev200 --users 10
python -m eval.pipeline.dev.pipeline_50x5.run_concurrent --users 10 \
    --base http://192.168.88.138:3000 \
    --sets-dir eval/pipeline/dev/pipeline_50x5/runs/dev200_baseline/qa_sets \
    --out eval/pipeline/dev/pipeline_50x5/runs/dev200
python -m eval.pipeline.dev.pipeline_50x5.judge_50x5 --results-dir <run目录> --out-dir <run目录> \
    --meta eval/qa_sets/v2_20261001/dev_200.json            # 追加 --force=全量重判
python eval/pipeline/dev/pipeline_50x5/make_dev200_report.py  # 汇总报告 + 追加题集历史
```

**题集历史自动记录**:每次重判/出报告时,`make_dev200_report.py` 自动把本轮 judge 5 项
指标追加到 `eval/runs/dev/history/<题集名>_history.jsonl`(一行一轮,含时间/run/备注,
`--note` 传备注;指标与末条完全相同则自动去重跳过)。文件里含全部历史轮次,可直接看
指标随优化的变化曲线。

## 题库(仅三套)

- **dev-200**:`eval/qa_sets/v2_20261001/dev_200.json`(开发集,200 题,9 字段含
  gold_standard/facts/task_type/expected_rounds;judge 的 `--meta` 读它做拒答题单列。
  并发拆分用 `make_qasets_dev200.py` 生成)
- **heldout**:`eval/qa_sets/v2_20261001/heldout_200.json`(留出集,出题来源与迭代调参隔离,
  配置冻结后一次定版)
- **dev 备选池**:`qa_sets/supplement_pool_2000.json`(2000 题备选,80 文档 x 25 题)。
  **占用登记**:从池中抽题写入任何题集后,必须回写该题的 `used_by` 字段
  (如 `["dev_200#dev_013"]`);抽题前先过滤已带 `used_by` 的题,防止重复取用。
  2026-10-04 dev-200 换题已占用 36 题(P01-02 等,见 dev_200.json 的 replaced_from_pool)。

旧 100 题库(web/public/eval-qa.json)及其衍生题集已于 2026-10-02 移除,不得再引用。

## 对比改动前后

改动前后各跑一轮回归(`--skip-run` 可只重判)或 dev-200,对比 `judge_summary_50x5.json` /
dev-200 的 `REPORT.md`:预期 faithfulness/context_precision/recall 不降或升、
剩余真幻觉清单条目减少。

## 并发回归测试(一键,dev-200 题集)

真实 HTTP(SSE)链路 10 并发用户 x 20 题(dev-200 均衡拆分,`make_qasets_dev200.py` 生成)→
RAGAS 口径 LLM 裁判 → 与基线 `eval/pipeline/dev/pipeline_50x5/regression_baseline.json`
对比,任一质量指标退化超过 ±0.05 退出码 1(可挂 CI 或改动前后各跑一次)。

```bash
eval\pipeline\dev\pipeline_50x5\regression_50x5.bat                    # 全量(跑测+裁判)
eval\pipeline\dev\pipeline_50x5\regression_50x5.bat --users 5          # 冒烟
eval\pipeline\dev\pipeline_50x5\regression_50x5.bat --skip-run         # 只重判/重比对现有结果
eval\pipeline\dev\pipeline_50x5\regression_50x5.bat --update-baseline  # 刷新基线(写回 pipeline_50x5/)
```

流程 = `run_concurrent.py`(SSE 真链路,429/断流自动退避重试)→ `judge_50x5.py`(faithfulness/
context_precision/context_recall/answer_relevancy,幻觉率=faithfulness<0.75 占比)。
对比口径:judge 5 项 + run 3 项(完成率/来源命中率/错误率,运行健康度参考)。

基线(2026-10-02 最终固化,dev-200 口径修正后重判,方舟 ep-8vvns 主 / ep-44sxb 裁判,
vLLM Qwen3-32B/14B @ AutoDL,VM Docker 部署):
忠实度 0.8133 / 检索精确率 0.5297 / 召回率 0.6116 / 切题 0.94 / 幻觉率 25.5%(37 条)/
完成率 100% / 错误率 0% / 来源命中 77%(拒答题 9 条单列,不计入总口径)。
