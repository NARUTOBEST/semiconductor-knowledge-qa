# 半导体设备知识库智能问答系统

> **关于仓库历史的说明**：本项目因个人原因疏忽，忘记在项目刚开始开发时建立仓库，也忘记在后续开发过程中于 GitHub 上 commit 每一次修改，因此提交历史集中且不能反映真实开发节奏。**但本项目以人格担保是从 0 到 1 独立开发的**，全部设计、编码、部署与评测均由本人完成。评测脚本与题集已随仓库开源，所有指标可复现（见[评测](#-评测体系与指标)）。

面向半导体产线工程师的领域知识问答助手：覆盖设备手册、SEMI 标准、行业研报、CAD 图纸/演示视频等约 **2.4 万份**异构资料（**55 万**知识块），解决工程师定位报警码与工艺参数耗时长、通用大模型编造参数导致现场误用的问题。答案带出处页码、逐句可核查。

## ✨ 核心特性

- **三级范式路由**：simple / raglite / react 按问题复杂度自动分流 + 质检门低置信自动升级重答，用户无感
- **防幻觉闸门**：判定"有支撑"必须给出可字符串验证的原文引文，裁判造假即失败；低置信整段不出后端，降级为翻阅指引
- **混合检索**：BGE-m3 稠密 + 稀疏 + RRF 融合，章节路径注入重排（重排池保底 32，经 1000 题 × 10 配置免 LLM 扫描调优），低召回自动改写重查；报警码精确召回兜底、稀疏保底救表格块
- **图文双模态**：2.1 万 CAD 零件/装配自动渲染入库，跨模态检索
- **记忆系统**：后台异步记忆图 + 升迁门，会话记忆不阻塞回答
- **高并发流式链路**：SSE 流式输出、看门狗断流恢复、空答案跨模型救援、双层限流
- **全链路故障韧性**：熔断降级 + 本地 WAL/spool 兜底，恢复后幂等回填；崩溃安全的上下文压缩（先落盘再删除，重跑幂等）

## 🏗️ 架构

```
浏览器 ── cloudflared 公网隧道 ──> VM (Docker Compose 五容器)
                                    ├── web        (Next.js :3000)
                                    ├── backend    (FastAPI :8001)
                                    ├── qdrant / redis / postgres
                                    └── SSH 隧道 ──> AutoDL GPU 服务器
                                                      ├── vLLM 推理 :8000/:8001（主模型 Qwen3-32B-FP8 + 副模型 Qwen3-14B-FP8，均稠密）──> LiteLLM 网关 :4000（main/light 别名，云端 API 仅兜底）
                                                      └── 检索微服务 :8002（BGE-m3 / Reranker / CLIP）
```

- 三机分工：LLM 与向量检索在 GPU，业务后端与存储在 VM，公网入口经隧道
- 检索微服务与业务后端解耦，通过 MCP 协议 + 内部令牌鉴权互访

## 📊 评测体系与指标

自建 RAGAS 口径评测线束：20 并发 × 50 题真实 HTTP（SSE）全量回归、题级断点续跑、GPU 本地 LLM-as-Judge（Qwen3-32B 裁判，零云成本、无限流）+ 确定性检索硬指标（MRR / NDCG@K / Rfact@K / P_ret / R_pool），与基线自动对比，可一键复跑：

```bash
eval\pipeline\pipeline_50x5\regression_50x5.bat          # 全量 20 用户 × 50 题（1000 条记录）
eval\pipeline\pipeline_50x5\fetch_pools.py               # 离线补查 1000 题重排前候选池（R_pool 分母）
eval\pipeline\pipeline_50x5\concurrent_metrics.py        # 检索硬指标计算（--pools 池内召回）
eval\pipeline\pipeline_50x5\judge_50x5.py                # GPU LLM-as-Judge（忠实度/幻觉率/上下文召回/切题/答案准确）
eval\pipeline\pipeline_50x5\sweep_retrieval.py           # 免 LLM 参数扫描（/debug_cfg 热调参，1000 题 × 10 配置）
eval\pipeline\pipeline_full100\run_eval.py               # 全量 100 题白盒评测
```

**检索调优方法**：直调检索服务 `/search_text`（不经 LLM）扫 RERANK_POOL_MIN{8,16,24,32,48} × score_ratio{0.4,0} × 截断模式（τ+断崖 / 固定 K），真值分母统一取 48 候选全集，选定 **RERANK_POOL_MIN=32**（MRR +2.5pt、NDCG@5 +4.6pt、R@10 +5.8pt，P_ret 仅 -1.7pt）；扫描还证明 τ/MIN_K/断崖/固定 K 变体收益均 <1pt，池深是唯一显著杠杆。

**最新一轮实测（2026-09-20，20 并发 × 50 题 = 1000 条记录，内网全链路：VM Docker → SSH 隧道 → GPU vLLM/检索微服务）**：

| 类别 | 指标 | 数值 |
|---|---|---|
| 运行 | 完成率 / 错误率 | 98.3% / 1.7% |
| 运行 | 延迟 p50 / p95 / 均值（20 并发） | 13.4s / 70.1s / 34.9s |
| 检索 | MRR / NDCG@3 / NDCG@5 | 0.7289 / 0.6973 / 0.7361 |
| 检索 | Rfact@5 / Rfact@10（要点级召回） | 0.8481 / 0.8481 |
| 检索 | P_ret（精确率，返回块数口径） | 0.6043 |
| 质量 | 忠实度 Faithfulness（LLM 裁判） | **0.9821** |
| 质量 | 幻觉率 | **2.56%**（21/821 例，逐题可复核） |
| 质量 | 上下文召回 Context Recall | **0.9562** |
| 质量 | 答案切题度 Answer Relevancy | **0.9028** |
| 质量 | 答案准确度 Answer Accuracy | 0.7789 |
| 路由 | tier 路由准确率 | 90.3% |

> 口径说明：**Rfact@K** = 金标要点的证据块进入返回前 K 名的要点占比（分母=要点数，跨版本可比，是召回能力的真实口径）；块级 R@K 的分母为金标文档在候选全集中的全部命中切块（均值 ~14 块/题），受返回容量（MAX_K=8）结构性封顶（上限 ≈ 8/14 ≈ 0.56），故不作为召回主指标；**P_ret** 为精确率豁免项（返回块数口径对动态截断敏感）。p95 尾部由 react 多步推理串行 LLM 调用在 20 并发下的排队等待主导。

完整产物见 [`eval/runs/50x5_20260920_poolmin32/`](eval/runs/50x5_20260920_poolmin32/)（REPORT.md 含 10 配置扫描全表与逐轮对比）。

## 🚀 快速开始

```bash
# 后端五容器（VM 上）
cd projectdocker && docker compose up -d

# GPU 检索微服务（AutoDL，BGE-m3/Reranker/CLIP）
bash ssh_helper/setup_gpu.sh       # 环境（vLLM + 检索依赖）
bash ssh_helper/start_gpu.sh       # 启动推理与检索服务
bash ssh_helper/deploy_gpu.sh      # 同步检索代码并重启

# 本地开发
start.bat
```

## 📁 目录结构

```
├── RAG/                  # 文档解析、分块、嵌入入库（PDF/PPT/Word/CAD 渲染）
├── agent_reasoning/      # 三级范式：simple / raglite（PE）/ react
├── mcp_servers/          # 检索微服务（MCP 协议，GPU 侧部署）
├── context management/   # 崩溃安全的上下文压缩
├── server/               # FastAPI 业务后端（SSE 流式、限流、鉴权）
├── web/                  # Next.js 前端
├── projectdocker/        # Docker Compose 五容器编排
├── eval/                 # 评测：pipeline/（可复用流水线）+ runs/（每次评测产物）
├── tests/                # 单元/集成测试
└── ssh_helper/           # GPU/VM 部署与隧道脚本
```

## 📄 License

仅供学习交流使用。
