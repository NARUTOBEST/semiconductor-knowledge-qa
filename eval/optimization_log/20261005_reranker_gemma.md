# 2026-10-05 轮次6: 更强 reranker(bge-reranker-v2-gemma)A/B

## 动机(来自金块审计 20261005_chunk_audit.md)

低召回 30 题卡点分布:ranking_fail 56.7%(内容在库、文档已检回、金块没进 top-4)/
doc_missed 30% / 其他 13%。主桶是排序问题,更强重排模型直接攻击该桶,无需重入库。

## 方法

- 模型:`BAAI/bge-reranker-v2-gemma`(gemma-2b 架构 decoder-only,18层/2048h,
  fp32 存储 ~10GB,bf16 加载 ~5GB)替换 `BAAI/bge-reranker-v2-m3`(568M 编码器)。
- 代码:`config/config.py` RERANK_MODEL 支持 `RERANK_MODEL` 环境变量覆盖(默认不变);
  `RAG/embed.py` Reranker 按模型名自动选择 `FlagLLMReranker`(gemma/minicpm 系,
  use_bf16)或 `FlagReranker`(xlm-roberta 系,use_fp16)。
- GPU 侧:`/root/autodl-fs/models/bge-reranker-v2-gemma` +
  `~/.cache/huggingface/hub_local/BAAI__bge-reranker-v2-gemma` 软链(与 bge-m3 同模式);
  `/root/autodl-tmp/app/ret.env` 加 `RERANK_MODEL=BAAI/bge-reranker-v2-gemma`
  (回退=删该行+重启检索)。
- 下载:hf-mirror + curl 断点续传 3 分片并行(hf_hub 客户端两次挂死,见下"坑")。

## 结果(dev-200,coding 裁判同口径)

| 配置 | faith | 幻觉率 | recall | precision | 拒答剔除 |
|---|---|---|---|---|---|
| 轮次1 收窄(v2-m3) | 0.9696 | 3.91% | 0.7772 | 0.6340 | 10 |
| 轮次6 gemma | 0.9473 | 6.94% | **0.7986** | **0.6575** | 15 |

- recall +0.021 / precision +0.024,双双越过噪声带(±0.02)向 0.80 目标推进;
- faith -0.022(仍 ≥0.90 达标)、幻觉率 3.91%→6.94%(仍 <10% 达标);
- 口径注意:本轮 8 题因 LLM 请求超时未作答(运行期 VM→ARK 网络抖动,见下节),
  判定池 173 vs 轮次1 约 190,数值有轻微池差。

## 结论:保留(待用户确认)

检索侧两项目标指标(recall/precision)首次同向显著改善,方向与金块审计
(ranking_fail 主桶)一致;生成侧两项虽回退但仍双双达标。

## 坑实录

1. GPU envs-gateway 的 pip/huggingface-cli 脚本 shebang 指向旧路径
   /root/autodl-tmp/envs/gateway(实例迁移过目录),必须用
   `/root/envs-gateway/bin/python -m ...`。
2. huggingface_hub snapshot_download 两次挂死:network_turbo 通道 0 字节;
   hf-mirror 通道 67MB 卡住——客户端无请求超时,挂了不会自己重试。
   解法:curl `--speed-limit 20000 --speed-time 30` 卡死自动断 + `until` 循环
   `-C -` 断点续传,3 分片并行。
3. Windows Git Bash 直连远端路径参数会被 MSYS 改写成 C:/Git/...(ssh_push
   /root/... → C:/Git/root/...),须 `export MSYS_NO_PATHCONV=1`。
4. curl `-C -` 重复续传会在已完成的分片尾部追加垃圾数据(safetensors 报
   "incomplete metadata, file not fully covered");按 header 计算的应有长度
   `truncate -s <len>` 截断即修复。
5. **VM 断网事故(影响本轮 eval 首跑全废)**:VM 的 systemd-resolved 上游
   DNS 被配成 233.5.5.5/144.144.144.144(疑为 223.5.5.5/114.114.114.114 错拼),
   域名解析全部超时 → backend 所有 LLM 调用报 "模型请求失败: Connection error.",
   200 题全部空答案、每题重试 ~6 分钟。修复:/etc/systemd/resolved.conf
   DNS=223.5.5.5 114.114.114.114 + restart systemd-resolved,容器经 docker
   内嵌 DNS 动态转发即时生效,无需重启 backend。
6. judge --meta 的键必须是 query_id(题目 json 里是 id 字段,需手动加映射);
   且 judge_refusal 在首判时就写入记录,改 meta 后须就地 patch jsonl 再重算汇总。
