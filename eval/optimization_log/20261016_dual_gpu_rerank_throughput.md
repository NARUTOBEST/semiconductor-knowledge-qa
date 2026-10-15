# 20261016 方案3落地:G23 单实例双卡 rerank 吞吐(v7)

## 改动点

1. **G23 双卡实例**(vGPU-32GB×2,克隆自 D29,数据盘克隆 Succeed):检索服务
   每卡一进程——卡0 `CUDA_VISIBLE_DEVICES=0 RETRIEVAL_PORT=8002`,卡1
   `=1/=8003`,共享 GPU 本机 qdrant(127.0.0.1:6333)。
2. **start_tunnel.sh 分组远端端口**:GPU_TUNNEL_TARGETS 组末新增第 4 段远端端口
   (缺省 8002,向后兼容),20 条隧道拆两组:8002-8011→卡0:8002,
   8012-8021→卡1:8003。backend `RETRIEVAL_SERVICE_URLS`(20 URL)**零改动**。
3. **LLM_TIMEOUT 60→90**(VM .env,`docker compose up -d --force-recreate backend`
   生效):与 REACT_STREAM_DEADLINE_S=90 对齐,治云端 LLM 首 token >60s 被
   单发掐死的空答案。
4. **rerank 排队可观测**(RAG/embed.py,_BatchedReranker):follower 等待 >1s
   记日志(副本数+等待水位)。

## 根因(v6 事故复盘,D29 单卡 10u)

- **检索侧**:10u 到达率(~2 路/秒)>> 单副本排空率(1×~14s/批),_BatchedReranker
  的 `_pending` 无界队列滚雪球(GPU 日志:rerank avg 82s / max 1971s);VM 侧
  工具层超时把排队的全判死 → **171/200 无检索作答**(LLM 裸答)。GPU 白算
  2683 次搜索。**已证伪的方案**:扩 RERANK_REPLICAS=2(ret.env 注释:双副本
  同卡共享算力,总 FLOPs 不变)。
- **LLM 侧**:检索修复后仍有 23 条空答案(elapsed≈60s,检索块 8-15 在)——
  云端 LLM 10u 下 TTFT>60s,单发超时。提到 90s 后 23→5→0。

## 对照数据(judged 口径,10u×20 题全量 dev200)

| 指标 | v5(G51 单卡) | round21(历史最好) | **v7(G23 双卡)** |
|---|---|---|---|
| faithfulness | 0.9451 | 0.9712 | **0.9615** |
| precision | 0.5423 | ~0.54(结构代价-14pp) | **0.5503** |
| recall | 0.8787 | 0.8821 | **0.8880** |
| relevancy | 0.9512 | — | **0.9689** |
| 幻觉率 | 7.65% | 4.35% | **4.37%** |

v7 vs v5 全面提升(recall +0.9pp / faith +1.6pp / 幻觉 -3.3pp);
vs round21:recall +0.6pp,faith -1.0pp,幻觉率持平。判分目录
`runs/20261016_v7_10x20_judge`(refusal_excluded=16,judged 199)。

## 运维注

- 双卡服务启动(GPU):**必须先 `set -a; . ./ret.env; set +a`**(QDRANT_URL 与
  重排参数全在 ret.env,漏了会掉 qdrant 本地文件模式空库,20261016 heldout 轮
  踩坑),再 `CUDA_VISIBLE_DEVICES={0,1} RETRIEVAL_PORT={8002,8003}` 起服务;
  已固化 GPU 侧 `/root/start_ret_dual.sh`。qdrant 二进制用
  `/root/autodl-tmp/bin/qdrant`(1.19.0;/root/qdrant 是 1.12.4 读不了现库,
  其余 1.19 副本缺 GLIBC_2.38);等服务"预热完成"日志再冒烟(模型加载 ~40s)。
- ssh_push.py 直调必须带 `MSYS_NO_PATHCONV=1`,否则 /home/... 被改写成
  C:/Git/home/... 推到错误位置(本次踩坑)。
- 换实例流程:克隆等数据盘 Succeed → deploy_gpu.sh norestart(带
  GPU_SSH_ENV=新 env)→ qdrant → 双卡服务 → 隧道双组重启 → 冒烟 → 全量。
- **空答案签名判别**:elapsed≈60s 且带检索块=LLM 超时;srcs=0 且带答案=
  检索降级;elapsed=0=流中断(评测端会重试)。三类都不可入库判分。
- G23 已关机;D29 已关机。判分前 6333 转发必须存活。
