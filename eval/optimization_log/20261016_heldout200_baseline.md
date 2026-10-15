# 20261016 heldout_200 独立测试集首次全量(10u×20,G23 双卡)

## 概述

heldout 泛化集成集首次全量评测:8 篇 dev 未见文档(SECS 500/EUV 光刻/化学光刻/
先进封装材料/光学光刻 Mack/印制电路/超洁净处理/微纳加工)×25 题,规格与 dev-200
同分布(easy 32%/medium 48%/hard 20%,信息查询 60%/计算 20%/编排 12%/异常 8%)。
题集 `eval/qa_sets/v2_20261001/heldout_200.json`(build_datasets_heldout.py 生成,
分布/去重/同源相似对三检 pass)。

## 结果(judged 200,refusal_excluded 16)

| 指标 | heldout_200 | dev200 v7(同日同栈) | 差 |
|---|---|---|---|
| faithfulness | 0.9455 | 0.9615 | -1.6pp |
| precision | 0.5789 | 0.5503 | +2.9pp |
| recall | 0.8259 | 0.8880 | **-6.2pp** |
| relevancy | 0.9764 | 0.9689 | +0.8pp |
| 幻觉率 | 7.07%(13 条) | 4.37% | +2.7pp |

- 拒答桶 16 题(faith 0.975,precision 0.37):未覆盖类题正常走"未找到"口径。
- **泛化解读**:未见文档族 recall 掉 6.2pp、幻觉升 2.7pp——检索泛化是主要短板
  (dev200 的词面/型号通道对陌生文档失效),precision 反而更高(陌生题更谨慎)。
  后续优化优先级:陌生文档族的检索泛化 > 通用指标。

## 运行过程

- 三轮收敛:pass1 200 条中 22 坏(云端 LLM 超时家族 10 条 wall 60-145s 带块空答;
  流断/无检索 12 条)→ pass2 剩 2 → pass3 0,全程无检索侧雪崩。
- GPU 双卡健康:rerank 排队 wait>1s 日志 max 2.3s(有界),双卡副本分流正常。

## 运维注(本次踩坑,重要)

1. **qdrant 二进制必须用 `/root/autodl-tmp/bin/qdrant`(1.19.0)**:`/root/qdrant`
   是 1.12.4,读不了现库的 `shard_key_mapping.json`(`[]` 数组格式)直接 panic;
   qdrant119bin/qd113/qd115 全因 GLIBC_2.38 起不来。
2. **检索服务启动必须 `set -a; . ./ret.env; set +a`**:v7 归档命令漏了这步——
   env/env.env 里 `QDRANT_URL` 是注释态,漏 source 会掉进 qdrant 本地文件模式
   (空 qdrant_db + 双进程抢锁 `Storage folder ... already accessed by another
   instance` → 全部 srcs=0)。已固化 GPU 侧 `/root/start_ret_dual.sh`。
3. 服务模型加载 ~40s(reranker 权重 398 分片),**起服后必须等"预热完成"日志
   再冒烟**,加载期查询会超时。
4. VM 主机密钥变更(paramiko AutoAddPolicy 直连即可;OpenSSH 需先
   `ssh-keygen -R 192.168.88.138`)。
5. meta 事件 `sources_count` 与 sources 事件不同口径(恒 0 也有引用),评测按
   sources 事件收集,勿以 meta 字段判坏题。
6. make_qasets_dev200.py 加 `--qa` 参数:heldout_200 等同规格题集可复用分片器。
