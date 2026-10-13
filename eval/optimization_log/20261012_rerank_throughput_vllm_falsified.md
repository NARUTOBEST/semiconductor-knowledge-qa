# 20u 重排排队风暴治理:长度分桶 + vLLM score 证伪 + 回退进程内 + 混合重排定型

日期:2026-10-12 ~ 2026-10-13 | 协议:20u×10 dev200 + 直连压测(ssh_helper/http_burst.sh)

## 背景

round17 后并发稳态达成,20u 下延迟瓶颈收敛到重排:phase timing 实测 rerank 占
请求延迟 96%(78-145s),单查仅 1.09s——纯 GPU 算力排队(热态单查快、wall≈Σ
串行工作量)。MCP tools/call 47s vs 底层 search 2.6s 的排队风暴由此而来。

## 治理序列与结果

1. **无效杠杆(实测)**:扩副本 REPLICAS=2(同卡共享 FLOPs);K 96→48(池大小
   由 ratio 决定);批 8→64 单独上(padding 浪费抵消)。
2. **长度分桶批处理(embed.py _Qwen3Reranker.compute_score 按 token 数排序
   成批再各自 pad):wall 159→121s,零质量影响,保留。**
3. **vLLM score 服务(:8003, --runner pooling)证伪**:延迟达标(wall 159→49s)
   但打分质量废——官方 hf-overrides 配方修复合成短块(正例 0.96),真实长块仍
   错乱(同池金块进程内第 1(3.56)/vLLM 第 75(0.115));全量代价:"未找到"
   51.5%,faith 0.861/recall 0.381。已回退(RERANK_API_BASE 置空)。
   嫌疑:vLLM 0.29 pooling 对 Qwen3-Reranker 长输入池化/模板问题,待专项。
4. **混合双路重排定型(round18)**:主路 Qwen3-Reranker-4B + 副路
   bge-reranker-v2-m3,各路 min-max 归一 W1=0.6 融合;进程内 transformers +
   长度分桶。20u×10:faith 0.9393 / precision 0.6714 / recall 0.7549 /
   幻觉 8.74%(judged 199)——**幻觉率历史最好**。
   坑:RERANK2_API_BASE 未设时副路不参与(配置边角)。

## 配套修复(同期待遇)

- **srcs=0 双根因**:mcp_bridge 5s 读超时 + ald_image 缺失熔断;分片并行传
  GPU(13 倍提速),VM 容器名 semi-* 前缀,deploy 后须 restart。
- **术语表字符串替换(round19)负结果**:三 badcase 一致回退(金块多为中文块,
  bge-m3 双语对齐+翻译双路已覆盖,中英替换丢中文 sparse 词面)。词表资产保留,
  flag RETRIEVAL_GLOSSARY_REPLACE=0。round19_final 20u×10:0.9219 / 0.6841 /
  0.747 / 10.33%。

## 结论

进程内混合双路为生产基线;vLLM 路留代码不上线;延迟侧由级联重排接棒
(见 20261013_cascade_rerank.md)。
