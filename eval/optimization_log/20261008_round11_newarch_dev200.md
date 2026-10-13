# 轮次11:新架构部署 + dev200 单用户串行重测(反问开启口径)

日期:2026-10-08
数据:`runs/20261008_round11_single/`(200/200,0 error)、`runs/20261008_round11_clarifyfix/`(37 反问题修复后重答)、
轮次10 同口径重判:`runs/20261007_round10_rejudge_fixed/`
前置:`20261008_round10_20u200.md`

## 本轮配置(相对轮次10)

- **架构迁移**(按 DEPLOY.md):LLM 全走火山按量接入点(主 deepseek-v4-flash-压测 /
  副 doubao-seed-2.0-lite-压测),GPU 换 RTX 4090 24G 只跑检索;翻译路直连副模型接入点。
- **并行检索回退串行**(轮次11 回退,`tools/mcp_bridge.py` 删 `_multi_query`)。
- **冲突反问开启**(`RETRIEVAL_CONFLICT_CLARIFY=1`)+ run_http.py 评测器自动应答
  (选项按金文档 stem difflib 点选,等价知道自己在问什么的用户)。
- vLLM 重排器**验证失败回退进程内 FlagLLMReranker**(见下"结论4")。

## 判分器两处修复(影响口径,重要)

1. **facts 字符串撕碎**:judge gt 构造对 str 型 facts 逐字符迭代(gt 变成
   "砷；：；N；F；P；...")。已在 judge_50x5.py 归一化 `str→[str]`。
   轮次10 的已发布 recall 全部经此 bug(字符级"要点"被 LLM 宽松放行→虚高)。
2. **空上下文宽松失效模式**(复现 3× 稳定):dev_005 来源不含"砷"→judge 判
   recall 1.0;来源含"砷"但含金量不足→判 0.4。即 contexts 完全无关时 judge
   反而放行,部分相关时反而严格。**跨轮 recall 绝对值对比需谨慎**,本记录
   所有 recall 结论均基于同 judge 版本重判后的配对比较。

## 结果(全部用修复版 judge;轮次10 数字为重判值)

| 指标 | 轮次10(重判) | 轮次11原始(反问bug) | 轮次11拼接(修复反问) |
|---|---|---|---|
| 忠实度 | 0.9179 | 0.9230 | **0.9230** |
| 幻觉率 | 11.5%(23) | 9.5%(19) | **10.5%(21)** |
| context recall | 0.7312* | 0.6278 | 0.6380 |
| context precision | 0.5368 | 0.5176 | **0.5364** |
| answer relevancy | 0.9345 | 0.9008 | 0.9147 |

(*轮次10 recall 另受"空上下文宽松"影响,偏乐观)

### 非反问题 163 题配对(同 judge、干净口径)

- recall:**持平**(-0.0009)——串行回退无 recall 代价,与轮次10"并行不换 recall"结论一致
- precision:**+0.0803**(真实收益:并行检索的 precision 损伤如约恢复,印证轮次10预测)
- faithfulness:+0.0107;幻觉 17→15;relevancy +0.0184
- 延迟:单题 14~110s、多数 25~60s(单用户串行,含反问题)

### 反问题 37 题(本轮主战场)

- 轮次10(无反问,直查):recall 0.604
- 轮次11原始(反问 bug):recall 0.050 —— 34/37 归零
- 轮次11修复后:recall 0.105(仍远低于无反问路径)
- **bug 根因(已修)**:反问应答轮的投机检索拿**选项文本(=文档标题)**当 query,
  原问题信息需求被丢弃,命中全是标题/概览块。修复:
  `server/chat/service.py` `_SCOPE_PENDING` 存原问题,`_scope_resolve` 返回
  `(stem, 原问题)`,应答轮检索改用原问题 + scope 锁。
- **残余缺口(下轮主攻)**:修复后仍 0.105 vs 无反问 0.604。疑似 scope 锁定
  加权过强把候选挤向标题块 + FIXED_K=4 太窄。候选方向:应答轮检索绕过
  scope 强加权只留弱加权;或 scope 锁定时对原问题 query 走双路(原问题+选定文档内子查询)。

## 结论

1. **新架构(云端 LLM + 4090 检索机)可用**:非反问题全指标持平或更好,
   precision +0.08 是本轮最大真实收益;LLM 云端化对质量无可测影响。
2. **串行回退正确**:recall 无代价、precision 恢复。
3. **反问流程检索缺陷是当前 recall 唯一大头**:37 题把整体 recall 从 ~0.76
   拖到 0.64。反问功能本身要保留(用户明确要求),下轮专攻应答轮检索质量。
4. **vLLM 0.29 无法承载 bge-reranker-v2-gemma**(CausalLM 无打分头:/score 不注册,
   classify 硬转退化恒 0.5/0.5)→ 进程内 FlagLLMReranker 保留;仅 v2-m3 可走
   vLLM(需重标 τ,暂不动)。已固化 DEPLOY.md。
5. 单用户串行 + run_http(反问自动点选)是当前最稳的全量评测口径;200 题 ~2.5h。

## 事故与运维记录

1. **MSYS 路径改写**:直调 ssh_push 未设 `MSYS_NO_PATHCONV=1`,远端路径被改写为
   `C:/Git/...` 假路径。复合命令引号嵌套经 Git Bash→ssh 两层转义屡被吃(退出码 127
   /静默失败)→ 统一改"本地写脚本→ssh_push→执行"模式。
2. **VM→GPU 隧道认证**:新实例未收录 VM 公钥 → Permission denied。把 VM
   id_ed25519.pub 加入实例 authorized_keys 解决。
3. **backend_src/env/env.env 的 RETRIEVAL_CONFLICT_CLARIFY=1**:badcase 验证后未关回,
   批量评测首跑冒烟命中反问得空答案。本轮起反问常开 + run_http 自动应答,
   该开关不再需要来回切。
4. **dev_195 sources 事件丢失**(1/200,答案在但 0 来源):SSE 偶发,单独补跑解决。
5. **judge 判分前 Qdrant ping 守门**延续有效;本轮全程 forward_6333 未再出静默垃圾分。
6. 4090 实例用毕已关机(余额紧张,¥1.98/时)。
