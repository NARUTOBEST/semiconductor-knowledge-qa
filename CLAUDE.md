# 项目指令(C:\project3)

## Git 提交规范

- **每完成一处代码/文档改动,必须立即 `git add` + `git commit`**,不要攒批、
  不要留给用户手动提交。commit message 用中文一句话说清"改了什么/为什么",
  收尾时如仍有未提交改动,主动提交后再报告。

## 评测规范

- **dev ragas 评测自 2026-10-06 起一律单用户串行**,不用多用户并发(run_concurrent.py 的
  `--users N` 并发模式弃用):并发会触发后端 per-IP 429 限流与 nginx 偶发 5xx,
  auth 阶段动辄卡 40 分钟,且故障排查困难。单用户(一个账号顺序问完 200 题)
  稳定可复现,单题 ~50s、全程 ~3h 属预期(入口:run_single.py)。
  既有并发跑完的答案数据可沿用判分,不必重答。
- qa_sets 目录(dev200_baseline/qa_sets/user1..10.json)只是题目分片,
  单用户跑时全部合并为一个队列,不必按分片对应用户。
- 评测输出必须新目录(断点续跑按已答题去重,旧目录会把上次 error 记录当已完成)。
- 长任务用 PowerShell `Start-Process -WindowStyle Hidden` 分离进程 +
  `-RedirectStandardOutput/-RedirectStandardError` 落日志(harness 后台任务会被杀)。
- judge 判分与答题口径一致(coding plan 裁判,meta 键必须是 query_id)。
- **优化归档:每次优化被评测指标提升证实后,必须在
  `eval/optimization_log/` 写一篇优化方法文档**(YYYYMMDD_主题.md:改动点/
  根因/对照数据/运维注),并更新该目录 README.md 索引与生产配置定格;
  顺手更新 auto-memory。小修(bug fix 无全量指标)至少记入 memory,
  若对指标有可量化影响则同样补全量对照后再归档。

## AutoDL 实例规范

- **凡是任务用到了 AutoDL GPU 实例(按小时计费,¥1.58/时),每次使用完之后
  都必须记得关机**(或直接执行关机),严禁空转计费。任务收尾时明确提醒用户,
  或在确认无后续任务后直接关机。
