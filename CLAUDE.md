# 项目指令(C:\project3)

## Git 提交规范

- **每完成一处代码/文档改动,必须立即 `git add` + `git commit`**,不要攒批、
  不要留给用户手动提交。commit message 用中文一句话说清"改了什么/为什么",
  收尾时如仍有未提交改动,主动提交后再报告。

## 评测规范

- **dev 评测一律 20 用户 × 10 题并发**(run_concurrent.py `--users 20`,题目分片
  qa_sets/user1..20.json 每片 10 题;入口与断点续跑见 run_concurrent.py)。
  **严禁单用户串行口径**(run_single.py 弃用,仅作单题调试用途)。
  并发前置条件:后端 RETRIEVAL_SERVICE_URLS 必须为 20 URL(双卡各 10 端口,
  验证口径:`docker exec semi-backend env | grep RETRIEVAL_SERVICE_URLS | tr , '\n' | wc -l`,
  验容器运行时而非 env 文件——compose 真读的是 projectdocker/.env,改后须
  `docker compose up -d --force-recreate backend`)。评测输出必须新目录
  (断点续跑按已答题去重,旧目录会把上次 error 记录当已完成)。
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
- **任何测试(全量 dev200 评测、子集/冒烟/单题验证)一跑完,立即关机 AutoDL
  实例,不等后续指令**:测试与测试之间如需再开机,由下一次任务重新启动。
  长评测可在关机前确认输出目录已落盘、断点可续跑。
