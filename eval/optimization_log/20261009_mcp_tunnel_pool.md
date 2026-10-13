# 多隧道 × 多 MCP 会话连接池(2026-10-09)

## 背景

轮次13 实测 20 并发首题齐发 11 题 SSE 超时。根因链:单 SSH 隧道 + 单 MCP 会话
是链路队头阻塞单点——20 并发的 96 对重排全挤一条 transport 的流控窗口。
熔断(防雪崩)与攒批/双副本(提吞吐)都不改变"单条隧道"这一瓶颈。

## 改动

1. **tools/mcp_bridge.py 池化重构**
   - `_Conn`:单连接机械(原 `_ServerBridge` 的 _run/_serve/_open_session/_call
     整体迁入),新增 `inflight` 在途计数(handler 路由依据)、`label=retrieval#N`。
     一条连接挂死只重建自己(`_reopen` 语义不变),不拖累池内其他成员。
   - `_ServerBridge` 变池化门面:cfg `"url"`(单连接=旧行为)或 `"urls"`(列表,
     每项一条独立隧道+会话);`_pick()` 取 inflight 最少的就绪连接,同分
     round-robin 轮转;全未就绪走第一个(保持"未连接"错误语义)。
   - 注册只发生一次(任一连接首次就绪;注册连接重连才补注册刷新 schema)。
   - `_servers_config` 默认按 `RETRIEVAL_SERVICE_URLS` 展开多地址。
2. **config/config.py** 新增 `RETRIEVAL_SERVICE_URLS`(逗号分隔;缺省回退单
   `RETRIEVAL_SERVICE_URL`)。
3. **ssh_helper/start_tunnel.sh** 多端口:`GPU_TUNNEL_PORTS="8002 8003 8004"`
   默认三条隧道(均转发 GPU 侧 8002),每端口独立 watchdog 子 shell——一条
   隧道断了只重建自己。回退单隧道:留 8002 一个端口。
4. **env/env.env** 加 `RETRIEVAL_SERVICE_URLS=http://127.0.0.1:8002,8003,8004`。
5. **tests/test_mcp_pool.py**(9 用例)+ test_mcp_bridge.py(8 用例)全过:
   _pick 路由(同分轮转/最少在途优先/死会话绕开/全未就绪回退)、配置展开、
   memory 传输双连接端到端(两会话各自服务 + 一条摘除另一条兜底)。

## 部署记录(2026-10-09)

- VM `/home/lly/start_tunnel.sh` 已换多端口版并重启 watchdog(1 父 + 3 端口
  子 shell,GPU 关机中按 5s 重试,日志 /tmp/tunnel_r14.log)。
- backend 已 restart(bind mount 生效),容器日志确认 3 条连接 retrieval#0/1/2
  各自独立重试。
- GPU 侧(攒批+双副本+熔断)待下次开机随 deploy_gpu.sh 下发;之后 20u200
  复测验证队尾延迟。**测完关机(CLAUDE.md 强制)**。

## 踩坑

- `pkill -f start_tunnel.sh` 会匹配 ssh_run 自身的 bash -c 命令行(命令里含
  明文脚本路径),把自己的会话杀了(退出 127)。用 bracket-trick
  `pkill -f "start_tunne[l]"` 且 pkill 与启动拆两条 ssh 命令。
- Windows 侧编辑的 .sh 带 CRLF,推到 VM 后 bash 报 `syntax error near 'do\r'`;
  VM 上 `sed -i "s/\r$//"` 修正(本地也已转 LF)。
- VM 上 docker 需 sudo(`echo $PASS | sudo -S docker ...`)。

## 预期收益(诚实口径)

吞吐仍受 GPU 总算力守恒约束(攒批+双副本 ~3×);本项把单隧道串行窗口换成
K=3 条并行 transport,主要消的是 **网络侧队头阻塞**(轮次13 中 8-24s 的
SSE 排队里非 GPU 计算部分),20 并发下队尾应显著低于 30s 工具超时。
