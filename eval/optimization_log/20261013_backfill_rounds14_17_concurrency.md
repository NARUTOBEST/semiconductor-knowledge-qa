# 轮次14-17:20 并发链路治理四连(LLM 齐发停滞 → 三状态熔断 → gate10 → 会话级限流)

日期:2026-10-09 ~ 2026-10-10 | 协议:20u×200(后两轮 20u×10),run_concurrent.py

## 背景

round13d 后检索侧已不再是瓶颈(qdrant 上 GPU + 翻译路修复),20 并发的故障面
转移到 LLM 侧与传输层:600s 静默挂死、auth 限流卡 40 分钟、判分小幅波动。

## 各轮改动与结果

### round14(qdrant 上 GPU 后首轮 20u200)
- 结论:**检索零瓶颈**(search 全程 <2.6s),瓶颈转移到 LLM 齐发停滞——600s
  超时 27 题集中首波;auth 限流改"只计失败次数"(成功请求不计数)。
- 判分(faith 0.9294 / recall 0.7044 / 幻觉 10.5%,judged 197)。
- 另:dev_001/dev_007 定为病态题(金块本身有争议,跨轮波动大)。

### round15(LLM 三状态熔断全链路)
- 按接入点(site)计数的三状态熔断:正常 → 观察期 → 放弃期;放弃期报
  "作答无效"而非硬答。**坑:tool_circuit.py 有 backend/GPU 两份,改动须同步。**

### round16(gate10 + 熔断)
- 20u200:0 熔断、0 闸超时,但 600s 静默挂死仍在(挂点在入口后第一段,
  TTL 回收锁兜底)。判分同题集小幅回退 2-3pp(噪声内)。
- 判分(faith 0.907 / recall 0.6631 / 幻觉 15.15%,judged 198,口径:refusal 未剔除,与上下行不可比)。

### round17(会话级限流 per-session 槽 + TTL120)
- **0 错误、无挂死,历史首次**。判分 faith 0.9375 / 幻觉 7.14%(20u200 口径)。
- 运维坑:deploy_vm 后须 `docker compose restart`(up -d 不重启已挂载容器)。

## 600s 挂死根治(独立专项,2026-10-11 ~ 10-12)

- **根因定案:`:3000` 是 semi-web(Next.js standalone),`/api/:path*` 经其
  rewrites 转发 127.0.0.1:8001——Next rewrites 代理对 SSE 长流高并发挂死**
  (keepalive 连接复用+缓冲)。对照实验:直连 :8001 0/200 挂死 vs 经 :3000 的
  27/200,单一变量坐实。
- 修法三层:① semi-nginx 容器(host 网络 :3000)反代 /api→8001
  (proxy_buffering off / read_timeout 3600s),/ → 3005;semi-web 移 3005;
  ② backend SSE 心跳(空闲 5s 下发 `: ping` 注释行,server/chat/router.py
  `_heartbeat_gen`);③ VM compose 固化 host 网络(docker-proxy 实验对照 0/200)。
- 评测口径:一律 `--base http://192.168.88.138:3000`(内网);**本机系统代理
  (127.0.0.1:7897)会劫持 LAN 评测请求,启动评测必带
  NO_PROXY=192.168.88.138,127.0.0.1,localhost**。

## 结论

保留全部。20 并发从"27 题挂死 + auth 卡 40 分钟"治理到 0 错误 0 挂死;
指标波动均在噪声带,质量侧无回退证据。
