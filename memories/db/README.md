# Agent 三层记忆系统初始化(Redis + PostgreSQL)

三层记忆落在不同后端:

| 层 | 用途 | 后端 / 结构 | 初始化方式 |
|---|---|---|---|
| 工作记忆 | LangGraph checkpoint(会话级 state 快照、跨轮续跑) | Redis `checkpoint:*` / `checkpoint_write:*` | `RedisSaver.setup()` 自动建索引 |
| 短期记忆 | 会话事件流水(审计/回放/溯源) | Redis `mem:*`(运行期自动创建) | 无需建表 |
| 长期记忆 | 用户个人偏好(跨会话个性化) | **PostgreSQL + pgvector**,`user_profile` + `long_mem_00..N-1` 分片表 | `long_term.setup()` 幂等建表 |

- Redis(工作 + 短期)连接参数来自 env(`REDIS_HOST/PORT/PASSWORD/DB` 或 `REDIS_URL`)。
- 长期记忆连接参数来自 env(`LONG_PG_URI` 或 `POSTGRES_HOST/PORT/DB/USER/PASSWORD`),
  按 `blake2b(username) % LONG_MEM_SHARD_COUNT`(默认 16)分表,向量列 `vector(1024)`
  存 BGE-m3 dense(复用检索微服务 `/embed_text`)。**长期记忆是旁路增强**:PG 不可用时
  自动降级(不抽取、不注入),不影响聊天;`LONG_MEM_ENABLED=0` 可整体关闭。

## 本地开发(WSL2)

在 WSL2 里装并启动 Redis(任选其一):

```bash
# 方式 A:apt 装 redis-server
sudo apt update && sudo apt install -y redis-server
sudo service redis-server start

# 方式 B:直接用 docker 跑一个
docker run -d --name redis -p 6379:6379 redis:7-alpine
```

WSL2 默认开启 localhost 转发,Windows 侧业务代码经 `127.0.0.1:6379` 即可访问;
若访问不通(少数环境转发失效),再按 WSL IP(`wsl hostname -I`)做 netsh portproxy:

```powershell
netsh interface portproxy add v4tov4 listenport=6379 listenaddress=127.0.0.1 connectport=6379 connectaddress=<WSL_IP>
```

Windows 侧 env 设置 `REDIS_HOST=127.0.0.1 REDIS_PORT=6379`(有密码则 `REDIS_PASSWORD=...`)。

## Ubuntu Server / docker-compose

compose 内置 `redis` 服务,backend 经 `REDIS_HOST=redis` 访问,无需手工初始化。

## 一键初始化 / 健康检查(幂等)

在**项目根目录**:

```bash
python memories/db/init_all.py            # PING + 建工作记忆 checkpoint 索引
python memories/db/init_all.py --check    # 只探测连通性
```

短期流水键在首次写事件时由 `memories/storage/short/short_term.py` 自动创建;
工作记忆 checkpoint 索引也会在后端首次 `working_saver()` 时由 `setup()` 幂等创建,
init_all.py 主要用于部署时提前验证与建索引。
