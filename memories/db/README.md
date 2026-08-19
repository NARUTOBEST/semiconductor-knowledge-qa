# Agent 记忆系统数据库初始化

多库分表:三个独立 PostgreSQL 数据库,禁止跨库 join。

| 库 | 用途 | 建表方式 |
|---|---|---|
| `agent_working_db` | 工作记忆(LangGraph checkpoint) | `PostgresSaver.setup()` 自动生成 |
| `agent_short_db` | 短期会话流水 `session_events` | `01_short_session_events.sql` |
| `agent_long_db` | 长期记忆 `long_term_memories` + vector | `02_long_term_memories.sql` |

连接参数全部来自 `env/env.env`(`WORKING_PG_URI`/`SHORT_PG_URI`/`LONG_PG_URI`),禁止硬编码。

## 步骤 1:在 PostgreSQL 手动建库(仅首次)

```sql
CREATE DATABASE agent_working_db;
CREATE DATABASE agent_short_db;
CREATE DATABASE agent_long_db;
```

也可命令行:`createdb -h 127.0.0.1 -U postgres agent_working_db`(其余两个同理)。

## 步骤 2-5:一键初始化(幂等,可重复执行)

在**项目根目录**执行:

```bash
.venv_mineru/Scripts/python.exe memories/db/init_all.py
```

脚本依次:三个库 `CREATE EXTENSION vector` → working 库 `PostgresSaver.setup()` → short/long 库执行业务 DDL。
仅检查连接不建表:`python memories/db/init_all.py --check`。

## WSL2 端口转发风险(重要)

Python 业务代码在 Windows,PG/Redis 在 WSL2,通过 `127.0.0.1` 访问依赖 **netsh 端口转发**。
**WSL2 重启后端口转发规则会失效**,需在管理员 PowerShell 重建(按实际 WSL IP):

```powershell
netsh interface portproxy add v4tov4 listenport=5432 listenaddress=127.0.0.1 connectport=5432 connectaddress=<WSL_IP>
netsh interface portproxy add v4tov4 listenport=6379 listenaddress=127.0.0.1 connectport=6379 connectaddress=<WSL_IP>
```

查看 WSL IP:`wsl hostname -I`;查看已有规则:`netsh interface portproxy show all`。
**生产环境不要依赖此方案**,应使用独立数据库服务器或固定网络地址。

## 手工执行 SQL(可选,不用 init_all.py 时)

```bash
psql -h 127.0.0.1 -U postgres -d agent_working_db -f create_extensions.sql
psql -h 127.0.0.1 -U postgres -d agent_short_db   -f create_extensions.sql
psql -h 127.0.0.1 -U postgres -d agent_long_db    -f create_extensions.sql
psql -h 127.0.0.1 -U postgres -d agent_short_db   -f 01_short_session_events.sql
psql -h 127.0.0.1 -U postgres -d agent_long_db    -f 02_long_term_memories.sql
# working 库 checkpoint 表仍需由 PostgresSaver.setup() 创建(跑一次 init_all.py 或调 working_saver())
```
