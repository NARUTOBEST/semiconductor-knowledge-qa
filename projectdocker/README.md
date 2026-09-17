# 半导体设备知识问答系统 —— 项目机离线部署包(单机,零模型 + 远程 GPU 网关)

把业务系统打包成可离线运行的 Docker 部署。**目标 Ubuntu 机器无需联网、无需 GPU、无模型文件**——
镜像和知识库都在本包内。**检索(BGE-m3/CLIP/Reranker)与 LLM(vLLM×2+LiteLLM)全部部署在 GPU 机**
(权重由 `ossutil` 从阿里云 OSS 下载,部署脚本见项目 `ssh_helper/`),项目机通过 **SSH 反向隧道**
访问 GPU 机的 4000(LLM)与 8002(检索)。

## 包内容
| 路径 | 说明 |
|---|---|
| `images/pkg.tar` | 全部 5 个镜像(backend/web/qdrant/redis-stack/pgvector-postgres),**不含任何模型** |
| `snapshots/` | Qdrant 知识库快照(ald_text 40 万块 + ald_image 14.9 万块),约 3.4G |
| `docker-compose.yml` | 自包含编排(5 服务,只用镜像,不构建) |
| `.env.example` | 环境变量模板(说明用途;实际 .env 由脚本生成) |
| `setup-env.sh` | 一键生成 .env(密码随机、网关/令牌内置、清空旧数据) |
| `load-images.sh` | 导入镜像 |
| `restore-snapshots.sh` | 恢复知识库 |

## 前提
1. 目标机装好 **Docker Engine + docker compose 插件**,内存 ≥8G(本机不跑模型,占用小)。
2. 一台 **GPU 机**已部署 LLM 网关(:4000)与检索服务(:8002)(见项目 `ssh_helper/`),并能从
   本机 SSH 登录(用于反向隧道)。两端 API key/token 已内置对齐,无需手工配置。
3. 开放 **3000** 端口(唯一对用户对外)。Qdrant(6333)/backend(8001)只绑 127.0.0.1。

## 部署步骤
```bash
# 0) 把本目录整个拷到 Ubuntu 机,进入目录
cd projectdocker

# 1) 导入镜像(离线,不联网)
bash load-images.sh

# 2) 一键生成配置(无需手填任何东西):JWT/Redis/Postgres 密码本地 openssl 随机生成;
#    网关地址/门禁 key、检索服务地址/服务间 token、云端兜底均已内置。
#    脚本总是【删除旧 .env 并重建、写全所有变量】(旧文件备份为 .env.bak),
#    并【清空旧数据】./data/redis(checkpoint+短期流水)与 ./data/pg(长期记忆),
#    保证用新密码全新初始化;./data/auth(账号)、./data/sessions(摘要)保留不动。
#    重跑会丢失既有会话记忆/偏好,首次部署或确认可弃时执行。
bash setup-env.sh

# 3) 建立 SSH 隧道(单条连接双向转发,保活循环):
#    -L 4000/-L 8002:本机 127.0.0.1:4000/8002 → GPU 机 LiteLLM/检索服务
#    -R 6333:GPU 机 127.0.0.1:6333 → 本机 qdrant(检索服务回查知识库)
#    nohup bash -c 'while true; do ssh -N -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes \
#      -L 4000:127.0.0.1:4000 -L 8002:127.0.0.1:8002 -R 6333:127.0.0.1:6333 \
#      -p <GPU机SSH端口> root@<GPU机地址>; sleep 5; done' \
#      > /tmp/tunnel.log 2>&1 &

# 4) 起全套
docker compose up -d

# 5) 初始化长期记忆库(幂等;首次部署执行一次)
docker compose exec backend python -m memories.db.init_all

# 6) 恢复知识库(幂等,已存在自动跳过)
bash restore-snapshots.sh
```

## 验证
```bash
docker compose ps                 # 各服务 healthy / running
curl http://127.0.0.1:8001/health # {"api":"ok","qdrant":"ok","llm":"ok"}(隧道通后 llm=ok)
```
浏览器访问 `http://<本机IP>:3000`。注册第一个账号时,请求头带
`X-Bootstrap-Token: <ADMIN_BOOTSTRAP_TOKEN>`(见 .env)即成为管理员。

## 常见问题
- **`llm` 非 ok / 问答不回复**:反向隧道没通。确认本机 `curl 127.0.0.1:4000/health/lambd`
  有响应;隧道进程存活(`ps aux | grep 'ssh -N'`);GPU 机两服务 healthy。
- **检索报错**:同上,确认本机 `curl 127.0.0.1:8002/health` 返回 ok(经隧道)。
- **图片不显示**:知识库图片需 TOS 对象存储签名(`.env` 的 TOS 段);纯文本问答不受影响。
- **数据持久化**:qdrant 在 docker 命名卷 `qdrant_storage`;Redis 在 `./data/redis`(AOF);
  用户库在 `./data/auth`;会话摘要在 `./data/sessions`。快照只需首次恢复一次。
- **端口冲突**:改 `.env` 的 `WEB_PORT`。
