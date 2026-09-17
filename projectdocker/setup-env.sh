#!/bin/bash
# ============================================================
# 一键生成 .env —— 无需人工填写任何密钥/地址。
#   - 总是【删除已存在的 .env 并重新生成】,所有变量一次写全;
#   - JWT/Redis/Postgres 密码本地 openssl 随机生成;
#   - LLM 网关地址/key、检索服务地址/服务间 token、云端兜底均已内置
#     (与 GPU 机侧脚本同一串,自动对齐);
#   - 【清空旧数据】./data/redis(checkpoint+短期流水)与 ./data/pg(长期记忆),
#     保证用新密码全新初始化;./data/auth(账号)、./data/sessions(摘要)保留不动。
# 用法:
#   bash setup-env.sh          # 重建 .env(旧文件备份为 .env.bak)+ 清空旧数据
# 重跑会丢失既有会话记忆/偏好,首次部署或确认可弃时执行。
# ============================================================
set -e
cd "$(dirname "$0")"

# 密钥不入库:从本地 env/env.env 读取(该文件被 .gitignore 排除)
_SECRETS_FILE="../env/env.env"
if [ ! -f "$_SECRETS_FILE" ]; then
  echo "缺少 $_SECRETS_FILE(需含 OPENAI_API_KEY / RETRIEVAL_INTERNAL_TOKEN),请先准备后再执行"; exit 1
fi
source <(grep -E '^(OPENAI_API_KEY|RETRIEVAL_INTERNAL_TOKEN)=' "$_SECRETS_FILE" | tr -d '')
: "${OPENAI_API_KEY:?OPENAI_API_KEY 未设置}"
: "${RETRIEVAL_INTERNAL_TOKEN:?RETRIEVAL_INTERNAL_TOKEN 未设置}"

rand() { openssl rand -hex "$1" 2>/dev/null || head -c 64 /dev/urandom | od -An -tx1 | tr -d ' \n' | cut -c1-"$(( $1 * 2 ))"; }

if [ -f .env ]; then
  mv -f .env .env.bak
  echo "已将旧 .env 备份为 .env.bak,重新生成 ..."
fi

JWT_SECRET=$(rand 32)
REDIS_PASSWORD=$(rand 16)
POSTGRES_PASSWORD=$(rand 16)
ADMIN_BOOTSTRAP_TOKEN=$(rand 16)

# 清空旧数据(新密码全新初始化);账号/摘要保留
rm -rf ./data/redis ./data/pg
mkdir -p ./data/redis ./data/pg ./data/auth ./data/sessions

cat > .env <<EOF
# 自动生成于 $(date '+%F %T');请勿提交到版本库
JWT_SECRET=${JWT_SECRET}
JWT_EXPIRE_HOURS=24
ADMIN_BOOTSTRAP_TOKEN=${ADMIN_BOOTSTRAP_TOKEN}
REDIS_PASSWORD=${REDIS_PASSWORD}
POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
LONG_MEM_ENABLED=1

# LLM 接入(火山方舟按 Token 付费接入点,压测用;OpenAI 兼容,经 gateway 通道)
#   主模型: ep-20260917012203-8vvns (DeepSeek-V4-Flash正式版 260731)
#   副模型: ep-20260917012501-44sxb (Doubao-Seed-2.0-lite 260428)
#   容器内访问宿主机隧道/外网统一走 host.docker.internal(compose 已配 host-gateway)
LLM_GATEWAY_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
LLM_GATEWAY_API_KEY=${OPENAI_API_KEY}
GATEWAY_MODEL_MAIN=ep-20260917012203-8vvns
GATEWAY_MODEL_LIGHT=ep-20260917012501-44sxb
GROUNDING_MODEL=light

# 检索微服务(GPU 机,经 SSH 反向隧道到宿主机 0.0.0.0:8002)
RETRIEVAL_SERVICE_URL=http://host.docker.internal:8002
RETRIEVAL_INTERNAL_TOKEN=${RETRIEVAL_INTERNAL_TOKEN}

# 云端兜底(与 gateway 同 key 同接入点;业务侧直连兜底备用)
OPENAI_API_KEY=${OPENAI_API_KEY}
OPENAI_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
OPENAI_TEXT_MODEL=ep-20260917012203-8vvns
OPENAI_FALLBACK_MODEL=ep-20260917012501-44sxb

# 50 用户并发压测:全局槽位与排队超时(须 config.py 支持 env 覆盖)
RATE_LIMIT_PER_USER_CONCURRENT=1
RATE_LIMIT_GLOBAL_CONCURRENT=16
RATE_LIMIT_QUEUE_TIMEOUT=120

# 端口
WEB_PORT=3000
WEB_BIND=0.0.0.0
QDRANT_BIND=127.0.0.1
EOF

chmod 600 .env
echo "已生成全新 .env(权限 600,所有变量已写全)。下一步: docker compose up -d"
