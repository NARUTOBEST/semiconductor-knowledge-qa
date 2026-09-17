#!/bin/bash
# 本地后端源码 -> VM(192.168.88.138)~/projectdocker/backend_src 同步,
# 并确保 docker-compose backend 挂载 ./backend_src:/app(bind mount 加速迭代,
# 免重建镜像;依赖已在镜像 site-packages)。最后重启 backend 容器。
# 用法(项目根执行): bash ssh_helper/deploy_vm.sh [restart|norestart]
set -e
cd "$(dirname "$0")/.."
# Git Bash(MSYS)会把 "/root/..." 形式的参数改写,禁用之
export MSYS_NO_PATHCONV=1
export MSYS2_ARG_CONV_EXCL="*"
HOST=192.168.88.138
PORT=22
USER=lly
PASS=123
MODE=${1:-restart}

# backend 运行面(与镜像 /app 一致的最小集;RAG 不需要,检索在 GPU 机)
for d in agent_reasoning config "context management" env mcp_servers memories server tools; do
  echo "== push $d"
  python ssh_helper/ssh_push.py --host $HOST --port $PORT --user $USER \
    --password "$PASS" --local "$d" --remote "projectdocker/backend_src/"
done

python ssh_helper/ssh_run.py --host $HOST --port $PORT --user $USER \
  --password "$PASS" --timeout 60 --cmd '
cd ~/projectdocker
mkdir -p backend_src/trace
# compose 已含 bind mount 则跳过
if ! grep -q "./backend_src:/app" docker-compose.yml; then
  sed -i "s#- ./data/auth:/app/data/auth#- ./backend_src:/app\n      - ./data/auth:/app/data/auth#" docker-compose.yml
  echo "== compose updated: backend bind mount ./backend_src:/app"
fi'

if [ "$MODE" = "norestart" ]; then echo "== SYNC ONLY (no restart)"; exit 0; fi

python ssh_helper/ssh_run.py --host $HOST --port $PORT --user $USER \
  --password "$PASS" --timeout 120 --cmd '
cd ~/projectdocker && docker compose up -d backend 2>&1 | tail -2
sleep 3 && docker ps --format "{{.Names}} {{.Status}}" | grep backend'
echo "== DONE"
