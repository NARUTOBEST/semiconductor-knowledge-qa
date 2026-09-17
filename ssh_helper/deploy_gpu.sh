#!/bin/bash
# 本地代码 -> GPU 机 /root/autodl-tmp/app 增量同步,并重启检索服务 :8002。
# 用法(项目根执行):  bash ssh_helper/deploy_gpu.sh [restart|norestart]
# 依赖:ssh_helper/ssh_push.py(目录自动 tar.gz -> 远端解压)。
set -e
cd "$(dirname "$0")/.."
# Git Bash(MSYS)会把 "/root/..." 形式的参数改写成 "C:/Git/...",禁用之
export MSYS_NO_PATHCONV=1
export MSYS2_ARG_CONV_EXCL="*"
HOST=connect.westb.seetacloud.com
PORT=34553
USER=root
PASS="dRaJ25yjOKQc"
APP=/root/autodl-tmp/app
MODE=${1:-restart}

# 检索服务运行面:mcp_servers + RAG + config + env(其余 agent 侧代码不在 GPU 上)
for d in mcp_servers RAG config env; do
  echo "== push $d -> $APP/$d"
  python ssh_helper/ssh_push.py --host $HOST --port $PORT --user $USER \
    --password "$PASS" --local "$d" --remote "$APP/"
done
echo "== push ret.env"
python ssh_helper/ssh_push.py --host $HOST --port $PORT --user $USER \
  --password "$PASS" --local ssh_helper/ret.env --remote "$APP/ret.env"

if [ "$MODE" = "norestart" ]; then echo "== SYNC ONLY (no restart)"; exit 0; fi

echo "== restart retrieval :8002"
python ssh_helper/ssh_run.py --host $HOST --port $PORT --user $USER \
  --password "$PASS" --timeout 60 --cmd '
# 防自杀:启动命令文本里含明文 "mcp_servers.retrieval.service"(nohup 行),
# pkill -f 会连同自身 shell 一起杀掉(表现为无输出 exit 127/服务起不来)。
# 改为 pgrep 枚举后排除自身 $$ 与父进程再 kill。
for pid in $(pgrep -f "mcp_servers[.]retrieval[.]service"); do
  [ "$pid" != "$$" ] && [ "$pid" != "$PPID" ] && kill "$pid" 2>/dev/null
done
sleep 2
cd /root/autodl-tmp/app
# env 优先系统盘镜像副本(方案B),缺失回退数据盘
ENV_HOME=/root/envs-gateway
[ -x "$ENV_HOME/bin/python" ] || ENV_HOME=/root/autodl-tmp/envs/gateway
export PATH=/root/miniconda3/bin:$ENV_HOME/bin:$PATH
export LD_LIBRARY_PATH=$ENV_HOME/lib:${LD_LIBRARY_PATH:-}
set -a; source /root/autodl-tmp/app/ret.env; set +a
nohup $ENV_HOME/bin/python -m mcp_servers.retrieval.service \
  > /root/autodl-tmp/logs/retrieval.log 2>&1 &
sleep 6  # 等 nohup 子进程脱离会话,否则 ssh_run 退出会连带走掉(实测)
pgrep -af "mcp_servers[.]retrieval[.]service" >/dev/null && echo "retrieval UP" || echo "retrieval FAILED to start"'
echo "== DONE"
