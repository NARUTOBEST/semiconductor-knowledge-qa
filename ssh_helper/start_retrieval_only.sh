#!/bin/bash
# 12G 显存主机用:只启动检索微服务(云 API 做对话,无需 vllm/litellm)
pkill -f "vllm en[g]ine" 2>/dev/null
pkill -f "from vllm" 2>/dev/null
pkill -f "litellm" 2>/dev/null
for pid in $(pgrep -f "mcp_servers[.]retrieval[.]service"); do
  [ "$pid" != "$$" ] && [ "$pid" != "$PPID" ] && kill "$pid" 2>/dev/null
done
sleep 2
cd /root/autodl-tmp/app
ENV_HOME=/root/envs-gateway
[ -x "$ENV_HOME/bin/python" ] || ENV_HOME=/root/autodl-tmp/envs/gateway
export PATH=/root/miniconda3/bin:$ENV_HOME/bin:$PATH
export LD_LIBRARY_PATH=$ENV_HOME/lib:${LD_LIBRARY_PATH:-}
set -a; source /root/autodl-tmp/app/ret.env; set +a
nohup $ENV_HOME/bin/python -m mcp_servers.retrieval.service \
  > /root/autodl-tmp/logs/retrieval.log 2>&1 &
sleep 6
pgrep -af "mcp_servers[.]retrieval[.]service" >/dev/null && echo RETRIEVAL_UP \
  || { echo RETRIEVAL_FAIL; tail -10 /root/autodl-tmp/logs/retrieval.log; }
