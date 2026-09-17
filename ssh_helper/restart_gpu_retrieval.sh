#!/bin/bash
# GPU 机检索服务重启(bash 语义:source/set -a)
cd /root/autodl-tmp/app || exit 1
pkill -f "mcp_servers.retrieval.service"
sleep 3
mkdir -p logs
set -a
source ret.env
set +a
export ENV_HOME=/root/envs-gateway
nohup $ENV_HOME/bin/python -m mcp_servers.retrieval.service > logs/retrieval.log 2>&1 &
sleep 25
echo "== health:"
curl -s -m 5 127.0.0.1:8002/health
echo
echo "== log tail:"
tail -5 logs/retrieval.log
