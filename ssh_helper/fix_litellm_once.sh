#!/bin/bash
# GPU 机上一性化修复:清重复 litellm -> 单实例规范启动 -> 带key健康验证
for pid in $(pgrep -f 'litellm --[c]onfig'); do kill "$pid" 2>/dev/null; done
sleep 2
cd /root/autodl-tmp/gw
set -a; source gpu.env; set +a
export VLLM_API_KEY=dummy
export VLLM_MAIN_BASE_URL=http://127.0.0.1:8000/v1
export VLLM_LIGHT_BASE_URL=http://127.0.0.1:8001/v1
nohup /root/envs-gateway/bin/python3.12 /root/envs-gateway/bin/litellm \
  --config litellm_config.yaml --port 4000 > /root/autodl-tmp/logs/litellm.log 2>&1 &
sleep 10
echo "procs=$(pgrep -cf 'litellm --[c]onfig')"
curl -s -m5 -H "Authorization: Bearer $LITELLM_MASTER_KEY" http://127.0.0.1:4000/v1/models | head -c 300
echo
curl -s -m2 -o /dev/null -w "health_noauth:%{http_code}\n" http://127.0.0.1:4000/health/liveliness
