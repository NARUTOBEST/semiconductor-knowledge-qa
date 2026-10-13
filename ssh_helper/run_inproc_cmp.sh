#!/bin/bash
# 进程内 Qwen3 重排 vs vLLM 对照实验(GPU):停 vLLM 腾显存 -> 同池打分 -> 对比金块排名
set -e
for pid in $(pgrep -f EngineCore); do kill -9 "$pid" 2>/dev/null || true; done
for pid in $(pgrep -f "vllm serve"); do kill -9 "$pid" 2>/dev/null || true; done
sleep 6
set -a; source /root/autodl-tmp/app/ret.env; set +a
ENV_HOME=/root/envs-gateway
export PATH=$ENV_HOME/bin:$PATH; export LD_LIBRARY_PATH=$ENV_HOME/lib:$LD_LIBRARY_PATH
cd /root/autodl-tmp/app
$ENV_HOME/bin/python /root/autodl-tmp/inproc_cmp.py 2>&1 | grep -vE "Loading weights|local_dir|it/s\]|torch_dtype|CUDACaching"
bash /root/autodl-tmp/gw/start_vllm_rerank.sh
