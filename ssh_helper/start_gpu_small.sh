#!/bin/bash
# 小卡(vGPU-32GB)版启动:跳过 vLLM main(coding plan 承担主模型),只起
#   vLLM light :8001(Qwen3-14B-FP8, util 0.65)+ LiteLLM :4000 + 检索 :8002。
# 幂等:已健康的服务自动跳过。用法: bash start_gpu_small.sh
set -u
GW=/root/envs-gateway/bin
ENV_HOME=/root/envs-gateway
M=/root/autodl-tmp/models
LOG=/root/autodl-tmp/logs
mkdir -p "$LOG"
export LD_LIBRARY_PATH=$ENV_HOME/lib:${LD_LIBRARY_PATH:-}
export PATH=$ENV_HOME/bin:$PATH
_gpu_env="$(dirname "$0")/gpu.env"
[ -f "$_gpu_env" ] && . "$_gpu_env"
: "${LITELLM_MASTER_KEY:?缺少 LITELLM_MASTER_KEY(gpu.env)}"
: "${CLOUD_API_KEY:?缺少 CLOUD_API_KEY(gpu.env)}"

# ---- 0. autodl-fs 权重软链(复刻 start_gpu.sh 的 link_src 逻辑) ----
FS=/root/autodl-fs/models
if [ -d "$FS" ]; then
  HC=/root/.cache/huggingface/hub_local
  mkdir -p "$HC" "$M"
  link_src() {
    local src; src=$(ls -d "$FS"/$1 2>/dev/null | head -1)
    if [ -n "$src" ]; then rm -rf "$2"; ln -sfn "$src" "$2"; echo "== link $2 -> $src"
    else echo "== WARN $FS 下未找到 $1"; fi
  }
  link_src "${LIGHT_GLOB:-*Qwen3-14B*FP8*}"   "$M/llm-light"
  link_src "*bge-m3"                     "$HC/BAAI__bge-m3"
  link_src "*bge-reranker-v2-m3"         "$HC/BAAI__bge-reranker-v2-m3"
  link_src "*bge-reranker-v2-gemma*"     "$HC/BAAI__bge-reranker-v2-gemma"
  link_src "*clip-ViT-B-32*"             "$HC/sentence-transformers__clip-ViT-B-32"
else
  echo "== WARN $FS not mounted"
fi
[ -e "$M/llm-light" ] || { echo "== FATAL 轻模型权重缺失($M/llm-light),退出"; exit 1; }

# ---- 1. vLLM light :8001(小卡布局:util 0.65 ≈ 20.8G,留 ~8G 给检索模型) ----
if curl -s -m2 http://127.0.0.1:8001/health >/dev/null 2>&1; then
  echo "vllm-light already up"
else
  CUDA_VISIBLE_DEVICES=0 nohup $GW/python -m vllm.entrypoints.openai.api_server \
    --model $M/llm-light --served-model-name vllm-light \
    --max-model-len 8192 --max-num-seqs 64 --gpu-memory-utilization ${LIGHT_GPU_FRAC:-0.65} \
    --reasoning-parser qwen3 \
    --trust-remote-code --port 8001 > $LOG/vllm_light.log 2>&1 &
  echo "vllm-light starting pid=$!"
  for i in $(seq 1 60); do
    curl -s -m2 http://127.0.0.1:8001/health >/dev/null 2>&1 && { echo "vllm-light healthy"; break; }
    sleep 5
  done
  curl -s -m2 http://127.0.0.1:8001/health >/dev/null 2>&1 || { echo "VLLM_LIGHT_FAIL"; tail -20 $LOG/vllm_light.log; }
fi

# ---- 2. LiteLLM :4000(幂等修 shebang;light -> 8001) ----
if head -1 "$GW/litellm" 2>/dev/null | grep -q "autodl-tmp/envs"; then
  sed -i "1s|.*|#!$ENV_HOME/bin/python3.12|" "$GW/litellm" "$GW/litellm-proxy" 2>/dev/null
  echo "== fixed litellm shebang"
fi
if curl -s -m2 http://127.0.0.1:4000/health/liveliness >/dev/null 2>&1; then
  echo "litellm already up"
else
  pkill -f "litellm --[c]onfig" 2>/dev/null; sleep 2
  cd /root/autodl-tmp/gw
  export LITELLM_MASTER_KEY VLLM_API_KEY=dummy
  export VLLM_MAIN_BASE_URL=http://127.0.0.1:8000/v1
  export VLLM_LIGHT_BASE_URL=http://127.0.0.1:8001/v1
  export CLOUD_BASE_URL=https://ark.cn-beijing.volces.com/api/coding/v1
  export CLOUD_API_KEY
  nohup $GW/litellm --config litellm_config.yaml --port 4000 > $LOG/litellm.log 2>&1 &
  echo "litellm starting pid=$!"
  sleep 8
fi

# ---- 3. 检索 :8002(qdrant 经 VM 隧道 -R 6333) ----
if curl -s -m2 http://127.0.0.1:8002/health >/dev/null 2>&1; then
  echo "retrieval already up"
else
  for pid in $(pgrep -f "mcp_servers[.]retrieval[.]service"); do
    [ "$pid" != "$$" ] && [ "$pid" != "$PPID" ] && kill "$pid" 2>/dev/null
  done
  sleep 2
  cd /root/autodl-tmp/app
  set -a; source /root/autodl-tmp/app/ret.env; set +a
  CUDA_VISIBLE_DEVICES=0 nohup $GW/python -m mcp_servers.retrieval.service > $LOG/retrieval.log 2>&1 &
  echo "retrieval starting pid=$!"
fi
echo "== START_SMALL ISSUED =="
