#!/bin/bash
# GPU 服务器三服务启动(权重/依赖就绪后执行)。全部 nohup 后台 + 日志落盘。
# 【GPU 机以后只部署 LLM;检索全套已迁回项目机 docker-project(CPU/内存)】
# 环境优先系统盘镜像副本(/root/envs-gateway,方案B自定义镜像随系统盘保存),
# 不存在(老实例/未做镜像)回退数据盘原路径
ENV_HOME=/root/envs-gateway
[ -x "$ENV_HOME/bin/python" ] || ENV_HOME=/root/autodl-tmp/envs/gateway
export PATH=/root/miniconda3/bin:$ENV_HOME/bin:$PATH   # EngineCore 子进程要能找到 ninja 等
export LD_LIBRARY_PATH=$ENV_HOME/lib:${LD_LIBRARY_PATH:-}   # conda icu/libstdc++ 优先于系统旧库
GW=$ENV_HOME/bin
M=/root/autodl-tmp/models
LOG=/root/autodl-tmp/logs
mkdir -p $LOG

# ---- 0. 权重改走 autodl-fs 网盘(幂等 symlink;实例重启后重跑本脚本即可) ----
#   3 个检索模型 -> ~/.cache/huggingface/hub_local/<owner>__<repo>
#     (RAG/embed.py::_ensure_snapshot_offline 硬编码该路径,HF_HUB_OFFLINE=1 离线加载)
#   2 个 Qwen  -> /root/autodl-tmp/models/(vLLM 直读)
FS=/root/autodl-fs/models
if [ -d "$FS" ]; then
  HC=/root/.cache/huggingface/hub_local
  mkdir -p "$HC" /root/autodl-tmp/models
  link() { # link <fs_src> <dst>  (dst 为缺失或空目录时建软链)
    if [ -e "$2" ] && [ ! -L "$2" ] && [ -n "$(ls -A "$2" 2>/dev/null)" ]; then
      echo "== keep existing $2"; return 0
    fi
    rm -rf "$2"; ln -sfn "$1" "$2"; echo "== link $2 -> $1"
  }
  link "$FS/BAAI__bge-m3"                    "$HC/BAAI__bge-m3"
  link "$FS/BAAI__bge-reranker-v2-m3"        "$HC/BAAI__bge-reranker-v2-m3"
  link "$FS/sentence-transformers__clip-ViT-B-32" "$HC/sentence-transformers__clip-ViT-B-32"
  link "$FS/Qwen3-30B-A3B-GPTQ-Int4"         "$M/Qwen3-30B-A3B-GPTQ-Int4"
  link "$FS/Qwen3-8B-AWQ"                    "$M/Qwen3-8B-AWQ"
else
  echo "== WARN $FS not mounted; using existing local weights"
fi

# ---- 1. vLLM 主模型 :8000 ----
if ! curl -s -m2 http://127.0.0.1:8000/health >/dev/null 2>&1; then
  nohup $GW/python -m vllm.entrypoints.openai.api_server \
    --model $M/Qwen3-30B-A3B-GPTQ-Int4 --served-model-name vllm-main \
    --max-model-len 16384 --gpu-memory-utilization 0.60 \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --reasoning-parser qwen3 \
    --trust-remote-code --port 8000 > $LOG/vllm_main.log 2>&1 &
  echo "vllm-main starting pid=$!"
else echo "vllm-main already up"; fi

# ---- 1.5 等 main 健康再起 light(避免两者并发抢显存) ----
if ! curl -s -m2 http://127.0.0.1:8000/health >/dev/null 2>&1; then
  echo "waiting for vllm-main to become healthy ..."
  for i in $(seq 1 60); do
    curl -s -m2 http://127.0.0.1:8000/health >/dev/null 2>&1 && break
    sleep 10
  done
fi

# ---- 2. vLLM 轻模型 :8001 ----
if ! curl -s -m2 http://127.0.0.1:8001/health >/dev/null 2>&1; then
  nohup $GW/python -m vllm.entrypoints.openai.api_server \
    --model $M/Qwen3-8B-AWQ --served-model-name vllm-light \
    --max-model-len 8192 --max-num-seqs 64 --gpu-memory-utilization 0.30 \
    --reasoning-parser qwen3 \
    --trust-remote-code --port 8001 > $LOG/vllm_light.log 2>&1 &
  echo "vllm-light starting pid=$!"
else echo "vllm-light already up"; fi

# ---- 3. LiteLLM :4000 ----
if ! curl -s -m2 http://127.0.0.1:4000/health/liveliness >/dev/null 2>&1; then
  cd /root/autodl-tmp/gw
  export LITELLM_MASTER_KEY=REDACTED-KEY
  export VLLM_API_KEY=dummy
  export VLLM_MAIN_BASE_URL=http://127.0.0.1:8000/v1
  export VLLM_LIGHT_BASE_URL=http://127.0.0.1:8001/v1
  export CLOUD_BASE_URL=https://ark.cn-beijing.volces.com/api/coding/v1
  export CLOUD_API_KEY=REDACTED-API-KEY
  nohup $GW/litellm --config litellm_config.yaml --port 4000 > $LOG/litellm.log 2>&1 &
  echo "litellm starting pid=$!"
else echo "litellm already up"; fi

# ---- 4. 检索服务 :8002(BGE-m3/CLIP/Reranker, cuda;权重在 ~/.cache/huggingface/hub_local)----
# qdrant 在项目机 VM,经 VM 发起的 SSH 隧道 -R 6333 回到 VM 本机
if ! curl -s -m2 http://127.0.0.1:8002/health >/dev/null 2>&1; then
  cd /root/autodl-tmp/app
  set -a; source /root/autodl-tmp/app/ret.env; set +a
  nohup $GW/python -m mcp_servers.retrieval.service > $LOG/retrieval.log 2>&1 &
  echo "retrieval starting pid=$!"
else echo "retrieval already up"; fi

echo "== START SCRIPT ISSUED =="
