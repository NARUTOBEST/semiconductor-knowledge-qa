#!/bin/bash
# GPU 机启动 vLLM 重排服务 :8003(Qwen3-Reranker-4B,score/rerank task)。
# 幂等:已在跑则跳过。用法:bash start_vllm_rerank.sh [MODEL_PATH]
# 检索服务经 RERANK_API_BASE=http://127.0.0.1:8003/v1 走 continuous batching,
# 20u 并发下 vLLM 内部持续合批,治进程内 rerank 排队风暴(20261012 定案方案二)。
set -e
MODEL_PATH=${1:-/root/autodl-fs/models/Qwen3-Reranker-4B}
ENV_HOME=/root/envs-gateway
LOG=/root/autodl-tmp/logs/vllm_rerank.log

if pgrep -f "vllm[.]serve.*8003" >/dev/null 2>&1; then
  echo "vllm rerank already running"
  exit 0
fi

export LD_LIBRARY_PATH=$ENV_HOME/lib:${LD_LIBRARY_PATH:-}
export VLLM_WORKER_MULTIPROC_METHOD=fork
# 克隆实例坑:入口脚本 shebang 指向旧数据盘路径时自动修正(与 litellm 同款幂等修复)
head -1 "$ENV_HOME/bin/vllm" | grep -q "$ENV_HOME/bin/python" || \
  sed -i "1s|^#!.*|#!$ENV_HOME/bin/python|" "$ENV_HOME/bin/vllm"
nohup $ENV_HOME/bin/vllm serve "$MODEL_PATH" \
  --served-model-name Qwen/Qwen3-Reranker-4B \
  --runner pooling \
  --hf-overrides '{"architectures": ["Qwen3ForSequenceClassification"], "classifier_from_token": ["no", "yes"], "is_original_qwen3_reranker": true}' \
  --port 8003 \
  --host 127.0.0.1 \
  --gpu-memory-utilization 0.45 \
  --max-model-len 1536 \
  --max-num-seqs 64 \
  > "$LOG" 2>&1 &

echo "vllm rerank starting (pid $!), log: $LOG"
