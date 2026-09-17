#!/bin/bash
# Download the two Qwen weights for vLLM via ModelScope (hf-mirror 401s on
# Xet-backed repos). Targets: /root/autodl-tmp/models/<name>
set -e
export PATH=/root/miniconda3/bin:$PATH
pip install --quiet modelscope 2>&1 | tail -1 || true
mkdir -p /root/autodl-tmp/models

dl() {
  local repo="$1" dst="$2"
  if [ -f "$dst/.download_ok" ]; then echo "== SKIP $repo (done)"; return 0; fi
  echo "== downloading $repo -> $dst"
  modelscope download --model "$repo" --local_dir "$dst" && touch "$dst/.download_ok" \
    || echo "== FAIL $repo"
}

dl "Qwen/Qwen3-30B-A3B-GPTQ-Int4" /root/autodl-tmp/models/Qwen3-30B-A3B-GPTQ-Int4
dl "Qwen/Qwen3-8B-AWQ"           /root/autodl-tmp/models/Qwen3-8B-AWQ
echo "== WEIGHTS SCRIPT FINISHED =="