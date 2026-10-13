#!/bin/bash
# 下载 Qwen3-Reranker-4B(混合重排主路,轮次18)到 autodl-fs 网盘。
# hf-mirror 对 Xet-backed 仓库 401(见 weights_qwen.sh 注),统一走 ModelScope。
set -e
export PATH=/root/miniconda3/bin:$PATH
pip install --quiet modelscope 2>&1 | tail -1 || true

DST=/root/autodl-fs/models/Qwen3-Reranker-4B
if [ -f "$DST/.download_ok" ]; then echo "== SKIP (done)"; exit 0; fi
echo "== downloading Qwen/Qwen3-Reranker-4B -> $DST"
modelscope download --model "Qwen/Qwen3-Reranker-4B" --local_dir "$DST" \
  && touch "$DST/.download_ok" && echo "== DOWNLOAD OK" || echo "== DOWNLOAD FAIL"
