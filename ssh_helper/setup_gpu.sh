#!/bin/bash
# GPU server native deployment setup (no Docker — DinD is blocked on AutoDL).
# 单一共享 conda env:vllm 的 torch 复用给检索(BGE-m3/CLIP/Reranker),数据盘 50G 才装得下。
#   gateway (python 3.12): vllm + litellm[proxy] + FlagEmbedding + sentence-transformers
#                          + fastapi/uvicorn + qdrant-client + mcp + boto3
set -e
export PATH=/root/miniconda3/bin:$PATH
. /root/miniconda3/etc/profile.d/conda.sh
export PIP_CACHE_DIR=/root/autodl-tmp/pipcache
export HF_HOME=/root/autodl-tmp/hf_cache
mkdir -p /root/autodl-tmp/envs /root/autodl-tmp/pipcache /root/autodl-tmp/hf_cache

# China mirrors (fast)
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple 2>/dev/null || true
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main 2>/dev/null || true
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free 2>/dev/null || true
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge 2>/dev/null || true
conda config --set show_channel_urls yes 2>/dev/null || true

echo "===== [1/2] gateway env (vllm + litellm + retrieval) ====="
conda create -p /root/autodl-tmp/envs/gateway python=3.12 -y
conda activate /root/autodl-tmp/envs/gateway
pip install --upgrade pip
pip install vllm "litellm[proxy]"
echo "gateway vllm:" && python -c "import vllm; print('vllm', vllm.__version__)" 2>&1 | tail -3 || true
# 检索服务依赖(torch 复用 vllm wheel 自带的 CUDA 运行时,不装系统 CUDA)
pip install FlagEmbedding sentence-transformers fastapi uvicorn qdrant-client "mcp>=2.2.0" boto3 pillow
echo "retrieval deps:" && python -c "import FlagEmbedding, sentence_transformers, fastapi, qdrant_client, mcp; print('retrieval deps OK')" 2>&1 | tail -3 || true

echo "===== [2/2] cleanup caches ====="
conda clean -a -y >/dev/null 2>&1 || true
rm -rf "$PIP_CACHE_DIR"/* 2>/dev/null || true

echo "===== ALL SETUP DONE ====="