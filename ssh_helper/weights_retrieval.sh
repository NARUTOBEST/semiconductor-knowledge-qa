#!/bin/bash
# Pre-download retrieval weights (BGE-m3 / BGE-reranker-v2-m3 / CLIP) via ModelScope
# into the exact hub_local path RAG/embed.py::_ensure_snapshot expects
# (~/.cache/huggingface/hub_local/<repo with / -> __>), so the retrieval
# service starts with local_files_only (no first-run download).
set -e
export PATH=/root/miniconda3/bin:$PATH
pip install --quiet modelscope 2>&1 | tail -1 || true

CACHE=/root/.cache/huggingface/hub_local
mkdir -p "$CACHE"

dl() {
  local repo="$1" dst="$2"; shift 2
  if [ -f "$dst/.download_ok" ]; then echo "== SKIP $repo (done)"; return 0; fi
  echo "== downloading $repo -> $dst"
  if modelscope download --model "$repo" --local_dir "$dst" "$@"; then
    touch "$dst/.download_ok"
  else
    echo "== FAIL $repo"
  fi
}

dl "BAAI/bge-m3"             "$CACHE/BAAI__bge-m3"             --exclude "*.onnx" "onnx/*" "*.ot" "*.msgpack" "*.DS_Store" "imgs/*"
dl "BAAI/bge-reranker-v2-m3" "$CACHE/BAAI__bge-reranker-v2-m3"
# CLIP: official mirror first, AI-ModelScope fork as fallback
dl "sentence-transformers/clip-ViT-B-32" "$CACHE/sentence-transformers__clip-ViT-B-32"
if [ ! -f "$CACHE/sentence-transformers__clip-ViT-B-32/.download_ok" ]; then
  dl "AI-ModelScope/clip-ViT-B-32" "$CACHE/sentence-transformers__clip-ViT-B-32"
fi
echo "== RETRIEVAL WEIGHTS SCRIPT FINISHED =="