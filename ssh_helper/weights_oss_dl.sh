#!/bin/bash
# 从阿里云 OSS 下载 5 套权重:LLM×2 -> /root/autodl-tmp/models;检索三件套 -> ~/.cache/huggingface/hub_local
set -e
mkdir -p /root/autodl-tmp/models ~/.cache/huggingface/hub_local
cd /root/autodl-tmp

# ossutil v2 linux(仅装一次)
if ! command -v ossutil >/dev/null 2>&1; then
  wget -q https://gosspublic.alicdn.com/ossutil/v2/2.1.1/ossutil-2.1.1-linux-amd64.zip -O /tmp/ossutil.zip
  apt-get install -y unzip >/dev/null 2>&1 || true
  unzip -oq /tmp/ossutil.zip -d /tmp/ossutilx
  install /tmp/ossutilx/*/ossutil /usr/local/bin/ossutil && chmod +x /usr/local/bin/ossutil
fi

B=oss://my-image-bucket-lly/models
CK=/root/autodl-tmp/.ck
dl() { echo "=== $2 -> $1 ==="; ossutil cp -r -f -u --parallel 8 --job 4 --checkpoint-dir $CK "$2" "$1"; }

dl /root/autodl-tmp/models/ $B/Qwen3-8B-AWQ/
dl /root/autodl-tmp/models/ $B/Qwen3-30B-A3B-GPTQ-Int4/
for d in BAAI__bge-m3 BAAI__bge-reranker-v2-m3 sentence-transformers__clip-ViT-B-32; do
  dl ~/.cache/huggingface/hub_local/ $B/$d/
done

echo "== WEIGHTS DONE =="
ls /root/autodl-tmp/models
du -sh /root/autodl-tmp/models ~/.cache/huggingface/hub_local
