#!/bin/bash
# GPU 服务器服务启动(2026-10-11 改版:混合重排全进程内,不再起 vLLM)。
#   只起检索服务 :8002(编码 + 双路重排都在进程内),nohup 后台 + 日志落盘。
#   vLLM :8003 已弃用:vLLM 0.29 载不上 CausalLM 打分(/score 未注册,classify
#   硬转退化恒 0.5),gemma 时代即回退进程内;混合重排(Qwen3-Reranker-4B +
#   bge-reranker-v2-m3)继续走进程内 _BatchedReranker 攒批,不留 vLLM 白吃显存。
# 显存布局(RTX 4080S 24G):Qwen3-4B bf16 ~8G + bge-m3 ~1.2G + 编码三件套
#   (bge-m3 embed/CLIP)~3G + 余量,充裕。
#   不再起 vLLM LLM/LiteLLM(backend 公网直连 Ark api/v3,见 DEPLOY.md)。
ENV_HOME=/root/envs-gateway
[ -x "$ENV_HOME/bin/python" ] || ENV_HOME=/root/autodl-tmp/envs/gateway
export PATH=/root/miniconda3/bin:$ENV_HOME/bin:$PATH   # EngineCore 子进程要能找到 ninja 等
export LD_LIBRARY_PATH=$ENV_HOME/lib:${LD_LIBRARY_PATH:-}   # conda icu/libstdc++ 优先于系统旧库
GW=$ENV_HOME/bin
LOG=/root/autodl-tmp/logs
mkdir -p $LOG

# ---- 0. 权重走 autodl-fs 网盘(幂等 symlink;实例重启后重跑本脚本即可) ----
#   重排/编码模型 -> ~/.cache/huggingface/hub_local/<owner>__<repo>
#     (RAG/embed.py::_ensure_snapshot_offline 硬编码该路径,HF_HUB_OFFLINE=1 离线加载;
#      autodl-fs 内目录名不限定,按通配符探测后链接到规范的 <owner>__<repo> 名)
FS=/root/autodl-fs/models
if [ -d "$FS" ]; then
  HC=/root/.cache/huggingface/hub_local
  mkdir -p "$HC"
  link() { # link <fs_src> <dst>  (dst 为缺失或空目录时建软链)
    if [ -e "$2" ] && [ ! -L "$2" ] && [ -n "$(ls -A "$2" 2>/dev/null)" ]; then
      echo "== keep existing $2"; return 0
    fi
    rm -rf "$2"; ln -sfn "$1" "$2"; echo "== link $2 -> $1"
  }
  link_src() { # link_src <glob> <dst>:autodl-fs 下按通配符找权重目录
    local src; src=$(ls -d "$FS"/$1 2>/dev/null | head -1)
    if [ -n "$src" ]; then link "$src" "$2"; else echo "== WARN $FS 下未找到 $1"; fi
  }
  link_src "*bge-m3"                     "$HC/BAAI__bge-m3"
  link_src "*bge-reranker-v2-m3"         "$HC/BAAI__bge-reranker-v2-m3"
  link_src "*Qwen3-Reranker-4B"          "$HC/Qwen__Qwen3-Reranker-4B"
  link_src "*clip-ViT-B-32*"             "$HC/sentence-transformers__clip-ViT-B-32"
else
  echo "== WARN $FS not mounted; using existing local weights"
fi

# ---- 1. 检索服务 :8002(BGE-m3/CLIP 编码 + 双路重排全进程内) ----
# qdrant 在项目机 VM,经 VM 发起的 SSH 隧道 -R 6333 回到 VM 本机
if ! curl -s -m2 http://127.0.0.1:8002/health >/dev/null 2>&1; then
  cd /root/autodl-tmp/app
  set -a; source /root/autodl-tmp/app/ret.env; set +a
  CUDA_VISIBLE_DEVICES=${RETRIEVAL_GPUS:-0} nohup $GW/python -m mcp_servers.retrieval.service > $LOG/retrieval.log 2>&1 &
  echo "retrieval starting pid=$!"
else echo "retrieval already up"; fi

echo "== START SCRIPT ISSUED =="
