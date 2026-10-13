#!/bin/bash
# 本地代码 -> GPU 机 /root/autodl-tmp/app 增量同步,并重启检索服务 :8002。
# 用法(项目根执行):
#   bash ssh_helper/deploy_gpu.sh [restart|norestart|llm|all]
#     restart(默认) 同步代码 + 重启检索 :8002
#     norestart      仅同步代码
#     llm            同步代码 + 推送 LLM 网关配置 + 幂等启动 LLM 全栈(start_gpu.sh:
#                    vLLM main/light + LiteLLM :4000 + 检索,已在跑的服务自动跳过)
#     all            llm + 重启检索
# 依赖:ssh_helper/ssh_push.py(目录自动 tar.gz -> 远端解压)。
set -e
cd "$(dirname "$0")/.."
# Git Bash(MSYS)会把 "/root/..." 形式的参数改写成 "C:/Git/...",禁用之
export MSYS_NO_PATHCONV=1
export MSYS2_ARG_CONV_EXCL="*"
# 连接信息从本地 ssh_helper/gpu_ssh.env 读取(不入库):HOST/PORT/USER/PASS
source "$(dirname "$0")/gpu_ssh.env"
APP=/root/autodl-tmp/app
GWDIR=/root/autodl-tmp/gw
MODE=${1:-restart}

# LLM 网关侧(litellm_config.yaml / start_gpu.sh / gpu.env -> $GWDIR;密钥只落 GPU 机本地)
if [ "$MODE" = "llm" ] || [ "$MODE" = "all" ]; then
  echo "== push litellm_config_alias.yaml -> $GWDIR/litellm_config.yaml"
  python ssh_helper/ssh_push.py --host $HOST --port $PORT --user $USER \
    --password "$PASS" --local ssh_helper/litellm_config_alias.yaml --remote "$GWDIR/litellm_config.yaml"
  echo "== push start_gpu.sh -> $GWDIR/"
  python ssh_helper/ssh_push.py --host $HOST --port $PORT --user $USER \
    --password "$PASS" --local ssh_helper/start_gpu.sh --remote "$GWDIR/start_gpu.sh"
  echo "== push gpu.env -> $GWDIR/(密钥,留在 GPU 机)"
  python ssh_helper/ssh_push.py --host $HOST --port $PORT --user $USER \
    --password "$PASS" --local ssh_helper/gpu.env --remote "$GWDIR/gpu.env"
fi

# 检索服务运行面:mcp_servers + RAG + config + env(其余 agent 侧代码不在 GPU 上)
for d in mcp_servers RAG config env; do
  echo "== push $d -> $APP/$d"
  python ssh_helper/ssh_push.py --host $HOST --port $PORT --user $USER \
    --password "$PASS" --local "$d" --remote "$APP/"
done
echo "== push ret.env"
python ssh_helper/ssh_push.py --host $HOST --port $PORT --user $USER \
  --password "$PASS" --local ssh_helper/ret.env --remote "$APP/ret.env"
echo "== push tool_circuit.py"
python ssh_helper/ssh_push.py --host $HOST --port $PORT --user $USER \
  --password "$PASS" --local tool_circuit.py --remote "$APP/tool_circuit.py"
echo "== push start_vllm_rerank.sh -> $GWDIR/"
python ssh_helper/ssh_push.py --host $HOST --port $PORT --user $USER \
  --password "$PASS" --local ssh_helper/start_vllm_rerank.sh --remote "$GWDIR/start_vllm_rerank.sh"

# 生成 VM 侧隧道目标文件(换 GPU 实例后重跑本脚本即自动更新)
printf 'GPU_SSH_HOST=%s\nGPU_SSH_PORT=%s\n' "$HOST" "$PORT" > ssh_helper/tunnel_target.env
echo "== tunnel_target.env updated (HOST=$HOST PORT=$PORT)"

if [ "$MODE" = "norestart" ]; then echo "== SYNC ONLY (no restart)"; exit 0; fi

if [ "$MODE" = "llm" ] || [ "$MODE" = "all" ]; then
  echo "== start LLM stack (start_gpu.sh,幂等;LIGHT_ALIAS_MAIN=${LIGHT_ALIAS_MAIN:-0})"
  python ssh_helper/ssh_run.py --host $HOST --port $PORT --user $USER \
    --password "$PASS" --timeout 120 --cmd "
mkdir -p /root/autodl-tmp/logs; LIGHT_ALIAS_MAIN=${LIGHT_ALIAS_MAIN:-0} bash $GWDIR/start_gpu.sh > /root/autodl-tmp/logs/start_gpu.out 2>&1; sleep 8; cat /root/autodl-tmp/logs/start_gpu.out"
fi

if [ "$MODE" = "llm" ]; then echo "== DONE (LLM issued)"; exit 0; fi

echo "== start vLLM rerank :8003 (幂等)"
python ssh_helper/ssh_run.py --host $HOST --port $PORT --user $USER \
  --password "$PASS" --timeout 60 --cmd "bash $GWDIR/start_vllm_rerank.sh"

echo "== restart retrieval :8002"
python ssh_helper/ssh_run.py --host $HOST --port $PORT --user $USER \
  --password "$PASS" --timeout 60 --cmd '
# 防自杀:启动命令文本里含明文 "mcp_servers.retrieval.service"(nohup 行),
# pkill -f 会连同自身 shell 一起杀掉(表现为无输出 exit 127/服务起不来)。
# 改为 pgrep 枚举后排除自身 $$ 与父进程再 kill。
for pid in $(pgrep -f "mcp_servers[.]retrieval[.]service"); do
  [ "$pid" != "$$" ] && [ "$pid" != "$PPID" ] && kill "$pid" 2>/dev/null
done
sleep 2
cd /root/autodl-tmp/app
# env 优先系统盘镜像副本(方案B),缺失回退数据盘
ENV_HOME=/root/envs-gateway
[ -x "$ENV_HOME/bin/python" ] || ENV_HOME=/root/autodl-tmp/envs/gateway
export PATH=/root/miniconda3/bin:$ENV_HOME/bin:$PATH
export LD_LIBRARY_PATH=$ENV_HOME/lib:${LD_LIBRARY_PATH:-}
# qdrant 本机常驻(GPU 机,2026-10-10 定稿;VM qdrant 仅作写入主库):杀旧 qdrant/LB
# 后重启。storage 目录 /root/autodl-tmp/qdrant_storage 需已就位(快照解压,见 DEPLOY.md)。
for pid in $(pgrep -f "autodl-tmp/bin/[q]drant"); do
  [ "$pid" != "$$" ] && [ "$pid" != "$PPID" ] && kill "$pid" 2>/dev/null
done
for pid in $(pgrep -f "qdrant_[l]b.py"); do
  [ "$pid" != "$$" ] && [ "$pid" != "$PPID" ] && kill "$pid" 2>/dev/null
done
{
  echo "storage:"
  echo "  storage_path: /root/autodl-tmp/qdrant_storage"
  echo "service:"
  echo "  host: 127.0.0.1"
  echo "  port: 6333"
} > /root/autodl-tmp/qdrant_config.yaml
nohup /root/autodl-tmp/bin/qdrant --config-path /root/autodl-tmp/qdrant_config.yaml \
  > /root/autodl-tmp/logs/qdrant.log 2>&1 &
sleep 1
set -a; source /root/autodl-tmp/app/ret.env; set +a
nohup env ${RERANK_FIXED_K:+RERANK_FIXED_K=$RERANK_FIXED_K} ${RETRIEVAL_GLOSSARY_DUAL:+RETRIEVAL_GLOSSARY_DUAL=$RETRIEVAL_GLOSSARY_DUAL} \
  $ENV_HOME/bin/python -m mcp_servers.retrieval.service \
  > /root/autodl-tmp/logs/retrieval.log 2>&1 &
sleep 6  # 等 nohup 子进程脱离会话,否则 ssh_run 退出会连带走掉(实测)
pgrep -af "mcp_servers[.]retrieval[.]service" >/dev/null && echo "retrieval UP" || echo "retrieval FAILED to start"'
echo "== DONE"
