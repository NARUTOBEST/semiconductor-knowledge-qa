#!/bin/bash
# 方案A 上线一键验证:K=8 基线 -> K=5 -> 双路对照实验 -> 自动选 K -> dev-200 全量回归。
#
# 前置:GPU 实例已在 AutoDL 控制台开机(本机无 API 无法远程开机)。
# 用法(项目根执行):
#   bash ssh_helper/run_plan_a.sh            # 全流程
#   bash ssh_helper/run_plan_a.sh --exp-only # 只做 K=8/K=5 实验与选型,不跑全量回归
#
# 全量回归需 eval 前端(:3000)可达:默认 VM http://192.168.88.138:3000,
# VM 不在时用 BASE_URL=http://127.0.0.1:3000 走本地应用(检索经 tunnel.py 转发)。
set -e
cd "$(dirname "$0")/.."
export MSYS_NO_PATHCONV=1
export MSYS2_ARG_CONV_EXCL="*"
# 旁路本地代理(127.0.0.1 被 Clash 劫持会 502)
export NO_PROXY=127.0.0.1,localhost,192.168.88.138,.volces.com
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
source ssh_helper/gpu_ssh.env

EXP_ONLY=0
[ "$1" = "--exp-only" ] && EXP_ONLY=1
BASE_URL=${BASE_URL:-http://192.168.88.138:3000}

# ret.env 是检索服务的本地配置源(deploy 时推送远端);K 通过改它下发
set_k() {
  if grep -q "^RERANK_FIXED_K=" ssh_helper/ret.env; then
    sed -i "s/^RERANK_FIXED_K=.*/RERANK_FIXED_K=$1/" ssh_helper/ret.env
  else
    echo "RERANK_FIXED_K=$1" >> ssh_helper/ret.env
  fi
}

echo "== [0/5] GPU 可达性检查"
python ssh_helper/ssh_run.py --host "$HOST" --port "$PORT" --user "$USER" \
  --password "$PASS" --timeout 20 --cmd "echo GPU-UP" \
  || { echo "!! GPU 不可达:请先在 AutoDL 控制台开机后重跑"; exit 2; }

echo "== [0/5] 本地隧道 8002 -> GPU 8002(已通则复用)"
if [ "$(python -c "import socket,sys;sys.exit(0 if socket.socket().connect_ex(('127.0.0.1',8002))==0 else 1)")" = "0" ]; then
  echo "   tunnel already up"
else
  nohup python ssh_helper/tunnel.py > /tmp/plan_a_tunnel.log 2>&1 &
  sleep 3
  python -c "import socket,sys;sys.exit(0 if socket.socket().connect_ex(('127.0.0.1',8002))==0 else 1)" \
    || { echo "!! 隧道启动失败,见 /tmp/plan_a_tunnel.log"; exit 2; }
  echo "   tunnel started"
fi

echo "== [1/5] 部署 K=8(基线),跑双路对照实验 --k 8"
set_k 8
bash ssh_helper/deploy_gpu.sh all   # all: 幂等拉起 vLLM+LiteLLM(:4000)+ 重启检索(:8002)
python eval/pipeline/dev/glossary_experiment.py --k 8 --out glossary_exp_k8

echo "== [2/5] 部署 K=5,跑双路对照实验 --k 5"
set_k 5
bash ssh_helper/deploy_gpu.sh restart
python eval/pipeline/dev/glossary_experiment.py --k 5 --out glossary_exp_k5

echo "== [3/5] 自动选 K(目标:recall@k 不低于 K=8 基线)"
python - <<'PY'
import json, os, re
root = os.getcwd()
def load(name):
    p = os.path.join(root, "eval/runs/dev/glossary_exp", name, "results.json")
    return json.load(open(p, encoding="utf-8"))["summary"]
s8, s5 = load("glossary_exp_k8"), load("glossary_exp_k5")
base = s8["dual0"]["recall@k"]      # K=8 单路 = 改造前基线
d8 = s8["dual1"]["recall@k"]        # K=8 双路
d5 = s5["dual1"]["recall@k"]        # K=5 双路
print(f"K=8: single={base:.4f} dual={d8:.4f}   K=5: dual={d5:.4f}")
# 双路把金块整体前移 → K=5 若追平 K=8 基线(差 0.5pt 内)即采用,否则回退 8
k = 5 if d5 >= base - 0.005 else 8
print(f"DECISION: RERANK_FIXED_K={k}")
env = os.path.join(root, "ssh_helper", "run_plan_a_decision.env")
open(env, "w").write(f"DECIDED_K={k}\n")
PY
source ssh_helper/run_plan_a_decision.env
set_k "$DECIDED_K"
bash ssh_helper/deploy_gpu.sh restart
echo "== 最终配置已部署: RERANK_FIXED_K=$DECIDED_K(双路默认开)"

if [ "$EXP_ONLY" = "1" ]; then echo "== DONE (exp-only)"; exit 0; fi

echo "== [4/5] dev-200 全量重跑(跑测 + 裁判 + 报告,前端 $BASE_URL)"
python -m eval.pipeline.dev.pipeline_50x5.make_qasets_dev200 --users 10
python -m eval.pipeline.dev.pipeline_50x5.run_concurrent --users 10 \
    --base "$BASE_URL" \
    --sets-dir eval/pipeline/dev/pipeline_50x5/runs/dev200_baseline/qa_sets \
    --out eval/pipeline/dev/pipeline_50x5/runs/dev200
python -m eval.pipeline.dev.pipeline_50x5.judge_50x5 \
    --results-dir eval/pipeline/dev/pipeline_50x5/runs/dev200 \
    --out-dir eval/pipeline/dev/pipeline_50x5/runs/dev200 \
    --meta eval/qa_sets/v2_20261001/dev_200.json --force
python eval/pipeline/dev/pipeline_50x5/make_dev200_report.py

echo "== [5/5] 完成。报告见 eval/pipeline/dev/pipeline_50x5/runs/dev200/ 与"
echo "   eval/runs/dev/history/。GPU 用完记得去 AutoDL 控制台关机(¥6.98/h)。"
