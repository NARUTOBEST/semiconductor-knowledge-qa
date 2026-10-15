#!/bin/bash
# VM -> GPU 隧道(LLM 已走火山接入点,不再转发 4000)
# -L 0.0.0.0:8002+i -> GPU 检索服务(backend 容器经 host.docker.internal 访问)
# 纯正向单向(2026-10-10 定稿):qdrant 副本常驻 GPU 本机,检索不跨隧道查库;
# VM 上 qdrant 仅作写入主库与 judge 读源,不需要任何 -R 反向通道。
# 连接参数从同目录 tunnel_target.env 读取(HOST/PORT;由 deploy_gpu.sh 依 gpu_ssh.env
# 自动生成并同步到 VM,换 GPU 实例后重跑 deploy_gpu.sh 即可,无需手改本文件)。
# 先读 $HOME 副本(deploy_vm.sh 推送目标),再读脚本同目录副本覆盖之——
# 两处并存时同目录优先,避免旧 HOME 值悄悄生效(踩坑实录)
[ -f "$HOME/tunnel_target.env" ] && . "$HOME/tunnel_target.env"
_dir="$(cd "$(dirname "$0")" && pwd)"
[ -f "$_dir/tunnel_target.env" ] && . "$_dir/tunnel_target.env"
: "${GPU_SSH_HOST:=connect.westd.seetacloud.com}"
: "${GPU_SSH_PORT:=22}"
# 多隧道(2026-10-09):GPU 侧只有 8002 一个服务,VM 侧开 K 个本地端口各自转发
# 到它——backend 的 MCP 连接池按 RETRIEVAL_SERVICE_URLS 一一绑定,分摊 SSH
# transport 流控窗口,治单隧道的队头阻塞(20 并发队尾超时单点)。
# 回退单隧道:GPU_TUNNEL_PORTS 只留 8002 即可。
: "${GPU_TUNNEL_PORTS:=8002 8003 8004 8005 8006 8007 8008 8009 8010 8011}"
# 方案3(20261015)双卡副本:GPU_TUNNEL_TARGETS="host1:sshport:本地端口组;host2:sshport:端口组"
# 例:"a.com:111:8002 8003 8004 8005 8006;b.com:222:8007 8008 8009 8010 8011"
# 20261016 单实例双卡(G23 vGPU-32GB×2):组末可加第4段指定远端端口(缺省 8002)——
# 卡0 服务 :8002、卡1 服务 :8003(RETRIEVAL_PORT 区分),各组隧道转发到各自远端。
# 例:"a.com:111:8002 8003 8004 8005 8006:8002;a.com:111:8007 8008 8009:8003"
# 设置后 GPU_SSH_HOST/PORT 与 GPU_TUNNEL_PORTS 失效(全部由 TARGETS 描述);
# 未设置时保持单卡旧口径,一行回退。
_ssh_opts=(-N -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes
           -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null)
_watch() {  # 单端口 watchdog 子 shell:该隧道挂了只重建自己,不拖累其他端口
  local host="$1" sshport="$2" lport="$3" rport="${4:-8002}"
  while true; do
    ssh "${_ssh_opts[@]}" -p "$sshport" "root@$host" \
        -L "0.0.0.0:${lport}:127.0.0.1:${rport}"
    sleep 5
  done
}
if [ -n "${GPU_TUNNEL_TARGETS:-}" ]; then
  IFS=';' read -ra _targets <<< "$GPU_TUNNEL_TARGETS"
  for _t in "${_targets[@]}"; do
    _h="${_t%%:*}"; _rest="${_t#*:}"
    _sp="${_rest%%:*}"; _rest2="${_rest#*:}"
    _pl="${_rest2%%:*}"
    # 第4段=远端端口(缺省 8002,单卡口径不受影响)
    _rp="8002"
    if [ "${_rest2#*:}" != "$_rest2" ]; then _rp="${_rest2#*:}"; fi
    for _p in $_pl; do
      _watch "$_h" "$_sp" "$_p" "$_rp" &
    done
  done
else
  for _p in $GPU_TUNNEL_PORTS; do
    _watch "$GPU_SSH_HOST" "$GPU_SSH_PORT" "$_p" &
  done
fi
wait
