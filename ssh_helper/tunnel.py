#!/usr/bin/env python3
"""本地 -> GPU 检索服务的 SSH 端口转发(评测用)。

把本地 127.0.0.1:8002 转发到 GPU 机的 127.0.0.1:8002(retrieval 仅绑定回环)。
用法:python ssh_helper/tunnel.py &  (Ctrl+C 或 kill 结束)
"""
import argparse
import os
import select
import socket
import socketserver
import threading

import paramiko

# 连接信息默认从 ssh_helper/tunnel_target.env(GPU_SSH_HOST/PORT,deploy_gpu.sh
# 生成)与 gpu_ssh.env(HOST/PORT/USER/PASS)读取,不再硬编码旧实例地址。
_HERE = os.path.dirname(os.path.abspath(__file__))


def _env(key, default):
    for fname in ("tunnel_target.env", "gpu_ssh.env"):
        path = os.path.join(_HERE, fname)
        if not os.path.exists(path):
            continue
        for line in open(path, encoding="utf-8"):
            if line.startswith(key + "="):
                return line.split("=", 1)[1].strip()
    return default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=_env("GPU_SSH_HOST", _env("HOST", "")))
    ap.add_argument("--port", type=int, default=int(_env("GPU_SSH_PORT", _env("PORT", "0"))))
    ap.add_argument("--user", default=_env("USER", "root"))
    ap.add_argument("--password", default=_env("PASS", ""))
    ap.add_argument("--remote", default="127.0.0.1:8002")
    ap.add_argument("--local", type=int, default=8002)
    args = ap.parse_args()
    if not args.host or not args.port:
        raise SystemExit("tunnel: 未取到 GPU 地址,请检查 ssh_helper/tunnel_target.env / gpu_ssh.env")

    rhost, rport = args.remote.rsplit(":", 1)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=args.host, port=args.port, username=args.user,
                   password=args.password, timeout=30,
)
    transport = client.get_transport()
    print(f"tunnel 127.0.0.1:{args.local} -> {rhost}:{rport} UP", flush=True)

    class Handler(socketserver.BaseRequestHandler):
        def handle(self):
            try:
                chan = transport.open_channel(
                    "direct-tcpip", (rhost, int(rport)),
                    self.request.getpeername())
            except Exception:
                return
            if chan is None:
                return
            while True:
                # 600s 空闲容忍:长请求(如批量 payload 写入触发 qdrant 索引优化器)期间
                # 连接上无数据流动,60s 会把还在等响应的连接掐断(踩坑实录)
                r, _w, _x = select.select([self.request, chan], [], [], 600)
                if not r:
                    break
                if self.request in r:
                    data = self.request.recv(65536)
                    if not data:
                        break
                    chan.sendall(data)
                if chan in r:
                    data = chan.recv(65536)
                    if not data:
                        break
                    self.request.sendall(data)
            try:
                chan.close()
            except Exception:
                pass
            try:
                self.request.close()
            except Exception:
                pass

    class TS(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

    with TS(("127.0.0.1", args.local), Handler) as srv:
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
