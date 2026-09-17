#!/usr/bin/env python3
"""本地 -> GPU 检索服务的 SSH 端口转发(评测用)。

把本地 127.0.0.1:8002 转发到 GPU 机的 127.0.0.1:8002(retrieval 仅绑定回环)。
用法:python ssh_helper/tunnel.py &  (Ctrl+C 或 kill 结束)
"""
import argparse
import select
import socket
import socketserver
import threading

import paramiko


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="connect.westc.seetacloud.com")
    ap.add_argument("--port", type=int, default=27324)
    ap.add_argument("--user", default="root")
    ap.add_argument("--password", default="JZ/M4qneIrKx")
    ap.add_argument("--remote", default="127.0.0.1:8002")
    ap.add_argument("--local", type=int, default=8002)
    args = ap.parse_args()

    rhost, rport = args.remote.rsplit(":", 1)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=args.host, port=args.port, username=args.user,
                   password=args.password, timeout=30,
                   allow_agent=False, look_for_keys=False)
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
                r, _w, _x = select.select([self.request, chan], [], [], 60)
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
