# -*- coding: utf-8 -*-
"""Windows 侧临时本地转发:127.0.0.1:6333 -> VM 127.0.0.1:6333 (semi-qdrant)。

qdrant 容器只绑 VM 的 127.0.0.1,Windows 侧 metrics/judge 需要该转发。
凭据同 ssh_helper/vm_ssh.env。用法: python ssh_helper/forward_6333.py [LPORT]
"""
import socket
import socketserver
import sys
import threading

import paramiko

HERE = os_dir = __import__("os").path.dirname(__file__)


def load_env():
    env = {}
    for line in open(HERE + "/vm_ssh.env", encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            env[k.strip()] = v.strip()
    return env


def pump(src, dst, tag=""):
    n = 0
    try:
        while True:
            data = src.recv(4096)
            if not data:
                break
            n += len(data)
            dst.sendall(data)
    except Exception as e:
        print(f"  [{tag}] pump err@{n}B: {e!r}", flush=True)
    finally:
        print(f"  [{tag}] pump end, {n}B", flush=True)
        try:
            dst.shutdown(2)
        except OSError:
            pass


def handler(chan, sock):
    print(f"  conn: sock={sock!r} fileno={sock.fileno()!r} "
          f"chan={chan!r}", flush=True)
    t = threading.Thread(target=pump, args=(sock, chan, "c->s"), daemon=True)
    t.start()
    pump(chan, sock, "s->c")   # 当前线程反向对拷
    t.join(timeout=5)
    chan.close()
    sock.close()


class Fwd(socketserver.BaseRequestHandler):
    def handle(self):
        chan = self.server.ssh_transport.open_channel(
            "direct-tcpip", ("127.0.0.1", self.server.dst_port),
            self.request.getpeername())
        if chan is None:
            print("  conn: open_channel FAILED", flush=True)
            return
        # 同步执行:handle() 一返回 socketserver 就会 shutdown/close 套接字,
        # 起线程后立即返回会让 pump 收到 10038(连接被提前关闭)。
        # ThreadingMixIn 已保证每个连接在独立线程里,此处阻塞即可。
        handler(chan, self.request)


class FwdServer(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    lport = int(sys.argv[1]) if len(sys.argv) > 1 else 6333
    env = load_env()
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(env["HOST"], int(env.get("PORT", 22)), env["USER"],
                env["PASS"], timeout=15)
    FwdServer.ssh_transport = cli.get_transport()
    FwdServer.dst_port = 6333
    print(f"forward 127.0.0.1:{lport} -> {env['HOST']}:6333", flush=True)
    FwdServer(("127.0.0.1", lport), Fwd).serve_forever()


if __name__ == "__main__":
    main()
