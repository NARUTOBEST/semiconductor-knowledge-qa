#!/usr/bin/env python3
"""大目录流式推送: tar(不压缩) -> 远端 tar xf,恒定低内存,带进度输出。

Usage:
  python push_big.py --host HOST --port PORT --password PASS --local DIR --remote DIR
"""
import argparse
import os
import sys
import tarfile
import time

import paramiko


class Counting:
    def __init__(self, f):
        self.f = f
        self.n = 0
        self.t0 = time.time()

    def write(self, b):
        self.n += len(b)
        now = time.time()
        if now - self.t0 >= 30:
            dt = now - self.t0
            print(f"  .. {self.n/1e9:.2f}GB {self.n/dt/1e6:.1f}MB/s", flush=True)
            self.t0 = now
        return self.f.write(b)

    def flush(self):
        return self.f.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, default=22)
    ap.add_argument("--user", default="root")
    ap.add_argument("--password", required=True)
    ap.add_argument("--local", required=True)
    ap.add_argument("--remote", required=True)
    a = ap.parse_args()

    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(hostname=a.host, port=a.port, username=a.user,
                password=a.password, timeout=120, banner_timeout=60,
                auth_timeout=60)
    base = os.path.basename(os.path.normpath(a.local))
    chan = cli.get_transport().open_session()
    chan.exec_command(
        f"mkdir -p {a.remote!r} && tar xf - -C {a.remote!r} "
        f"&& echo PUSH_OK_{base}")
    cnt = Counting(chan.makefile("wb"))
    t0 = time.time()
    with tarfile.open(fileobj=cnt, mode="w|", format=tarfile.GNU_FORMAT) as tf:
        tf.add(a.local, arcname=base)
    cnt.flush()
    chan.shutdown_write()
    out = b""
    while True:
        if chan.exit_status_ready() and not chan.recv_ready():
            break
        if chan.recv_ready():
            out += chan.recv(4096)
        time.sleep(0.2)
    while chan.recv_ready():
        out += chan.recv(4096)
    rc = chan.recv_exit_status()
    dt = time.time() - t0
    print(f"[PUSH] {base} {cnt.n/1e9:.2f}GB in {dt/60:.1f}min "
          f"({cnt.n/dt/1e6:.1f}MB/s) rc={rc} {out.decode(errors='replace').strip()}",
          flush=True)
    cli.close()
    sys.exit(rc)


if __name__ == "__main__":
    main()
