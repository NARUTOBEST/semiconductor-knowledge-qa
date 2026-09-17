#!/usr/bin/env python3
"""Push a local file or directory to a remote host over SSH (password auth).

Uses exec_command + stdin (cat / tar), NOT SFTP (AutoDL's SFTP subsystem
rejects absolute /root paths). A directory is tar.gz'd on the fly.

Usage:
  python ssh_push.py --host HOST --port PORT --user USER --password PASS \
      --local LOCAL --remote REMOTE      # LOCAL file -> REMOTE file
      --local LOCAL --remote DIR         # LOCAL dir  -> extract to DIR
"""
import argparse
import io
import os
import sys
import tarfile

import paramiko


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, default=22)
    ap.add_argument("--user", required=True)
    ap.add_argument("--password", required=True)
    ap.add_argument("--local", required=True)
    ap.add_argument("--remote", required=True)
    args = ap.parse_args()

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=args.host,
        port=args.port,
        username=args.user,
        password=args.password,
        timeout=120,
        banner_timeout=60,
        auth_timeout=60,
        allow_agent=False,
        look_for_keys=False,
    )

    is_dir = os.path.isdir(args.local)
    if is_dir:
        payload = io.BytesIO()
        base = os.path.basename(os.path.normpath(args.local))
        root = os.path.dirname(os.path.normpath(args.local))
        with tarfile.open(fileobj=payload, mode="w:gz") as tf:
            tf.add(args.local, arcname=base)
        payload.seek(0)
        data = payload.read()
        remote_cmd = f"mkdir -p {args.remote!r} && cd {args.remote!r} && tar xzf - && echo '[PUSH OK dir] {base}'"
    else:
        with open(args.local, "rb") as f:
            data = f.read()
        remote_cmd = f"mkdir -p $(dirname {args.remote!r}) && cat > {args.remote!r} && echo '[PUSH OK file] {args.remote}'"

    stdin, stdout, stderr = client.exec_command(remote_cmd, timeout=600)
    stdin.write(data)
    stdin.flush()
    stdin.channel.shutdown_write()
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    code = stdout.channel.recv_exit_status()
    client.close()
    sys.stdout.write(out)
    if err:
        sys.stderr.write(err)
    print(f"[PUSH] {args.local} -> {args.remote} (rc={code})")
    sys.exit(code if code else 0)


if __name__ == "__main__":
    main()