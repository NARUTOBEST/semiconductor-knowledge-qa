#!/usr/bin/env python3
"""Minimal SSH command runner using paramiko (password + optional key auth).

Usage:
  python ssh_run.py --host HOST --port PORT --user USER --password PASS --cmd "COMMAND"
  python ssh_run.py --host HOST --port PORT --user USER --password PASS --cmd "COMMAND" --timeout 600
"""
import argparse
import sys
import paramiko


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, default=22)
    ap.add_argument("--user", required=True)
    ap.add_argument("--password", required=True)
    ap.add_argument("--cmd", required=True)
    ap.add_argument("--timeout", type=int, default=120)
    args = ap.parse_args()

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(
            hostname=args.host,
            port=args.port,
            username=args.user,
            password=args.password,
            timeout=args.timeout,
            banner_timeout=60,
            auth_timeout=60,
            allow_agent=False,
            look_for_keys=False,
        )
    except Exception as e:  # noqa
        print(f"[SSH CONNECT ERROR] {e}", file=sys.stderr)
        sys.exit(2)

    stdin, stdout, stderr = client.exec_command(args.cmd, timeout=args.timeout)
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    code = stdout.channel.recv_exit_status()
    sys.stdout.write(out)
    if err:
        sys.stderr.write(err)
    client.close()
    sys.exit(code)


if __name__ == "__main__":
    main()