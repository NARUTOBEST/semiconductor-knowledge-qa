# -*- coding: utf-8 -*-
"""用 paramiko 把本地公钥推到目标机 authorized_keys,之后 ssh 免密。
用法: python ssh_setup_key.py <host> <port> <user> <password>
"""
import sys, os, paramiko

host, port, user, pwd = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
pub = open(os.path.expanduser("~/.ssh/id_ed25519.pub")).read().strip()

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(host, port=port, username=user, password=pwd, timeout=20)
cmd = (
    "mkdir -p ~/.ssh && chmod 700 ~/.ssh && "
    f"grep -qF '{pub}' ~/.ssh/authorized_keys 2>/dev/null || "
    f"echo '{pub}' >> ~/.ssh/authorized_keys; "
    "chmod 600 ~/.ssh/authorized_keys && echo KEY_OK"
)
_, out, err = c.exec_command(cmd)
print(out.read().decode(), err.read().decode())
c.close()
