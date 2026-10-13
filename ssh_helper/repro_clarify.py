# -*- coding: utf-8 -*-
"""复现 36 题空答案:本地起 14000->VM:4000 隧道,用 classify_and_clarify 真跑。
用法: python ssh_helper/repro_clarify.py
"""
import json, glob, os, sys, threading, socket, socketserver

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(HERE)

# --- 隧道(复用 forward_6333 的模式,dst 改 4000) ---
import paramiko

def load_env():
    env = {}
    for line in open(HERE + "/vm_ssh.env", encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            env[k.strip()] = v.strip()
    return env


def pump(src, dst):
    try:
        while True:
            data = src.recv(65536)
            if not data:
                break
            dst.sendall(data)
    except Exception:
        pass
    finally:
        try:
            dst.shutdown(2)
        except OSError:
            pass


class Fwd(socketserver.BaseRequestHandler):
    def handle(self):
        chan = self.server.ssh_transport.open_channel(
            "direct-tcpip", ("127.0.0.1", self.server.dst_port),
            self.request.getpeername())
        if chan is None:
            return
        threading.Thread(target=pump, args=(self.request, chan), daemon=True).start()
        pump(chan, self.request)


class FwdServer(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True


def start_tunnel(lport=14000, dst=4000):
    env = load_env()
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(env["HOST"], int(env.get("PORT", 22)), env["USER"],
                env["PASS"], timeout=15)
    FwdServer.ssh_transport = cli.get_transport()
    FwdServer.dst_port = dst
    t = threading.Thread(
        target=FwdServer(("127.0.0.1", lport), Fwd).serve_forever, daemon=True)
    t.start()
    print(f"tunnel 127.0.0.1:{lport} -> VM:{dst}", flush=True)


# --- 环境:LLM 走隧道网关 ---
os.environ.setdefault("LLM_GATEWAY_BASE_URL", "http://127.0.0.1:14000/v1")
os.environ.setdefault("LLM_GATEWAY_API_KEY",
                      "sk-8e4193de2b582e711b15c9bcb808ad585ddd0bf70d1884cc42397c99b5f165d6")
os.environ.setdefault("TIER_MODEL_SIMPLE", "light")
sys.path.insert(0, ROOT)

start_tunnel()

from agent_reasoning.router import classify_and_clarify  # noqa: E402

recs = {}
for f in glob.glob(os.path.join(
        ROOT, "eval/pipeline/dev/pipeline_50x5/runs/dev200_cap8/eval*.jsonl")):
    for line in open(f, encoding="utf-8"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        recs[r["id"]] = r
empty = sorted([r for r in recs.values() if not (r.get("answer") or "").strip()],
               key=lambda x: x["id"])

out = []
for r in empty:
    q = (r.get("q") or "").strip()
    try:
        e = classify_and_clarify(q, [])
    except Exception as ex:
        e = {"error": repr(ex)}
    out.append({"id": r["id"], "q": q, "need_clarify": e.get("need_clarify"),
                "tier": e.get("tier"), "source": e.get("source"),
                "clarify_q": e.get("question", "")})
    print(f"{r['id']} need_clarify={e.get('need_clarify')} tier={e.get('tier')} "
          f"src={e.get('source')} | {q[:40]}", flush=True)

n_clarify = sum(1 for o in out if o.get("need_clarify"))
print(f"\n== need_clarify=true: {n_clarify}/{len(empty)}", flush=True)
json.dump(out, open(os.path.join(HERE, "repro_clarify_result.json"), "w",
                    encoding="utf-8"), ensure_ascii=False, indent=1)
