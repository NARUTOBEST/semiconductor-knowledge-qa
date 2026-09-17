# -*- coding: utf-8 -*-
"""50x5 运行的 LLM-as-Judge(无第三方依赖,urllib 直连)。

口径与 eval/judge.py 一致:
- faithfulness(忠实度→幻觉率): 答案原子陈述被检索资料支撑的比例;<0.75 记幻觉
- context_precision(检索精确率): 检索块中对回答问题有用的占比
- context_recall(召回率): 参考要点(facts)能在资料中找到依据的占比
- answer_relevancy(切题度): LLM 判定(嵌入路省略,检索机带宽优先保压测链路)

用法: python eval/judge_50x5.py   (结果追加写回 jsonl + judge_summary_50x5.json)
"""
import json, os, re, sys, time, glob, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
# 密钥不入库:优先已有环境变量,否则从本地 env/env.env 读取
_ENV_FILE = os.path.join(ROOT, "env", "env.env")
if os.path.exists(_ENV_FILE):
    for _line in open(_ENV_FILE, encoding="utf-8"):
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())
ARK_KEY = os.getenv("OPENAI_API_KEY", "")
ARK_BASE = "https://ark.cn-beijing.volces.com/api/v3"
JUDGE_MODEL = "ep-20260917012501-44sxb"          # 副模型(轻量)做裁判
RET_BASE = "http://192.168.88.138:8002"
RET_TOKEN = os.getenv("RETRIEVAL_INTERNAL_TOKEN", "")
HALLUC_THRESHOLD = 0.75
WORKERS = 8

SYS = "你是严格的事实核查员。只输出 JSON,不要输出多余文字。"


def chat_json(user, retries=4):
    body = json.dumps({
        "model": JUDGE_MODEL,
        "messages": [{"role": "system", "content": SYS},
                     {"role": "user", "content": user}],
        "temperature": 0,
        "response_format": {"type": "json_object"},
        # Doubao-Seed-2.0-lite 混合思考:非流式+思考会拖到分钟级,显式关闭
        "thinking": {"type": "disabled"},
    }).encode()
    last = None
    for a in range(retries):
        try:
            req = urllib.request.Request(
                ARK_BASE + "/chat/completions", data=body,
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {ARK_KEY}"})
            with urllib.request.urlopen(req, timeout=60) as r:
                text = json.load(r)["choices"][0]["message"]["content"].strip()
            m = re.search(r"\{.*\}", text, re.S)
            return json.loads(m.group(0) if m else text)
        except Exception as e:
            last = e
            time.sleep(1.5 * (2 ** a))
    print(f"  judge LLM fail: {str(last)[:100]}", flush=True)
    return None


def get_chunk_full(chunk_id):
    try:
        req = urllib.request.Request(
            RET_BASE + "/get_chunk",
            data=json.dumps({"chunk_id": chunk_id}).encode(),
            headers={"Content-Type": "application/json",
                     "X-Internal-Token": RET_TOKEN})
        with urllib.request.urlopen(req, timeout=30) as r:
            return str((json.load(r) or {}).get("content") or "")
    except Exception:
        return ""


def faithfulness(answer, contexts):
    ctx = "\n\n".join(f"[资料{i+1}] {c}" for i, c in enumerate(contexts))
    d = chat_json(
        "下面是系统回答与它检索到的内部资料。请:\n"
        "1) 把【回答】拆成若干条可独立核验的原子陈述(claims),短句即可;\n"
        "2) 逐条判断该陈述能否由【资料】内容蕴含(supported=true/false)。\n"
        "判断标准(语义蕴含,不要求逐字对应):与资料含义一致的同义改写、直接推论、"
        "合理具体化都算 supported=true;只有资料无法推出、或与资料矛盾的陈述才算 false。\n"
        f"【回答】\n{answer}\n\n【资料】\n{ctx}\n\n"
        '输出格式:{"claims":[{"claim":"...","supported":true}]}')
    if not d or not isinstance(d.get("claims"), list) or not d["claims"]:
        return None
    sup = sum(1 for c in d["claims"] if c.get("supported"))
    return round(sup / len(d["claims"]), 4)


def context_precision(question, contexts):
    blocks = "\n".join(f"[资料{i+1}] {c}" for i, c in enumerate(contexts))
    d = chat_json(
        "你是检索相关性评判员。只输出 JSON。\n"
        f"问题:{question}\n\n以下是检索回的资料块,请逐块判断它对【回答该问题】"
        "是否有用(相关且能提供依据,relevant=true;无关/跑题=false)。\n"
        f"{blocks}\n\n"
        '输出格式:{"blocks":[{"index":1,"relevant":true}]}')
    if not d or not isinstance(d.get("blocks"), list) or not d["blocks"]:
        return None
    rel = sum(1 for b in d["blocks"] if b.get("relevant"))
    return round(rel / len(d["blocks"]), 4)


def context_recall(gt, contexts):
    ctx = "\n\n".join(f"[资料{i+1}] {c}" for i, c in enumerate(contexts))
    d = chat_json(
        "下面是问题的【参考答案】与系统检索到的【资料】。请把参考答案拆成若干要点,"
        "逐点判断该要点能否由资料支撑(supported=true/false)。\n"
        f"参考答案:{gt}\n\n资料:\n{ctx}\n\n"
        '输出格式:{"points":[{"point":"...","supported":true}]}')
    if not d or not isinstance(d.get("points"), list) or not d["points"]:
        return None
    sup = sum(1 for p in d["points"] if p.get("supported"))
    return round(sup / len(d["points"]), 4)


def answer_relevancy(question, answer):
    d = chat_json(
        "你评判问答切题度。只输出 JSON。\n"
        f"问题:{question}\n回答:{answer}\n\n"
        "判断回答是否正面、切题地回应了问题(1=完全切题,0=完全答非所问)。\n"
        '输出格式:{"relevance":0.0到1.0的数}')
    if isinstance(d, dict) and isinstance(d.get("relevance"), (int, float)):
        return round(max(0.0, min(1.0, float(d["relevance"]))), 4)
    return None


def judge_rec(rec):
    answer = (rec.get("answer") or "").strip()
    sources = rec.get("sources") or []
    if not answer or not sources:
        return rec
    contexts = []
    for s in sources:
        content = str(s.get("content", "")).strip()
        if not content:
            continue
        cid = s.get("chunk_id")
        if cid:
            full = get_chunk_full(str(cid))
            if full:
                content = full
        meta = " ".join(str(s.get(k, "")).strip()
                        for k in ("source_stem", "page", "heading") if s.get(k))
        contexts.append(f"(出处:{meta})\n{content}" if meta else content)
    if not contexts:
        return rec
    rec["judge_faithfulness"] = faithfulness(answer, contexts)
    rec["judge_context_precision"] = context_precision(rec.get("q", ""), contexts)
    rec["judge_answer_relevancy"] = answer_relevancy(rec.get("q", ""), answer)
    facts = rec.get("facts") or []
    if facts:
        gt = "；".join("、".join(str(x) for x in (f if isinstance(f, (list, tuple)) else [f]))
                      for f in facts)
        rec["judge_context_recall"] = context_recall(gt, contexts)
    f_val = rec["judge_faithfulness"]
    # 无参考要点=闲聊/元问题,回答不含可校验事实主张,faithfulness 无从谈起,
    # 不计入幻觉口径(RAGAS 口径:无 claims 时不判不忠实)
    rec["judge_chitchat"] = not facts
    rec["has_hallucination"] = bool(facts and f_val is not None
                                    and f_val < HALLUC_THRESHOLD)
    rec["judge_done"] = True
    return rec


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=os.path.join(HERE, "results_concurrent"),
                    help="原始评测 jsonl 目录(读取)")
    ap.add_argument("--out-dir", default=os.path.join(HERE, "results_50x5"),
                    help="判定结果落盘目录(可与 results-dir 相同=原地追加)")
    args = ap.parse_args()
    src_dir = args.results_dir
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    # 去重取本运行记录(最后一次写入)
    latest = {}
    for f in glob.glob(os.path.join(src_dir, "eval*.jsonl")):
        for line in open(f, encoding="utf-8"):
            try:
                r = json.loads(line)
                if str(r.get("id", "")).startswith("u") and r.get("user"):
                    latest[(r["user"], r["id"])] = r
            except Exception:
                pass
    by_user = {}
    for (u, _id), r in latest.items():
        by_user.setdefault(u, []).append(r)
    # 断点:已有 judge 字段的跳过
    todo = []
    for u, recs in sorted(by_user.items()):
        out_path = os.path.join(out_dir, f"{u}.jsonl")
        if os.path.exists(out_path):
            old = {}
            for line in open(out_path, encoding="utf-8"):
                try:
                    r = json.loads(line)
                    old[r["id"]] = r
                except Exception:
                    pass
            for r in recs:
                old.setdefault(r["id"], r)
            recs = list(old.values())
        todo.extend([(out_path, r) for r in recs
                     if not r.get("judge_done") and (r.get("answer") or "").strip()
                     and r.get("sources")])
    print(f"待判定: {len(todo)} 条", flush=True)
    judged_list = []
    with ThreadPoolExecutor(WORKERS) as ex:
        judged_list = list(ex.map(lambda t: judge_rec(t[1]), todo))
    for i, rec in enumerate(judged_list, 1):
        path = todo[i - 1][0]
        if i % 25 == 0 or i == len(todo):
            print(f"  judged {i}/{len(todo)}", flush=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    # 汇总
    all_recs = []
    for f in glob.glob(os.path.join(out_dir, "eval*.jsonl")):
        seen = {}
        for line in open(f, encoding="utf-8"):
            try:
                r = json.loads(line)
                seen[r["id"]] = r
            except Exception:
                pass
        all_recs.extend(seen.values())
    judged = [r for r in all_recs if r.get("judge_done")]

    def avg(k, pool):
        vals = [r[k] for r in pool if r.get(k) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    # 幻觉率分母剔除闲聊题(无 facts,无从校验)
    kn = [r for r in judged if not r.get("judge_chitchat")]
    summary = {
        "total": len(all_recs),
        "judged": len(judged),
        "chitchat_skipped": len(judged) - len(kn),
        "avg_faithfulness": avg("judge_faithfulness", kn),
        "avg_context_precision": avg("judge_context_precision", judged),
        "avg_context_recall": avg("judge_context_recall", judged),
        "avg_answer_relevancy": avg("judge_answer_relevancy", judged),
        "hallucination_count": sum(1 for r in kn if r.get("has_hallucination")),
        "hallucination_rate": round(sum(1 for r in kn if r.get("has_hallucination"))
                                    / len(kn), 4) if kn else None,
    }
    out = os.path.join(out_dir, "judge_summary_50x5.json")
    json.dump(summary, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print(f"== saved -> {out}")


if __name__ == "__main__":
    main()
