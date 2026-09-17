# -*- coding: utf-8 -*-
"""补齐 Qdrant 服务器缺失文档:文本缺失 7 + 图像缺失(真有图)47,取并集逐个入库。

复用 RAG/pdf/ingest.py 的 process_pdf/checklist/md5map 逻辑,完成后更新断点。
"""
import os, sys, json

os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
os.environ.setdefault("no_proxy", "127.0.0.1,localhost")
_HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "RAG", "pdf")
_RAG = os.path.dirname(_HERE)
for _p in (_HERE, _RAG, os.path.join(os.path.dirname(_RAG), "config")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import config as C
C.QDRANT_URL = "http://127.0.0.1:6333"   # 强制服务器模式(须在 embed/ingest 加载前设置)
import ingest  # noqa: E402  (QDRANT_URL 已设置,走服务器模式)

mt = json.load(open("missing_text.json", encoding="utf-8"))
mi = json.load(open("missing_image_real.json", encoding="utf-8"))
todo = sorted(set(mt) | set(mi))
print(f"待补入库文档: {len(todo)} (文本缺 {len(mt)} / 图像缺 {len(mi)},去重后 {len(todo)})")

# stem -> (auto_dir, src_pdf)
index = {}
for dp, dn, fn in os.walk(C.CLEAN_ROOT):
    if os.path.basename(dp) != "auto":
        continue
    for f in fn:
        if f.endswith("_content_list.json") and "_v2" not in f:
            stem = f[: -len("_content_list.json")]
            if stem in todo:
                index[stem] = (dp, None)

done = ingest.load_ckpt()
md5map = ingest.load_md5map()
client = ingest.get_client()
ingest.ensure_collections(client)
ingest.TE = embed_te = __import__("embed").get_text_encoder()
ingest.IE = __import__("embed").get_image_encoder()

ok = fail = 0
for i, stem in enumerate(todo, 1):
    auto = index[stem][0]
    rel = os.path.relpath(os.path.dirname(auto), C.CLEAN_ROOT)
    src_pdf = os.path.join(C.SRC_ROOT, rel, stem + ".pdf")
    try:
        nt, ni = ingest.process_pdf(stem, auto, src_pdf, client, do_describe=True)
        done.add(stem)
        ingest.save_ckpt(done)
        cm, _src = ingest.content_md5(auto, stem, src_pdf)
        if cm:
            md5map[cm] = md5map.get(cm, stem)
            ingest.save_md5map(md5map)
        ok += 1
        print(f"[{i}/{len(todo)}] OK  {stem[:60]:60s} text={nt:4d} img={ni:3d}", flush=True)
    except Exception as e:
        fail += 1
        print(f"[{i}/{len(todo)}] FAIL {stem[:60]:60s} {str(e)[:150]}", flush=True)
print(f"补齐完成: 成功 {ok} / 失败 {fail}")
