# -*- coding: utf-8 -*-
"""术语表字符串替换(轮次19):翻译双路之后、召回之前,把查询里的中文术语
替换为库内英文术语,英文查询做别名归一(变体/缩写 -> 规范词)。
预编译单条 alternation 正则(最长优先),整串一次扫描,单查询 <1ms。

术语表 glossary.json 格式(build_glossary2.py 产出):
  {"en_terms": [{"en","tf","zh":[...],"en2":[...]}], "zh2en": {...}, "en2en": {...}}
文件缺失/损坏/未开开关时全部直通(pass-through),零伤害。
"""
import json
import os
import re
import threading

import config as C

_lock = threading.Lock()
_cache = {"mtime": 0.0, "path": None, "zh2en": {}, "en2en": {},
          "zh_re": None, "en_re": None}


def _glossary_path():
    return (getattr(C, "GLOSSARY_FILE", "") or
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "glossary.json"))


def _compile(mapping):
    if not mapping:
        return None
    # 最长优先 alternation,保证长术语先匹配(不被短键切碎);
    # IGNORECASE:英文查询里 BOX/Box/box 都归一(中文无大小写,不受影响)
    keys = sorted(mapping, key=len, reverse=True)
    # 纯 ASCII 键(英文别名)加词边界:防 TRIP 命中 strip 内部这类误伤
    pats = []
    for k in keys:
        esc = re.escape(k)
        if k.isascii():
            pats.append(r"\b" + esc + r"\b")
        else:
            pats.append(esc)
    return re.compile("|".join(pats), re.IGNORECASE)


def _load():
    """带 mtime 热更新:改 json 即生效,不用重启服务。"""
    path = _glossary_path()
    try:
        mtime = os.path.getmtime(path)
        if _cache["path"] == path and _cache["mtime"] == mtime:
            return
        data = json.load(open(path, encoding="utf-8"))
        # 三道质量护栏:
        # ① zh->en 目标过短(≤2 字符缩写如 SO/PR/PC)不入表:中文词被替换成
        #    两个字母等于把稠密语义也扔了,纯亏;≥3 字符(EUV/FAB)才值得换。
        # ② zh 键 <3 字不入表:单字与双字泛词(释放/晶圆/清洗/系统)会命中
        #    复合词内部(埋氧层->RLS埋氧LAYER、反而->CARS而),碎片化毁查询;
        #    真术语(埋氧层/兆声波/光刻胶)基本 ≥3 字。
        # ③ en2en 英文别名键在 _compile 加 \b 词边界,防 strip 内部 TRIP 误伤。
        zh2en = {k.strip(): v.strip()
                 for k, v in (data.get("zh2en") or {}).items()
                 if k and len(k.strip()) >= 3 and v
                 and len(v.strip()) >= 3}
        en2en = {k.strip().lower(): v.strip()
                 for k, v in (data.get("en2en") or {}).items()
                 if k and k.strip() and v
                 # ④ 目标是 ≤5 字母全大写缩写的不入表:批译把 release/wafer/
                 #    etching 等常见英文词记成了 RLS/WAF/ETCH 的别名,归一会把
                 #    翻译路/英文查询里的普通词逆向缩成行话,毁掉稠密语义
                 and not (v.strip().isupper() and v.strip().isalpha()
                          and len(v.strip()) <= 5)}
        with _lock:
            _cache.update(mtime=mtime, path=path, zh2en=zh2en, en2en=en2en,
                          zh_re=_compile(zh2en), en_re=_compile(en2en))
    except (OSError, ValueError):
        with _lock:
            _cache.update(mtime=0.0, path=path, zh2en={}, en2en={},
                          zh_re=None, en_re=None)


def apply(query):
    """术语替换入口:含中文的句做 zh2en 替换,纯英文句做 en2en 别名归一
    (大小写不敏感,替换为规范词)。无匹配原样返回。"""
    if not query or not getattr(C, "RETRIEVAL_GLOSSARY_REPLACE", False):
        return query
    _load()
    with _lock:
        is_ascii = all(ord(ch) < 128 for ch in query)
        if is_ascii:
            mapping, rx = _cache["en2en"], _cache["en_re"]
        else:
            mapping, rx = _cache["zh2en"], _cache["zh_re"]
    if rx is None:
        return query

    def _sub(m):
        k = m.group(0)
        v = mapping.get(k) or mapping.get(k.lower()) or k
        if v != k:
            # 与相邻 CJK 之间补空格,防中英粘连成伪词(工艺+参数->PROCPARAMETER)
            pre = " " if m.start() > 0 and ord(query[m.start() - 1]) > 127 else ""
            post = (" " if m.end() < len(query)
                    and ord(query[m.end()]) > 127 else "")
            v = pre + v + post
        return v

    return rx.sub(_sub, query)


def stats():
    """/health 观测:术语表规模。"""
    _load()
    with _lock:
        return {"zh": len(_cache["zh2en"]), "en_alias": len(_cache["en2en"]),
                "version_mtime": _cache["mtime"]}
