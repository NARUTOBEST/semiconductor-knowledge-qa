# -*- coding: utf-8 -*-
"""PPTX 统一清洗脚本。

两阶段:
  1. 把资料库中的老格式 ``.ppt`` 用本机 PowerPoint(COM 自动化)批量另存为 ``.pptx``,
     落到 ``D:\\清洗文件\\pptx\\_converted\\<镜像路径>\\<stem>.pptx``(不污染源资料库);
     已存在则跳过。保真度高于 LibreOffice(中文字体、SmartArt、图表)。
  2. 用 python-pptx 解析全部 ``.pptx``(原始 + 转换后),按幻灯片抽取标题/正文/表格/
     图片,产出与 docx/pdf 同款的 ``<stem>_content_list.json``,供共享 chunker 复用。

设计要点(基于对 18 个 pptx 的实测):
  - 幻灯片本身是天然切块边界:标题作 text_level=1 标题,正文按文本框聚合,page_idx=页码。
  - 信息大量在图里(1420 张图,有 87 页仅 12 字的纯图册):直接导出 ppt/media 原始位图
    (无损)到 auto/images/,md5 去重 + 装饰小图过滤,入库阶段再做 VLM 描述。
  - 表格 24 个、chart 0 个:表格转 <table> HTML;无图表解析负担。
  - 页脚噪音(页码/版权行/厂商 URL/品牌标语)跨页高频重复,按文档频率(DF)自动剥离。
  - SmartArt 仅 2 篇含:从 ppt/diagrams/data*.xml 兜底抽 <a:t> 文字挂回所在幻灯片。

产物契约(对齐 RAG/pdf/chunker.py 实际读取字段):
  text:  {type, text, text_level?, page_idx}
  table: {type, table_body(html), table_caption:[], table_footnote:[], page_idx}
  image: {type, img_path(相对auto), image_caption:[], image_footnote:[], page_idx}

用法:
    python -m RAG.pptx.clean                 # 转换 .ppt + 清洗全部
    python -m RAG.pptx.clean --no-convert    # 只清洗已有 .pptx(不调 PowerPoint)
    python -m RAG.pptx.clean <文件或目录>     # 仅处理指定目标(自动判断是否需转换)
"""
import os
import sys
import re
import json
import time
import hashlib

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# 本目录名是 pptx,会遮蔽第三方 python-pptx:导入第三方前把 RAG 移出 sys.path。
_RAG = os.path.dirname(_HERE)
_PROJECT = os.path.dirname(_RAG)
for _p in (_PROJECT, os.path.join(_PROJECT, "config")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
sys.path = [p for p in sys.path if os.path.abspath(p) != _RAG]

import config as C  # noqa: E402
from pptx import Presentation  # noqa: E402
from pptx.util import Emu  # noqa: E402
from pptx.enum.shapes import MSO_SHAPE_TYPE  # noqa: E402

SRC_ROOT = C.SRC_ROOT
OUT_ROOT = r"D:\清洗文件\pptx"
CONVERTED_ROOT = os.path.join(OUT_ROOT, "_converted")  # .ppt -> .pptx 暂存
_LOG = os.path.join(OUT_ROOT, "_pptx_clean.log")

# ==================== 日志 ====================

_log_fh = None


def log(msg, echo=True):
    global _log_fh
    line = "[" + time.strftime("%H:%M:%S") + "] " + msg
    if echo:
        print(line, flush=True)
    try:
        if _log_fh is None:
            os.makedirs(OUT_ROOT, exist_ok=True)
            _log_fh = open(_LOG, "a", encoding="utf-8")
        _log_fh.write(line + "\n")
        _log_fh.flush()
    except Exception:
        pass


# ==================== 图片工具(同 docx 装饰图过滤) ====================

_EXT_BY_CT = {
    "image/png": ".png", "image/jpeg": ".jpg", "image/jpg": ".jpg",
    "image/gif": ".gif", "image/bmp": ".bmp", "image/webp": ".webp",
    "image/tiff": ".tif", "image/x-emf": ".emf", "image/x-wmf": ".wmf",
}
DECOR_THIN_EDGE = 10
DECOR_SPRITE_DIM = 32


def _image_size(blob):
    try:
        if blob[:8] == b"\x89PNG\r\n\x1a\n":
            return int.from_bytes(blob[16:20], "big"), int.from_bytes(blob[20:24], "big")
        if blob[:6] in (b"GIF87a", b"GIF89a"):
            return int.from_bytes(blob[6:8], "little"), int.from_bytes(blob[8:10], "little")
        if blob[:2] == b"BM":
            return int.from_bytes(blob[18:22], "little"), int.from_bytes(blob[22:26], "little")
        if blob[:2] == b"\xff\xd8":
            i = 2
            while i < len(blob):
                if blob[i] != 0xFF:
                    i += 1
                    continue
                marker = blob[i + 1]
                while marker == 0xFF:
                    i += 1
                    marker = blob[i + 1]
                if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
                    return (int.from_bytes(blob[i + 7:i + 9], "big"),
                            int.from_bytes(blob[i + 5:i + 7], "big"))
                seg = int.from_bytes(blob[i + 2:i + 4], "big")
                i += 2 + seg
    except (IndexError, ValueError):
        return None
    return None


def _is_decorative(blob):
    size = _image_size(blob)
    if size is None:
        return False
    w, h = size
    return min(w, h) < DECOR_THIN_EDGE or (w < DECOR_SPRITE_DIM and h < DECOR_SPRITE_DIM)


# ==================== .ppt -> .pptx (PowerPoint COM) ====================

_PP_OPEN_READONLY = -1    # msoTrue
_PP_OPEN_UNTITLED = 0     # msoFalse
_PP_OPEN_NOWINDOW = -1    # msoTrue? WithWindow=msoFalse=0 actually; use 0
_PP_SAVEAS_PPTX = 11      # ppSaveAsOpenXMLPresentation
_PP_ALERTS_NONE = 1


def _com_stem(fname):
    """剥扩展名,兼容 'xxx.pptx.ppt' 双重后缀,并去掉 Windows 会吞掉的尾点/空格。"""
    stem = os.path.splitext(fname)[0]
    while stem.lower().endswith((".pptx", ".ppt")):
        stem = os.path.splitext(stem)[0]
    return stem.rstrip(". ")


# COM 自动化常见错误码
_RPC_E_CALL_REJECTED = -2147418111   # PowerPoint 忙,拒接呼叫(可重试)
_RPC_E_SERVERBUSY = -2147418110


def _com_retry(fn, *args, tries=6, delay=2.0, **kw):
    """调用 PowerPoint COM;遇 CALL_REJECTED/SERVERBUSY 退避重试。"""
    import time as _t
    last = None
    for k in range(tries):
        try:
            return fn(*args, **kw)
        except Exception as e:
            last = e
            code = getattr(e, "hresult", None) or getattr(e, "winerror", None)
            if code in (_RPC_E_CALL_REJECTED, _RPC_E_SERVERBUSY) and k < tries - 1:
                _t.sleep(delay * (k + 1))
                continue
            raise
    raise last


# ==================== LibreOffice 兜底 ====================

# 安装后 soffice 的常见位置;找不到时返回 None(跳过兜底)
_SOFFICE_CANDIDATES = (
    r"C:\Program Files\LibreOffice\program\soffice.exe",
    r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
)


def _soffice_bin():
    """定位 LibreOffice soffice.exe;PATH 中或常见安装目录,找不到返回 None。"""
    import shutil
    p = shutil.which("soffice")
    if p:
        return p
    for c in _SOFFICE_CANDIDATES:
        if os.path.exists(c):
            return c
    return None


def _lo_convert(src, out_dir, stem, timeout=240):
    """用 LibreOffice headless 把单个 .ppt 转成 <out_dir>/<stem>.pptx。

    PowerPoint 拒绝打开的文件(损坏/受保护视图)常被 LibreOffice 容错解析成功。
    soffice 按原文件名生成输出(``x.pptx.ppt`` -> ``x.pptx.pptx``),故转换后
    统一重命名为目标 <stem>.pptx。成功返回 True。
    """
    import subprocess
    import tempfile
    soffice = _soffice_bin()
    if not soffice:
        return False
    # 独立 user profile,避免与任何已运行的 LibreOffice 实例锁冲突
    prof = os.path.join(tempfile.gettempdir(), "lo_pptx_" + stem[:12].replace(" ", "_"))
    prof_uri = "file:///" + prof.replace("\\", "/")
    cmd = [soffice, "-env:UserInstallation=" + prof_uri,
           "--headless", "--norestore", "--convert-to", "pptx",
           "--outdir", out_dir, src]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=timeout)
    except Exception:
        return False
    target = os.path.join(out_dir, stem + ".pptx")
    if os.path.exists(target):
        return True
    # soffice 输出名可能是 <原名去最后一个扩展名>.pptx,寻找 out_dir 下最新的 .pptx 改名
    want = os.path.splitext(os.path.basename(src))[0] + ".pptx"
    produced = os.path.join(out_dir, want)
    if os.path.exists(produced) and produced != target:
        try:
            os.replace(produced, target)
            return True
        except OSError:
            return False
    return False


def convert_all_ppt():
    """把 SRC_ROOT 下所有 .ppt 转成 .pptx 到 CONVERTED_ROOT(镜像层级),返回转换数。"""
    ppt_files = []
    for dp, _dn, fn in os.walk(SRC_ROOT):
        for f in fn:
            if f.lower().endswith(".ppt") and not f.startswith("~$"):
                ppt_files.append(os.path.join(dp, f))
    ppt_files.sort()
    if not ppt_files:
        return 0

    import win32com.client  # 延迟导入:--no-convert 时不需要
    app = None
    converted = lo_fallback = failed = skipped = 0

    def start_app():
        # DispatchEx 创建独立新进程,避免复用已被前次失败污染/残留的 PowerPoint 实例
        # (Dispatch 附着到运行中的实例时,实例异常会导致大批文件连环"打不开")。
        a = win32com.client.DispatchEx("PowerPoint.Application")
        try:
            a.DisplayAlerts = _PP_ALERTS_NONE
        except Exception:
            pass
        return a

    def kill_app(a):
        if a is None:
            return
        for _ in range(2):
            try:
                a.Quit()
                break
            except Exception:
                import time as _t
                _t.sleep(1)

    try:
        app = start_app()

        for i, src in enumerate(ppt_files, 1):
            rel = os.path.relpath(os.path.dirname(src), SRC_ROOT)
            stem = _com_stem(os.path.basename(src))
            out_dir = os.path.join(CONVERTED_ROOT, rel)
            out_path = os.path.join(out_dir, stem + ".pptx")
            if os.path.exists(out_path):
                skipped += 1
                continue
            os.makedirs(out_dir, exist_ok=True)
            try:
                try:
                    pres = _com_retry(
                        app.Presentations.Open,
                        src, ReadOnly=_PP_OPEN_READONLY,
                        Untitled=_PP_OPEN_UNTITLED, WithWindow=0)
                except Exception:
                    # 实例可能已僵死:重启 PowerPoint 再试一次
                    log("           [重启 PowerPoint 实例] %s" % stem[:40])
                    kill_app(app)
                    app = start_app()
                    pres = app.Presentations.Open(
                        src, ReadOnly=_PP_OPEN_READONLY,
                        Untitled=_PP_OPEN_UNTITLED, WithWindow=0)
                _com_retry(pres.SaveAs, out_path, _PP_SAVEAS_PPTX)
                _com_retry(pres.Close)
                converted += 1
                log("[转换 %d/%d] OK %s" % (i, len(ppt_files), stem[:50]))
            except Exception as e:
                # PowerPoint 拒绝/失败:用 LibreOffice headless 兜底转换
                if _lo_convert(src, out_dir, stem):
                    lo_fallback += 1
                    log("[转换 %d/%d] OK(LO兜底) %s" % (i, len(ppt_files), stem[:50]))
                else:
                    failed += 1
                    log("[转换 %d/%d] FAIL %s -> %s"
                        % (i, len(ppt_files), stem[:50], str(e)[:100]))
    finally:
        kill_app(app)

    log("PPT 转换: PowerPoint %d + LibreOffice兜底 %d, 跳过(已存在) %d, 失败 %d / 共 %d"
        % (converted, lo_fallback, skipped, failed, len(ppt_files)))
    return converted


# ==================== pptx -> content_list ====================

_FOOTER_COPYRIGHT = re.compile(r"(©|copyright|www\.|https?://|all rights reserved)", re.I)
_PUNCT_ONLY = re.compile(r"^[\s\W_]+$", re.UNICODE)


def _norm(text):
    return re.sub(r"\s+", " ", (text or "").strip())


def _is_page_number(text):
    t = text.strip()
    return len(t) <= 4 and (t.isdigit() or bool(re.fullmatch(r"\d+\s*/\s*\d+", t)))


def _table_html(table):
    out = ["<table>"]
    for row in table.rows:
        out.append("<tr>")
        for cell in row.cells:
            txt = _escape_html(_norm(cell.text))
            out.append("<td>%s</td>" % txt)
        out.append("</tr>")
    out.append("</table>")
    return "".join(out)


def _escape_html(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _smartart_texts(shape):
    """从 SmartArt 关系部件(data*.xml)兜底抽文字;失败返回 []。"""
    texts = []
    try:
        el = shape._element
        nsmap = {
            "dgm": "http://schemas.openxmlformats.org/drawingml/2006/diagram",
            "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
            "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
        }
        relids = el.findall(".//dgm:relIds", nsmap)
        if not relids:
            return texts
        for rid_el in relids:
            for attr in ("r:dm", "r:lo", "r:qs", "r:cs"):
                rid = rid_el.get("{%s}%s" % (nsmap["r"], attr.split(":")[1]))
                if not rid:
                    continue
                try:
                    part = shape.part.related_parts[rid]
                except (KeyError, AttributeError):
                    continue
                for t in part.blob.decode("utf-8", "ignore").splitlines():
                    pass  # 占位,真正解析在下面
                import re as _re
                for m in _re.findall(rb"<a:t[^>]*>(.*?)</a:t>", part.blob, _re.S):
                    s = m.decode("utf-8", "ignore").strip()
                    if s:
                        texts.append(s)
    except Exception:
        return texts
    return texts


def _walk_shapes(shapes, page_idx, ctx, out):
    """递归遍历一张幻灯片的 shapes,把 text/table/image 项追加到 out。"""
    for shape in shapes:
        try:
            # 组合
            if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
                _walk_shapes(shape.shapes, page_idx, ctx, out)
                continue
            # 图片
            if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                try:
                    img = shape.image
                    blob = img.blob
                except Exception:
                    blob = None
                if blob:
                    ctx["img_count"] += 1
                    digest = hashlib.md5(blob).hexdigest()[:12]
                    if digest not in ctx["emitted"] and digest not in ctx["skipped"]:
                        if _is_decorative(blob):
                            ctx["skipped"].add(digest)
                        else:
                            ext = _EXT_BY_CT.get(getattr(img, "content_type", ""),
                                                 "." + (img.ext or "png"))
                            fname = digest + ext
                            if digest not in ctx["written"]:
                                os.makedirs(ctx["images_dir"], exist_ok=True)
                                with open(os.path.join(ctx["images_dir"], fname), "wb") as f:
                                    f.write(blob)
                                ctx["written"].add(digest)
                            ctx["emitted"].add(digest)
                            out.append({
                                "type": "image",
                                "img_path": "images/" + fname,
                                "image_caption": [],
                                "image_footnote": [],
                                "page_idx": page_idx,
                            })
                continue
            # 表格
            if getattr(shape, "has_table", False):
                html = _table_html(shape.table)
                cells = re.findall(r"<td[^>]*>(.*?)</td>", html, re.S)
                if cells and any(c.strip() for c in cells):
                    out.append({
                        "type": "table",
                        "table_body": html,
                        "table_caption": [],
                        "table_footnote": [],
                        "page_idx": page_idx,
                    })
                continue
            # 文本框
            if getattr(shape, "has_text_frame", False):
                raw = shape.text_frame.text
                text = _norm(raw)
                if text:
                    ctx["texts"].append(text)
                continue
            # SmartArt / GraphicFrame:兜底抽文字
            sm = _smartart_texts(shape)
            for t in sm:
                t = _norm(t)
                if t:
                    ctx["texts"].append(t)
        except Exception as e:
            ctx["errors"].append("shape: " + str(e)[:80])


def _title_of(slide):
    """取幻灯片标题;无 title placeholder 时回退到页面最上方的短文本。"""
    try:
        if slide.shapes.title and slide.shapes.title.text.strip():
            return _norm(slide.shapes.title.text)
    except Exception:
        pass
    best = None
    try:
        for shape in slide.shapes:
            if not getattr(shape, "has_text_frame", False):
                continue
            txt = _norm(shape.text_frame.text)
            if not txt or len(txt) > 60:
                continue
            try:
                top = shape.top if shape.top is not None else 10**9
            except Exception:
                top = 10**9
            if best is None or top < best[0]:
                best = (top, txt)
    except Exception:
        pass
    return best[1] if best else None


def build_content_list(pptx_path, auto_dir):
    prs = Presentation(pptx_path)
    n = len(prs.slides)
    # DF 阈值:跨 >=max(3, 40%页) 出现的文本视为页脚/样板
    df_threshold = max(3, int(n * 0.4))
    images_dir = os.path.join(auto_dir, "images")
    ctx = {"images_dir": images_dir, "written": set(), "emitted": set(),
           "skipped": set(), "texts": [], "img_count": 0, "errors": []}

    # 第一遍:收集每页文本,计算文档频率
    page_texts = []      # 每页 list[str]
    page_items = []      # 每页 list[dict](table/image 顺序项)
    title_counts = {}
    for idx, slide in enumerate(prs.slides, 1):
        local = []
        ctx["texts"] = []
        _walk_shapes(slide.shapes, idx, ctx, local)
        texts = ctx["texts"]
        title = _title_of(slide)
        if title:
            title_counts[title] = title_counts.get(title, 0) + 1
        # 拆分文本框到独立字符串用于 DF
        page_texts.append((title, texts))
        page_items.append(local)

    # 统计非标题文本的跨页频次
    freq = {}
    for _title, texts in page_texts:
        for t in set(texts):
            freq[t] = freq.get(t, 0) + 1

    def _is_noise(t):
        if _is_page_number(t):
            return True
        if len(t) <= 80 and _FOOTER_COPYRIGHT.search(t):
            return True
        if not t or _PUNCT_ONLY.match(t):
            return True
        if freq.get(t, 0) >= df_threshold and len(t) <= 120:
            return True
        return False

    # 第二遍:组装 content_list
    items = []
    for idx, ((title, texts), local) in enumerate(zip(page_texts, page_items), 1):
        # 标题:跨页重复的"标题"(模板母版文字)当噪音去掉
        if title and title_counts.get(title, 0) < df_threshold:
            items.append({"type": "text", "text": title,
                          "text_level": 1, "page_idx": idx})
        # 正文文本框(合并同页非噪音文本为一个 text 项,保持幻灯片聚合)
        body = [t for t in texts if not _is_noise(t) and t != title]
        if body:
            items.append({"type": "text", "text": "\n".join(body), "page_idx": idx})
        # 表格/图片项
        items.extend(local)

    return items, ctx["errors"]


# ==================== 写出/断点/镜像 ====================

def write_content_list(auto_dir, stem, items):
    os.makedirs(auto_dir, exist_ok=True)
    path = os.path.join(auto_dir, stem + "_content_list.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    return path


def is_done(auto_dir, stem):
    p = os.path.join(auto_dir, stem + "_content_list.json")
    if not os.path.exists(p):
        return False
    try:
        with open(p, encoding="utf-8") as f:
            json.load(f)
        return True
    except Exception:
        return False


def mirror_out(pptx_path, root):
    rel = os.path.relpath(os.path.dirname(pptx_path), root)
    stem = _com_stem(os.path.basename(pptx_path))  # 去尾点/空格,与转换一致
    out_parent = os.path.join(OUT_ROOT, rel)
    auto_dir = os.path.join(out_parent, stem, "auto")
    return out_parent, auto_dir, stem


def clean_one(pptx_path, root):
    out_parent, auto_dir, stem = mirror_out(pptx_path, root)
    if is_done(auto_dir, stem):
        return True, "SKIP"
    try:
        items, errors = build_content_list(pptx_path, auto_dir)
        os.makedirs(out_parent, exist_ok=True)
        write_content_list(auto_dir, stem, items)
        n_txt = sum(1 for x in items if x["type"] == "text")
        n_img = sum(1 for x in items if x["type"] == "image")
        n_tbl = sum(1 for x in items if x["type"] == "table")
        msg = "OK(%d项: 文%d/图%d/表%d)" % (len(items), n_txt, n_img, n_tbl)
        if errors:
            msg += " warn=%d" % len(errors)
        return True, msg
    except Exception as e:
        return False, "FAIL: " + str(e)[:120]


# ==================== 收集 ====================

def collect_pptx(target=None):
    """返回 [(pptx_path, root_for_mirror)]。target 可空/文件/目录。"""
    if target:
        if os.path.isfile(target):
            root = CONVERTED_ROOT if os.path.abspath(target).startswith(
                os.path.abspath(CONVERTED_ROOT)) else SRC_ROOT
            return [(target, root)]
        scan_roots = [(target, SRC_ROOT)]
    else:
        scan_roots = [(SRC_ROOT, SRC_ROOT), (CONVERTED_ROOT, CONVERTED_ROOT)]
    out = []
    seen = set()
    for root, mirror_root in scan_roots:
        if not os.path.isdir(root):
            continue
        for dp, _dn, fn in os.walk(root):
            # 转换根与输出根都在 OUT_ROOT 下,清洗时不要扫到 auto/images 产物
            if os.sep + "auto" + os.sep in dp or dp.endswith(os.sep + "auto"):
                continue
            for f in fn:
                if f.lower().endswith(".pptx") and not f.startswith("~$"):
                    p = os.path.join(dp, f)
                    if p not in seen:
                        seen.add(p)
                        out.append((p, mirror_root))
    out.sort()
    return out


# ==================== 主流程 ====================

def clean_path(target=None, do_convert=True):
    log("=" * 60)
    log("[pptx 清洗] " + time.strftime("%Y-%m-%d %H:%M:%S"))
    if do_convert and not (target and os.path.isfile(target)
                           and target.lower().endswith(".pptx")):
        try:
            convert_all_ppt()
        except ImportError:
            log("[警告] 未安装 pywin32,跳过 .ppt 转换(仅处理已有 .pptx)")
        except Exception as e:
            log("[警告] .ppt 转换阶段异常: %s" % str(e)[:120])

    jobs = collect_pptx(target)
    log("待清洗 pptx: %d" % len(jobs))
    ok = skip = fail = 0
    for i, (path, root) in enumerate(jobs, 1):
        t0 = time.time()
        success, msg = clean_one(path, root)
        dt = time.time() - t0
        status = "SKIP" if msg == "SKIP" else ("OK" if success else "FAIL")
        if status == "OK":
            ok += 1
        elif status == "SKIP":
            skip += 1
        else:
            fail += 1
        log("[%d/%d] %s %s  %.1fs  %s" % (
            i, len(jobs), status.ljust(4), os.path.basename(path)[:48], dt,
            msg if status != "OK" else ""))
    log("-" * 60)
    log("OK=%d  SKIP=%d  FAIL=%d / 共 %d" % (ok, skip, fail, len(jobs)))


if __name__ == "__main__":
    args = sys.argv[1:]
    no_convert = "--no-convert" in args
    args = [a for a in args if a != "--no-convert"]
    target = args[0] if args else None
    clean_path(target, do_convert=not no_convert)
