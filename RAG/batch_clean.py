# -*- coding: utf-8 -*-
"""
批量清洗脚本:用 MinerU (pipeline 后端) 把源目录下所有 PDF 解析成干净产物,
镜像落到 清洗产物目录下。

用法:
    python batch_clean.py "<源目录>"    # 清洗指定目录
    python batch_clean.py               # 默认跑整个资料库

断点续跑:已清洗完成的(产物齐全)自动跳过。
大 PDF(页数 > BIG_PAGE_THRESHOLD)自动拆分成小 PDF 逐段处理,避免 MinerU 卡死。
日志 _batch_clean.log 为追加写,跨次累计,每条带 [HH:MM:SS] 时间戳。
"""
import os, re, sys, time, json, shutil, subprocess
import logging

# 抑制 pypdf 的 WARNING 级别日志(如 "Ignoring wrong pointing object X 0 (offset 0)")
# 这是 PDF 内部交叉引用表(xref)条目损坏的无害警告,pypdf 会自动跳过损坏对象,不影响实际解析
logging.getLogger("pypdf").setLevel(logging.ERROR)

# 让 mineru 的 httpx 绕过系统代理对 localhost 的拦截
os.environ.setdefault('NO_PROXY', '127.0.0.1,localhost')

# ==================== GPU 限速配置 ====================
# MinerU pipeline 后端根据显存自动选 batch_ratio(批大小系数):
#   >=8GB -> ratio=4 | >=6GB -> ratio=2 | <6GB -> ratio=1
# ratio 直接乘到各模型批大小上(公式识别 ratio*16, OCR检测 ratio*8),
# ratio 越大批量越大 -> GPU 占用越高。
#
# MINERU_VIRTUAL_VRAM_SIZE: 欺骗 MinerU 让它以为显存只有这么多 GB,
#   从而选更小的 batch_ratio,降低 GPU 占用。get_vram() 优先读此变量。
#   你的 RTX 4060 Laptop 真实 8GB -> 默认 ratio=4(GPU 近满载);
#   设为 7  -> ratio=2(约 50% 占用,推荐起始值);
#   设为 5  -> ratio=1(最低占用,清洗速度约慢 40-60%)。
#   调参方法:跑起来后看 nvidia-smi,微调此值直到占用满意。
os.environ.setdefault('MINERU_VIRTUAL_VRAM_SIZE', '7')

# MINERU_PROCESSING_WINDOW_SIZE: 每个处理窗口加载多少页(默认 64)。
#   减小可降低峰值显存、平滑 GPU 占用曲线(不影响 batch_ratio)。
#   8GB 显存建议 32;如果仍然爆显存(OOM)可降到 16。
os.environ.setdefault('MINERU_PROCESSING_WINDOW_SIZE', '32')
os.environ.setdefault('no_proxy', '127.0.0.1,localhost')

MINERU   = r"C:\project3\.venv_mineru\Scripts\mineru.exe"
SRC_ROOT = r"C:\project3\180-半导体设备相关资料！"
OUT_ROOT = r"D:\清洗文件"
LOG      = os.path.join(OUT_ROOT, "_batch_clean.log")
SPLIT_TEMP = os.path.join(OUT_ROOT, "_split_temp")   # 大 PDF 拆分临时目录

DEFAULT_SRC = SRC_ROOT                  # 默认清洗整个资料库(不传参时)

# ---- 大 PDF 自动拆分参数 ----
BIG_PAGE_THRESHOLD = 500    # 超过此页数自动拆分
CHUNK_PAGES        = 500    # 每段页数
PER_PDF_TIMEOUT    = 1800   # 普通 PDF  MinerU 超时(秒)=30min
PER_PART_TIMEOUT   = 1800   # 拆分后每段 MinerU 超时(秒)=30min

# 手动跳过清单:MinerU 卡住的 PDF,待单独处理
# 注:"semi标准合集"(7923页)已实现自动拆分;确认拆分逻辑稳定后可从此清单移除
SKIP_KEYWORDS = [
    "光子集成电路封装", "67374",
    "Datacon 8800 platform2_Maintenance",
    "ECTC2020ThermoCompressionBondingforlargedies",
    "热压粘合的启用要求66801", "10 纳米技术节点",
    "semi标准合集",
]

# ==================== 收集 PDF ====================

src_dir = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else DEFAULT_SRC

pdfs = []
for dp, dn, fn in os.walk(src_dir):
    for f in fn:
        if f.lower().endswith(".pdf"):
            pdfs.append(os.path.join(dp, f))
pdfs.sort()

os.makedirs(OUT_ROOT, exist_ok=True)
logf = open(LOG, "a", encoding="utf-8")


# ==================== 基础工具函数 ====================

def log(s):
    """带时间戳的日志,同时输出到控制台和日志文件。"""
    ts = time.strftime('%H:%M:%S')
    line = "[" + ts + "] " + s
    print(line, flush=True)
    logf.write(line + "\n"); logf.flush()


def is_done(out_parent, stem):
    """判定 PDF 是否清洗完成:md 与 content_list.json 都在且 json 可解析。"""
    auto = os.path.join(out_parent, stem, "auto")
    md = os.path.join(auto, stem + ".md")
    cl = os.path.join(auto, stem + "_content_list.json")
    if not (os.path.exists(md) and os.path.exists(cl)):
        return False
    try:
        with open(cl, "r", encoding="utf-8") as f:
            json.load(f)
        return True
    except Exception:
        return False


def get_pdf_pages(pdf_path):
    """用 pypdf 获取 PDF 页数;失败返回 0(回退为不分段)。"""
    try:
        from pypdf import PdfReader
        return len(PdfReader(pdf_path).pages)
    except Exception as e:
        log("     [警告] 读取页数失败({}), 不分段处理".format(e))
        return 0


def split_pdf(pdf_path, out_dir, chunk=CHUNK_PAGES):
    """把大 PDF 拆成多个小 PDF。

    返回 list[(part_path, start_page_1based, end_page_1based)]。
    文件名格式: <stem>_p<start>-<end>.pdf  (如 semi标准合集_p1-500.pdf)
    产物路径自然分开,每段作为独立 source 处理。
    """
    from pypdf import PdfReader, PdfWriter
    reader = PdfReader(pdf_path)
    total = len(reader.pages)
    stem = os.path.splitext(os.path.basename(pdf_path))[0]
    os.makedirs(out_dir, exist_ok=True)
    parts = []
    for start in range(0, total, chunk):
        end = min(start + chunk, total)
        writer = PdfWriter()
        for i in range(start, end):
            writer.add_page(reader.pages[i])
        part_name = "{}_p{}-{}.pdf".format(stem, start + 1, end)
        part_path = os.path.join(out_dir, part_name)
        with open(part_path, "wb") as f:
            writer.write(f)
        parts.append((part_path, start + 1, end))
    return parts


def get_existing_parts(split_dir):
    """复用已存在的拆分文件(断点续跑)。从文件名解析页码范围。"""
    if not os.path.isdir(split_dir):
        return None
    existing = sorted(f for f in os.listdir(split_dir) if f.lower().endswith(".pdf"))
    if not existing:
        return None
    parts = []
    for f in existing:
        part_path = os.path.join(split_dir, f)
        m = re.search(r'_p(\d+)-(\d+)\.pdf$', f, re.IGNORECASE)
        ps, pe = (int(m.group(1)), int(m.group(2))) if m else (0, 0)
        parts.append((part_path, ps, pe))
    return parts


def run_mineru(cmd, timeout):
    """运行 MinerU 命令;超时后用 taskkill /T /F 杀整个进程树(含孙进程)。

    MinerU pipeline 模式会启动本地 API 服务(孙进程),
    subprocess.run(timeout) 在 Windows 上杀不死孙进程,
    必须用 taskkill /T /F 强制终止整个进程树。

    返回 (returncode_or_None, stdout, stderr);None=超时。
    """
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace",
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        return proc.returncode, stdout, stderr
    except subprocess.TimeoutExpired:
        log("     [超时] 正在强制终止进程树 (PID={})...".format(proc.pid))
        try:
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                capture_output=True, timeout=30,
            )
        except Exception:
            pass
        try:
            stdout, stderr = proc.communicate(timeout=10)
        except Exception:
            stdout, stderr = "", ""
        return None, stdout, stderr


# ==================== 主流程 ====================

log("\n" + "=" * 60)
log("[新一次运行] " + time.strftime('%Y-%m-%d %H:%M:%S'))
log("源目录: " + src_dir)
log("PDF 数: " + str(len(pdfs)))
log("大PDF阈值: >{}页自动拆分, 每段{}页".format(BIG_PAGE_THRESHOLD, CHUNK_PAGES))
log("开始时间: " + time.strftime('%Y-%m-%d %H:%M:%S') + "\n")

results = []
for i, p in enumerate(pdfs, 1):
    stem = os.path.splitext(os.path.basename(p))[0]
    # rel 必须以本次实际处理的 src_dir 为基准(支持命令行传入自定义源目录);
    # 若固定用 SRC_ROOT 常量,自定义目录会跨盘符抛 ValueError 或落到镜像结构之外
    rel  = os.path.relpath(os.path.dirname(p), src_dir)
    out_parent = os.path.join(OUT_ROOT, rel)
    size_mib = round(os.path.getsize(p) / 1024 / 1024, 1)

    # ---- 手动跳过 ----
    if any(kw in stem for kw in SKIP_KEYWORDS):
        log("[{}/{}] SKIP(手动跳过): {}".format(i, len(pdfs), stem))
        results.append((stem, "SKIP", "manual"))
        continue

    # ---- 检测页数 ----
    pages = get_pdf_pages(p)

    # ---- 大 PDF 自动拆分处理 ----
    if pages > BIG_PAGE_THRESHOLD:
        log("[{}/{}] BIG PDF ({}页, {}MiB), 自动拆分: {}".format(
            i, len(pdfs), pages, size_mib, stem))

        split_dir = os.path.join(SPLIT_TEMP, stem)

        # 复用已有拆分文件(断点续跑)或重新拆分
        parts = get_existing_parts(split_dir)
        if parts:
            log("     复用已有拆分文件({}段)".format(len(parts)))
        else:
            t_split = time.time()
            parts = split_pdf(p, split_dir, CHUNK_PAGES)
            dt_split = time.time() - t_split
            log("     拆分完成: {}段, 耗时{}s".format(len(parts), round(dt_split)))

        os.makedirs(out_parent, exist_ok=True)
        all_ok = True
        for j, (part_path, pstart, pend) in enumerate(parts, 1):
            part_stem = os.path.splitext(os.path.basename(part_path))[0]
            part_size = round(os.path.getsize(part_path) / 1024 / 1024, 1)

            if is_done(out_parent, part_stem):
                log("     [{}/{}] SKIP(已清洗): {}".format(j, len(parts), part_stem))
                continue

            cmd = [MINERU, "-p", part_path, "-o", out_parent, "-b", "pipeline"]
            page_info = " p{}-{}".format(pstart, pend) if pstart else ""
            log("     [{}/{}] RUN: {}{}  ({}MiB)".format(
                j, len(parts), part_stem, page_info, part_size))

            t0 = time.time()
            rc, stdout, stderr = run_mineru(cmd, PER_PART_TIMEOUT)
            dt = time.time() - t0

            if rc is None:
                log("     -> FAIL  超时(>{}s)".format(PER_PART_TIMEOUT))
                all_ok = False
                results.append((part_stem, "FAIL", "timeout"))
                continue

            ok = is_done(out_parent, part_stem)
            log("     -> {}  {}s".format("OK" if ok else "FAIL", round(dt)))
            if not ok:
                log("     stderr: " + (stderr or '')[-400:])
                all_ok = False
            results.append((part_stem, "OK" if ok else "FAIL", str(round(dt)) + "s"))

        # 所有段完成 -> 清理临时拆分文件
        if all_ok:
            try:
                shutil.rmtree(split_dir)
                log("     [清理] 全部段完成, 已删除临时拆分文件")
            except Exception as e:
                log("     [清理] 删除临时文件失败: {}".format(e))
        else:
            log("     [提示] 部分段未完成, 保留临时文件供下次续跑")
        continue

    # ---- 普通 PDF 正常处理 ----
    if is_done(out_parent, stem):
        log("[{}/{}] SKIP(已清洗): {}".format(i, len(pdfs), stem))
        results.append((stem, "SKIP", "-"))
        continue

    os.makedirs(out_parent, exist_ok=True)
    cmd = [MINERU, "-p", p, "-o", out_parent, "-b", "pipeline"]
    if pages > 0:
        log("[{}/{}] RUN: {}  ({}页, {}MiB)".format(i, len(pdfs), stem, pages, size_mib))
    else:
        log("[{}/{}] RUN: {}  ({}MiB)".format(i, len(pdfs), stem, size_mib))

    t0 = time.time()
    rc, stdout, stderr = run_mineru(cmd, PER_PDF_TIMEOUT)
    dt = time.time() - t0

    if rc is None:
        log("     -> FAIL  超时(>{}s)".format(PER_PDF_TIMEOUT))
        results.append((stem, "FAIL", "timeout"))
        continue

    ok = is_done(out_parent, stem)
    log("     -> {}  {}s".format("OK" if ok else "FAIL", round(dt)))
    if not ok:
        log("     stderr: " + (stderr or '')[-600:])
    results.append((stem, "OK" if ok else "FAIL", str(round(dt)) + "s"))

# ==================== 汇总 ====================
log("\n=== 汇总 ===")
for stem, status, extra in results:
    log("  " + status.ljust(4) + " " + extra.rjust(8) + "  " + stem)
ok   = sum(1 for _, s, _ in results if s == "OK")
skip = sum(1 for _, s, _ in results if s == "SKIP")
fail = sum(1 for _, s, _ in results if s == "FAIL")
log("\nOK={}  SKIP={}  FAIL={}  / 共 {}".format(ok, skip, fail, len(results)))
log("结束时间: " + time.strftime('%Y-%m-%d %H:%M:%S'))
logf.close()
