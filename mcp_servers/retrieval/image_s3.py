# -*- coding: utf-8 -*-
"""图片对象存储(火山引擎 TOS,S3 兼容)——路径↔对象 key↔时效签名链接。

检索链路(:chunks 结果)把 payload 里 ``D:\\清洗文件`` 下的图片**绝对路径**
转成 TOS 私有桶的预签名 HTTPS 链接,前端/视觉模型直接用 URL 取图,
部署虚拟机不必复制清洗图片目录。上传链路(项目根 _tos_upload.py)也复用本模块。

未配置 TOS_* 环境变量时 :func:`image_url` 原样返回本地绝对路径(向后兼容,
本地开发仍可读本地文件)。

配置见 config.py 的 TOS_*:
  TOS_ENDPOINT   例 tos-cn-beijing.volces.com
  TOS_REGION     例 cn-beijing
  TOS_BUCKET     桶名
  TOS_ACCESS_KEY / TOS_SECRET_KEY
  TOS_URL_EXPIRES        签名有效期(秒),默认 3600
  TOS_USE_PATH_STYLE     MinIO 设 1,TOS 用默认虚拟主机风格
  TOS_KEY_PREFIX         桶内对象 key 前缀(默认 qingxi)
  CLEAN_FILES_ROOT       图片文件根(payload 路径前缀),默认 D:\\清洗文件
"""
import os
import re
import sys
import threading

_HERE = os.path.dirname(os.path.abspath(__file__))                 # mcp_servers/retrieval
_PROJECT = os.path.dirname(os.path.dirname(_HERE))                 # project root(上溯 2 级)
_CONFIG = os.path.join(_PROJECT, "config")
if _CONFIG not in sys.path:
    sys.path.insert(0, _CONFIG)

import config as C  # noqa: E402

_client = None
_video_client = None
_cfg_resolved = None
_lock = threading.Lock()


def _make_client(endpoint, ak, sk, region, path_style):
    """构造一个 S3 兼容客户端(共用签名/代理绕过/校验和适配策略)。"""
    import boto3
    from urllib.parse import urlparse
    from botocore.client import Config
    if not endpoint.startswith("http"):
        endpoint = "https://" + endpoint
    # 国内存储绕过本机 HTTP 代理(Clash 等),否则可能被绕到国外导致 403/超时
    host = urlparse(endpoint).hostname
    if host:
        for var in ("NO_PROXY", "no_proxy"):
            cur = os.environ.get(var, "")
            if host not in cur:
                os.environ[var] = (cur + "," if cur else "") + host
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=ak,
        aws_secret_access_key=sk,
        region_name=region or None,
        config=Config(
            signature_version="s3v4",
            # boto3≥1.36 默认对 PUT 附带 STREAMING-UNSIGNED-PAYLOAD-TRAILER
            # (CRC32 尾块),OSS/MinIO 等非 AWS S3 不支持 → 未要求时不计算校验和
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
            # 火山 TOS 强制虚拟主机风格(bucket.endpoint),path 风格会报 InvalidPathAccess;
            # 自建 MinIO 才用 path(对应 *_USE_PATH_STYLE=1)
            s3={"addressing_style": "path" if path_style else "virtual"},
        ),
    )


def _storage_config():
    """返回真正带 TOS_*/VID_* 配置的 config 模块。

    顶部 ``import config as C`` 在某些 import 顺序下会被项目根的 config/
    目录(namespace 包)抢占(后端容器内 server 先于检索模块加载时就会),
    此时 C 上没有存储配置——按文件路径显式加载兜底;C 正常时原样返回,
    行为与原来完全一致。
    """
    global _cfg_resolved
    if _cfg_resolved is not None:
        return _cfg_resolved
    if hasattr(C, "TOS_BUCKET") and hasattr(C, "TOS_URL_EXPIRES"):
        _cfg_resolved = C
        return C
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_retrieval_storage_config", os.path.join(_CONFIG, "config.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _cfg_resolved = mod
    return mod


def get_s3():
    """懒加载图片存储客户端(TOS)。未配置返回 None。"""
    global _client
    if _client is not None:
        return _client
    cfg = _storage_config()
    if not cfg.TOS_ENABLED:
        return None
    with _lock:
        if _client is None:
            _client = _make_client(cfg.TOS_ENDPOINT, cfg.TOS_ACCESS_KEY,
                                   cfg.TOS_SECRET_KEY, cfg.TOS_REGION,
                                   cfg.TOS_USE_PATH_STYLE)
    return _client


def get_s3_video():
    """懒加载视频存储客户端(独立 S3 兼容存储,如阿里云 OSS)。

    图片与视频可分开存放:图片在 TOS(VID_* 未配置时视频也回落 TOS)。
    未配置任何存储返回 None(与 get_s3 一致,调用方回退本地路径)。
    """
    global _video_client
    if _video_client is not None:
        return _video_client
    cfg = _storage_config()
    if not getattr(cfg, "VID_ENABLED", False):
        return get_s3()                     # 未配 VID_*:视频与图片同存储
    with _lock:
        if _video_client is None:
            _video_client = _make_client(cfg.VID_ENDPOINT, cfg.VID_ACCESS_KEY,
                                         cfg.VID_SECRET_KEY, cfg.VID_REGION,
                                         cfg.VID_USE_PATH_STYLE)
    return _video_client


# 清洗目录下的模态子目录(磁盘重整后的层级)。CAD 为大写,其余小写。
_MODALITY_DIRS = {"pdf", "docx", "pptx", "xlsx", "mp4", "image", "cad"}


def _split_root(path):
    """把任意平台的绝对路径(含 Windows 盘符 ``D:\\...``、UNC、POSIX ``/...``)
    拆成 (drive_or_root, body)。用于跨 OS 判断“本进程是 Linux、但 payload 路径
    是 Windows 入库时写入”的情况 —— 此时 os.path.isabs/relpath 会误判。
    返回 None 表示不是可识别的绝对路径。
    """
    if not path:
        return None
    s = str(path).replace("\\", "/")          # 统一为正斜杠,两种分隔符通吃
    m = re.match(r"^([A-Za-z]:)(/.*)$", s)    # Windows 盘符  D:/...
    if m:
        return m.group(1).upper(), m.group(2).lstrip("/")
    if s.startswith("//"):                     # UNC  //server/share/...
        body = s[2:]
        root = "//" + body.split("/", 1)[0]
        return root, body.split("/", 1)[1] if "/" in body else ""
    if s.startswith("/"):                      # POSIX 绝对路径
        return "/", s.lstrip("/")
    return None


def _abs_to_key_with_prefix(abs_path, prefix):
    """abs_to_key 的参数化核心:按指定桶内前缀换算(图片 TOS / 视频 VID 各用各的前缀)。"""
    if not abs_path:
        return None
    p_parts = _split_root(abs_path)
    r_parts = _split_root(C.CLEAN_FILES_ROOT)
    # 盘符仅在两边都是 Windows 盘符路径时比较(大小写不敏感);POSIX/混合时不拦盘符
    if not p_parts or not r_parts:
        return None
    p_drive, p_body = p_parts
    r_drive, r_body = r_parts
    if len(p_drive) == 2 and len(r_drive) == 2:          # 都是 X: 盘符
        if p_drive != r_drive:
            return None
    p_toks = [t for t in p_body.split("/") if t not in ("", ".")]
    r_toks = [t for t in r_body.split("/") if t not in ("", ".")]
    if p_toks[:len(r_toks)] != r_toks:                   # 必须在清洗根之下
        return None
    rel = "/".join(p_toks[len(r_toks):])
    if not rel:
        return None
    first = rel.split("/", 1)[0].lower()
    if first not in _MODALITY_DIRS:
        rel = "pdf/" + rel          # 旧 PDF payload 缺模态段,补回 pdf/
    if first == "cad":
        # CAD 渲染图只存桶根(根级 CAD/ 对 qingxi/CAD 100% 同名同大小覆盖,
        # qingxi/CAD 已删除);页图等其余模态仍走 qingxi/ 前缀。
        return rel
    return (prefix + "/" + rel) if prefix else rel


def abs_to_key(abs_path):
    """D:\\清洗文件\\<rel>  ->  <TOS_KEY_PREFIX>/<rel with '/'>。

    不在 CLEAN_FILES_ROOT 下的路径返回 None(无法映射)。跨平台:retrieval 微服务在
    Linux 容器内运行,而 Qdrant payload 里的 image_path 多为 Windows 入库时写入的
    ``D:\\清洗文件\\...``,故用 :func:`_split_root` 统一按正斜杠换算,不依赖 os.path
    (Linux 的 posixpath 不识别盘符/反斜杠,会把整串当文件名)。

    历史遗留:清洗目录曾重整、加入 pdf/ 等模态子目录层;重整之前入库的旧 PDF 点,
    payload 里的 image_path 仍是旧根路径(缺模态段,如 D:\\清洗文件\\0002 光刻资料\\...),
    而文件与 TOS 对象实际在 pdf/ 下。这类路径首段不在模态目录内,补回 ``pdf/`` 段,
    使签名 key 与实际上传 key(qingxi/pdf/...)对齐。
    """
    return _abs_to_key_with_prefix(abs_path, C.TOS_KEY_PREFIX)


_basename_idx = None      # 图片文件名(hash.jpg) -> 对象 key;用于还原历史相对路径引用
_idx_lock = threading.Lock()


def _ensure_basename_index():
    """懒加载 ald_image 的 图片文件名->对象key 索引(滚动一次 Qdrant)。

    历史文本块的 image_paths 有一批是相对路径 ``images/<hash>.jpg``(早期实验入库,
    source_path 已失效,无法按目录还原);但图片文件名是内容指纹、全局唯一,且这些图
    实际上传后的对象 key 末尾就是 ``.../auto/images/<hash>.jpg``,故按文件名反查即可。
    Qdrant 不可用时返回空 dict(相对路径回退原路径,不影响绝对路径)。
    """
    global _basename_idx
    if _basename_idx is not None:
        return _basename_idx
    with _idx_lock:
        if _basename_idx is None:
            idx = {}
            try:
                from qdrant_client import QdrantClient
                qc = QdrantClient(url=C.QDRANT_URL, timeout=60) if C.QDRANT_URL \
                    else QdrantClient(path=C.QDRANT_PATH, timeout=60)
                off = None
                while True:
                    pts, off = qc.scroll(C.IMAGE_COLLECTION, limit=2000, offset=off,
                                         with_payload=True, with_vectors=False)
                    if not pts:
                        break
                    for p in pts:
                        ip = (p.payload or {}).get("image_path", "")
                        k = abs_to_key(ip) if ip else None
                        if k:
                            idx[os.path.basename(k)] = k
                    if off is None:
                        break
            except Exception:
                idx = {}                        # Qdrant 不可用:降级(仅绝对路径可签名)
            _basename_idx = idx
    return _basename_idx


def _sign_key(s3, key, bucket, inline=False, content_type=None):
    params = {"Bucket": bucket, "Key": key}
    if inline:
        # 显式内联:覆盖桶/对象上的强制下载元数据,浏览器直接预览
        params["ResponseContentDisposition"] = "inline"
    if content_type:
        # 覆盖对象存储的 Content-Type(存量对象多为 binary/octet-stream,
        # 浏览器不认会变下载;按扩展名补正确类型才能在线预览)
        params["ResponseContentType"] = content_type
    return s3.generate_presigned_url(
        "get_object",
        Params=params,
        ExpiresIn=C.TOS_URL_EXPIRES,
    )


# 按扩展名判用途:文档(inline 预览)/视频(独立桶)/浏览器 Content-Type
_DOC_EXTS = {".pdf", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls"}
_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".mpeg", ".mpg",
               ".m4v", ".wmv", ".flv", ".ts"}
_CONTENT_TYPES = {
    ".pdf": "application/pdf", ".png": "image/png", ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg", ".gif": "image/gif", ".webp": "image/webp",
    ".bmp": "image/bmp", ".mp4": "video/mp4", ".webm": "video/webm",
}


def doc_proxy_url(key):
    """文档对象 key -> 后端在线预览代理 URL(相对路径,HMAC 签名 + 时效)。

    TOS 桶开启强制下载,预签名里的 response-content-disposition=inline 会被
    无视(attachment;filename 能盖,inline 不能),直链点了只会落盘下载;
    代理端点由后端强制 inline 返回,浏览器才能在线预览。
    未配置 JWT_SECRET 的进程(如 GPU 上的检索微服务)返回 None,调用方回落
    预签名直链(1h 内仍可下载打开)。
    """
    secret = os.getenv("JWT_SECRET", "")
    if not secret or not key:
        return None
    try:
        import hmac
        import hashlib
        import time as _t
        from urllib.parse import quote as _q
        cfg = _storage_config()
        exp = int(_t.time()) + int(cfg.TOS_URL_EXPIRES)
        sig = hmac.new(secret.encode(), f"doc|{key}|{exp}".encode(),
                       hashlib.sha256).hexdigest()
        return f"/api/media/doc?key={_q(str(key), safe='')}&exp={exp}&sig={sig}"
    except Exception:
        return None


def verify_doc_proxy(key, exp, sig):
    """校验 doc_proxy_url 生成的签名;过期/篡改/未配置密钥一律 False。"""
    secret = os.getenv("JWT_SECRET", "")
    if not secret or not key or not sig:
        return False
    try:
        import hmac
        import hashlib
        want = hmac.new(secret.encode(), f"doc|{key}|{int(exp)}".encode(),
                        hashlib.sha256).hexdigest()
        return hmac.compare_digest(want, str(sig))
    except Exception:
        return False


def resign_url(url):
    """历史消息里的预签名 URL 用当前凭证重签,过期会话回看仍可打开。

    仅处理本存储体系(TOS/VID 端点主机,虚拟主机或 path 风格都识别)的 URL:
    从 URL path 反解对象 key,按扩展名选图片/视频桶并补 inline/Content-Type。
    外链(旧 OSS 时代、第三方站点)与无法签名时原样返回,绝不抛错。
    """
    if not url or not isinstance(url, str) or not (
            url.startswith(("http://", "https://")) or url.startswith("/api/media/doc")):
        return url
    try:
        from urllib.parse import urlsplit, unquote, parse_qs
        cfg = _storage_config()
        sp = urlsplit(url)
        # 已是本代理链接(前端同步回存过):只需按当前密钥重签,别当外链放走
        if sp.path == "/api/media/doc":
            k = (parse_qs(sp.query).get("key") or [""])[0]
            return doc_proxy_url(k) or url
        host = (sp.hostname or "").lower()
        eps = set()
        for ep in (getattr(cfg, "TOS_ENDPOINT", ""), getattr(cfg, "VID_ENDPOINT", "")):
            if ep:
                h = urlsplit(ep if "://" in ep else "https://" + ep).hostname
                if h:
                    eps.add(h.lower())
        # 虚拟主机风格 bucket.tos-s3-xx.volces.com / bucket.oss-xx.aliyuncs.com
        if not any(host == ep or host.endswith("." + ep) for ep in eps):
            # 旧媒体存储(my-image-bucket-lly.oss-*.aliyuncs.com)对象已迁往 TOS:
            # 文档 qingxi/docs/<rel> -> <extdir>/<rel>(与 docs_key 同规则),
            # 视频 qingxi/mp4/<rel> -> 原 key(视频在 TOS 同路径)。其余外链不处理。
            if not host.endswith(".aliyuncs.com"):
                return url
            key = unquote(sp.path).lstrip("/")
            if key.startswith("qingxi/docs/"):
                rel = key[len("qingxi/docs/"):]
                d = _DOCS_EXT_DIR.get(os.path.splitext(rel)[1].lower())
                if not d:
                    return url
                key = d + "/" + rel
            elif not key.startswith("qingxi/mp4/"):
                return url
        else:
            key = unquote(sp.path).lstrip("/")
        if not key:
            return url
        if key.lower().startswith("qingxi/cad/"):
            # 历史消息缓存过 qingxi/CAD 的签名 URL;该前缀已删除,重签时剥前缀
            # 指向桶根 CAD/(内容 100% 一致,见 abs_to_key 的 CAD 注释)。
            key = key[len("qingxi/"):]
        ext = os.path.splitext(key)[1].lower()
        # 文档优先走后端代理(桶强制下载,直链无法 inline 预览);
        # 无 JWT_SECRET 时返回 None,继续走下面的预签名直链
        if ext in _DOC_EXTS:
            p = doc_proxy_url(key)
            if p:
                return p
        is_video = ext in _VIDEO_EXTS
        s3 = get_s3_video() if is_video else get_s3()
        if s3 is None:
            return url
        bucket = (cfg.VID_BUCKET if getattr(cfg, "VID_ENABLED", False) else cfg.TOS_BUCKET) \
            if is_video else cfg.TOS_BUCKET
        params = {"Bucket": bucket, "Key": key}
        if ext in _DOC_EXTS:
            params["ResponseContentDisposition"] = "inline"
        ctype = _CONTENT_TYPES.get(ext)
        if ctype:
            params["ResponseContentType"] = ctype
        return s3.generate_presigned_url("get_object", Params=params,
                                         ExpiresIn=int(cfg.TOS_URL_EXPIRES))
    except Exception:
        return url


def presigned(key, inline=False):
    """已验证存在的 TOS 对象 key(如入库时写入的 doc_key/video_key/image_key)
    -> 预签名 HTTPS 链接;跳过路径换算。不可签名返回 None(调用方回落路径换算)。"""
    if not key:
        return None
    s3 = get_s3()
    if s3 is None:
        return None
    try:
        return _sign_key(s3, str(key), C.TOS_BUCKET, inline=inline)
    except Exception:
        return None


def image_url(path):
    """图片路径 -> 预签名 HTTPS 链接(图片存储 TOS);未配置 TOS 时原样返回本地路径。

    支持:
    - 绝对路径(``D:\\清洗文件\\...``):按 :func:`abs_to_key` 换算(含旧 PDF 补 pdf/ 段);
    - 相对路径(``images/<hash>.jpg``,历史文本块):按文件名在 ald_image 索引反查对象 key。
    任何无法映射/签名失败都回退原始路径,保证检索不被打断。
    """
    if not path:
        return path
    s3 = get_s3()
    if s3 is None:
        return path                         # 未配置对象存储:本地路径(向后兼容)
    try:
        # 跨平台判定绝对路径(Linux 容器内也要认出 Windows 盘符路径),不能用 os.path.isabs
        key = abs_to_key(path) if _split_root(path) else None
        if not key:                         # 相对路径:按文件名(hash)反查
            bn = os.path.basename(path.replace("/", os.sep))
            key = _ensure_basename_index().get(bn)
        if not key:
            return path
        return _sign_key(s3, key, C.TOS_BUCKET)
    except Exception:
        return path                         # 签名失败不阻断检索


def video_url(path):
    """视频路径 -> 预签名 HTTPS 链接(视频存储,VID_* 配置的独立桶;未配置回落 TOS)。

    图片与视频分开存储:视频(mpeg/mp4)走 :func:`get_s3_video` 的桶与前缀,
    key 换算与 abs_to_key 同规则(前缀用 VID_KEY_PREFIX,缺省沿用 TOS_KEY_PREFIX)。
    未配置任何存储时原样返回本地路径。
    """
    if not path:
        return path
    s3 = get_s3_video()
    if s3 is None:
        return path
    try:
        prefix = (getattr(C, "VID_KEY_PREFIX", "") or C.TOS_KEY_PREFIX) \
            if getattr(C, "VID_ENABLED", False) else C.TOS_KEY_PREFIX
        key = _abs_to_key_with_prefix(path, prefix) if _split_root(path) else None
        if not key:
            return path
        bucket = C.VID_BUCKET if getattr(C, "VID_ENABLED", False) else C.TOS_BUCKET
        return _sign_key(s3, key, bucket)
    except Exception:
        return path


# ---- 原始文档(答案来源在线打开)----
# 文档本体在资料根目录(D:\\180-半导体设备相关资料！),已按扩展名分存 TOS:
# pdf/ docx/ pptx/ xlsx/ + <资料根相对路径>(上传脚本按 D:\\oss 镜像层级入桶)。
# source_path 里带着资料根目录名,据此换算;不再依赖 VID_* 独立视频桶
# (该桶已废弃,视频/文档统一回落 TOS)。
_DOCS_MARKER = "180-半导体设备相关资料！"
_SPLIT_STEM_RE = re.compile(r"^(.*)_p(\d+)-(\d+)$")   # 大 PDF 清洗拆分段 <stem>_p<x>-<y>
_DOCS_EXT_DIR = {".pdf": "pdf", ".doc": "docx", ".docx": "docx",
                 ".ppt": "pptx", ".pptx": "pptx", ".xls": "xlsx", ".xlsx": "xlsx"}


def docs_key(source_path):
    """payload source_path -> 文档对象 key;不可映射返回 None。

    - 取资料根目录名之后的相对路径,按扩展名补模态目录段:
      <rel> -> pdf/<rel> / docx/<rel> / pptx/<rel> / xlsx/<rel>(TOS);
    - 清洗期大 PDF 的拆分段文件(<stem>_p<x>-<y>.pdf)是临时产物已删,
      重映射回同目录原合集 <stem>.pdf(SEMI 标准合集等 171 个分段由此还原)。
    """
    if not source_path:
        return None
    s = str(source_path).replace("\\", "/")
    i = s.rfind(_DOCS_MARKER)
    if i < 0:
        return None
    rel = s[i + len(_DOCS_MARKER):].lstrip("/")
    if not rel:
        return None
    dirn, base = (rel.rsplit("/", 1) + [""])[:2] if "/" in rel else ("", rel)
    stem, ext = os.path.splitext(base)
    m = _SPLIT_STEM_RE.match(stem)
    if m and ext.lower() in (".pdf", ".ppt", ".pptx", ".doc", ".docx"):
        base = m.group(1) + ext
        rel = (dirn + "/" + base) if dirn else base
    d = _DOCS_EXT_DIR.get(ext.lower())
    if not d:
        return None
    return d + "/" + rel


def doc_url(source_path):
    """文档 source_path -> 预签名 HTTPS 链接(浏览器内联打开,不落下载)。

    ResponseContentDisposition=inline 显式覆盖:即便对象/桶带强制下载元数据,
    签名参数也把浏览器行为掰回内联预览(PDF 直接看,PPT/DOC 走下载或预览)。
    文档统一在 TOS。不可映射/未配置存储返回 ""(前端按无链接渲染)。
    """
    if not source_path:
        return ""
    s3 = get_s3()
    if s3 is None:
        return ""
    try:
        key = docs_key(source_path)
        if not key:
            return ""
        return _sign_key(s3, key, C.TOS_BUCKET, inline=True)
    except Exception:
        return ""
