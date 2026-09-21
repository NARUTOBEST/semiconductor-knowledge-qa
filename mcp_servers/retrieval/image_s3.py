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
_PROJECT = os.path.dirname(_HERE)                                  # project root
_CONFIG = os.path.join(_PROJECT, "config")
if _CONFIG not in sys.path:
    sys.path.insert(0, _CONFIG)

import config as C  # noqa: E402

_client = None
_video_client = None
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


def get_s3():
    """懒加载图片存储客户端(TOS)。未配置返回 None。"""
    global _client
    if _client is not None:
        return _client
    if not C.TOS_ENABLED:
        return None
    with _lock:
        if _client is None:
            _client = _make_client(C.TOS_ENDPOINT, C.TOS_ACCESS_KEY,
                                   C.TOS_SECRET_KEY, C.TOS_REGION,
                                   C.TOS_USE_PATH_STYLE)
    return _client


def get_s3_video():
    """懒加载视频存储客户端(独立 S3 兼容存储,如阿里云 OSS)。

    图片与视频可分开存放:图片在 TOS(VID_* 未配置时视频也回落 TOS)。
    未配置任何存储返回 None(与 get_s3 一致,调用方回退本地路径)。
    """
    global _video_client
    if _video_client is not None:
        return _video_client
    if not getattr(C, "VID_ENABLED", False):
        return get_s3()                     # 未配 VID_*:视频与图片同存储
    with _lock:
        if _video_client is None:
            _video_client = _make_client(C.VID_ENDPOINT, C.VID_ACCESS_KEY,
                                         C.VID_SECRET_KEY, C.VID_REGION,
                                         C.VID_USE_PATH_STYLE)
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


def _sign_key(s3, key, bucket):
    return s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=C.TOS_URL_EXPIRES,
    )


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
