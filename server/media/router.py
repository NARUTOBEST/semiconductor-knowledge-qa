# -*- coding: utf-8 -*-
"""文档在线预览代理路由。

TOS 桶开了强制下载(桶级行为):预签名 URL 里的
``response-content-disposition=inline`` 会被无视,响应一律带
``Content-Disposition: attachment``,浏览器点开来源卡片只会落盘下载,
PDF 没法在线看。本路由把对象从 TOS 流式转发回浏览器,并强制
``inline`` + 按扩展名给正确 Content-Type,前端来源卡片的 doc_url
指向这里即可在线预览(#page= 锚点也由浏览器 PDF 查看器接管)。

鉴权:URL 自带 HMAC-SHA256 签名(JWT_SECRET + 对象 key + 过期时间),
与对象存储预签名同一思路 —— 新标签页直接打开,无需携带 JWT。
签名生成/校验都复用 mcp_servers.retrieval.image_s3 的
doc_proxy_url / verify_doc_proxy。
"""
import logging
import os
from urllib.parse import quote

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

logger = logging.getLogger("media")
router = APIRouter()

# 按扩展名回 Content-Type(存量对象多为 binary/octet-stream,不给对类型
# 浏览器 PDF 查看器/Office 预览都不认)
_CONTENT_TYPES = {
    ".pdf": "application/pdf",
    ".doc": "application/msword",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".ppt": "application/vnd.ms-powerpoint",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".xls": "application/vnd.ms-excel",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}

_CHUNK = 1 << 20  # 1MB 流式转发


@router.get("/doc")
def get_doc(key: str, exp: int, sig: str):
    """流式转发 TOS 文档对象,强制 inline 返回(签名校验见模块 docstring)。"""
    from mcp_servers.retrieval.image_s3 import verify_doc_proxy, get_s3

    if not verify_doc_proxy(key, exp, sig):
        raise HTTPException(status_code=403, detail="invalid or expired link")
    s3 = get_s3()
    if s3 is None:
        raise HTTPException(status_code=503, detail="storage unavailable")
    try:
        obj = s3.get_object(Bucket=_bucket(), Key=key)
    except Exception:
        logger.warning("doc proxy fetch failed: %s", key[:160])
        raise HTTPException(status_code=404, detail="object not found")

    ext = os.path.splitext(key)[1].lower()
    name = os.path.basename(key)
    headers = {
        # RFC 5987 编码文件名,中文/空格不乱码;inline 扳回在线预览
        "Content-Disposition": f"inline; filename*=UTF-8''{quote(name)}",
        "Cache-Control": "private, max-age=600",
    }
    length = obj.get("ContentLength")
    if length is not None:
        headers["Content-Length"] = str(length)

    def _stream():
        body = obj["Body"]
        try:
            while True:
                chunk = body.read(_CHUNK)
                if not chunk:
                    break
                yield chunk
        finally:
            body.close()

    return StreamingResponse(
        _stream(),
        media_type=_CONTENT_TYPES.get(ext, "application/octet-stream"),
        headers=headers,
    )


def _bucket():
    """TOS 桶名(与 image_s3 同一套配置解析,规避 namespace 包遮蔽)。"""
    from mcp_servers.retrieval.image_s3 import _storage_config

    return _storage_config().TOS_BUCKET
