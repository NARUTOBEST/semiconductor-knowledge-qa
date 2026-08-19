# -*- coding: utf-8 -*-
"""嵌入:BGE-m3 文本(dense+sparse)+ CLIP 图像。懒加载,首次实例化时下载模型。"""
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")   # 国内 HF 镜像
import numpy as np

import config as C

_te = None      # BGEM3FlagModel
_ie = None      # CLIP SentenceTransformer
_device = None


def _device_str():
    """嵌入模型设备:优先用 config.EMBED_DEVICE(默认 cpu,给 LLM 腾出 GPU)。"""
    return getattr(C, "EMBED_DEVICE", None) or "cpu"


def _ensure_snapshot(repo_id):
    """先手动 snapshot_download 到本地目录,跳过 .DS_Store / imgs/ / onnx 等无用或会 403 的文件
    (bge-m3 仓库里混进 imgs/.DS_Store,hf-mirror 对它 403,无过滤的 snapshot_download 会让整个下载失败),
    再把本地路径交给下游模型,避免它再次全量拉取。

    离线优先:本地已缓存时用 local_files_only=True 直接返回(不联网校验 etag),
    断网环境也能用;仅当本地无缓存时才联网下载。"""
    from huggingface_hub import snapshot_download
    cache_root = os.path.join(os.path.expanduser("~"), ".cache",
                              "huggingface", "hub_local")
    local_dir = os.path.join(cache_root, repo_id.replace("/", "__"))
    ignore = ["*.DS_Store", "imgs/*", "*.onnx", "onnx/*", "*.ot", "*.msgpack"]
    if os.path.exists(local_dir):
        try:
            return snapshot_download(repo_id=repo_id, local_dir=local_dir,
                                     ignore_patterns=ignore, local_files_only=True)
        except Exception:
            pass  # 本地不完整,fallback 到联网下载
    return snapshot_download(repo_id=repo_id, local_dir=local_dir,
                             ignore_patterns=ignore)


# ---------- 文本嵌入 BGE-m3 ----------
class TextEncoder:
    def __init__(self, model_name=C.TEXT_EMBED_MODEL):
        from FlagEmbedding import BGEM3FlagModel
        local = _ensure_snapshot(model_name)
        self.model = BGEM3FlagModel(local, use_fp16=(_device_str() == "cuda"),
                                    device=_device_str())

    def encode(self, texts, batch_size=12):
        """返回 (dense: np.ndarray (n,1024), sparse: list[dict{token_id:weight}])。"""
        out = self.model.encode(list(texts), batch_size=batch_size,
                                return_dense=True, return_sparse=True,
                                return_colbert_vecs=False)
        dense = np.asarray(out["dense_vecs"], dtype=np.float32)
        # L2 归一化(Qdrant Cosine 也能做,这里显式归一更稳)
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        dense = dense / norms
        return dense, list(out["lexical_weights"])


def get_text_encoder():
    global _te
    if _te is None:
        _te = TextEncoder()
    return _te


# ---------- 图像嵌入 CLIP ----------
class ImageEncoder:
    def __init__(self, model_name=C.IMAGE_EMBED_MODEL):
        from sentence_transformers import SentenceTransformer
        local = _ensure_snapshot(model_name)
        self.model = SentenceTransformer(local, device=_device_str())

    def encode(self, image_paths, batch_size=16):
        from PIL import Image
        imgs = []
        for p in image_paths:
            try:
                # with 确保 PIL 惰性加载的文件句柄立即释放,大批量入库不积累
                with Image.open(p) as im:
                    imgs.append(im.convert("RGB"))
            except Exception:
                imgs.append(None)
        # 逐批编码,跳过打不开的图(用零向量占位,后续不入库)
        vecs = np.zeros((len(image_paths), 512), dtype=np.float32)
        i = 0
        while i < len(imgs):
            batch = imgs[i:i + batch_size]
            valid_idx = [j for j, im in enumerate(batch) if im is not None]
            if valid_idx:
                sub = [batch[j] for j in valid_idx]
                emb = self.model.encode(sub, convert_to_numpy=True,
                                        normalize_embeddings=True)
                for k, j in enumerate(valid_idx):
                    vecs[i + j] = emb[k]
            i += batch_size
        return vecs

    def encode_text(self, texts):
        """CLIP 文本编码(用于文本查询图库)。"""
        emb = self.model.encode(list(texts), convert_to_numpy=True,
                                normalize_embeddings=True)
        return np.asarray(emb, dtype=np.float32)


def get_image_encoder():
    global _ie
    if _ie is None:
        _ie = ImageEncoder()
    return _ie



# ---------- 重排 BGE-reranker-v2-m3 ----------
_re = None  # FlagReranker

class Reranker:
    """Cross-Encoder 重排器:对 (query, document) 对打分,精度远高于双塔检索。

    与 BGE-m3(双塔)的区别:
      双塔: query 和 doc 分别编码 -> 余弦相似度 -> 快但粗
      交叉: query 和 doc 拼接 -> Transformer 联合编码 -> 慢但准

    用法: 召回 20 条(BGE-m3)-> 重排取 top 3(Reranker)
    """
    def __init__(self, model_name=C.RERANK_MODEL):
        from FlagEmbedding import FlagReranker
        local = _ensure_snapshot(model_name)
        self.model = FlagReranker(
            local,
            device=_device_str(),
            use_fp16=(_device_str() == "cuda"),
        )

    def rerank(self, query, documents, batch_size=8):
        """对 (query, doc) 对打分,返回归一化分数列表(0~1)。

        documents 为字符串列表,每条截断到 RERANK_MAX_CONTENT 字符。
        """
        max_len = getattr(C, "RERANK_MAX_CONTENT", 512)
        pairs = [[query, doc[:max_len]] for doc in documents]
        scores = self.model.compute_score(
            pairs, normalize=True, batch_size=batch_size
        )
        # 单条时 compute_score 返回 float,需统一为 list
        if isinstance(scores, float):
            scores = [scores]
        return list(scores)


def get_reranker():
    """懒加载 Reranker(首次调用下载模型 ~2GB,后续走缓存)。"""
    global _re
    if _re is None:
        _re = Reranker()
    return _re

if __name__ == "__main__":
    # 自测:编码 3 条文本 + 1 张图
    print(f"device: {_device_str()}")
    print("加载 BGE-m3(首次会下载 ~2.3GB)...")
    te = get_text_encoder()
    d, s = te.encode(["ALD 原子层沉积工艺", "Savannah 200 thermal ALD system",
                      "wafer chuck temperature control"])
    print(f"  text dense: {d.shape}  sparse 非零词数: {[len(x) for x in s]}")
    print("加载 CLIP(首次会下载 ~600MB)...")
    ie = get_image_encoder()
    img = os.path.join(r"D:\清洗文件\0001 半导体设备资料合集\ALD原子沉积资料\ALD原子沉积资料\Oxford ALD Operation Manual\auto\images",
                       sorted(os.listdir(r"D:\清洗文件\0001 半导体设备资料合集\ALD原子沉积资料\ALD原子沉积资料\Oxford ALD Operation Manual\auto\images"))[0])
    v = ie.encode([img])
    print(f"  image vec: {v.shape}  norm={np.linalg.norm(v[0]):.3f}")
