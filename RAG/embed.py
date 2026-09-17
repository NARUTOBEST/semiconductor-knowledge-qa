# -*- coding: utf-8 -*-
"""嵌入:BGE-m3 文本(dense+sparse)+ CLIP 图像。懒加载,首次实例化时下载模型。"""
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")   # 国内 HF 镜像
import threading
import time

import numpy as np

import config as C

_te = None      # BGEM3FlagModel(在线:用户查询/记忆嵌入)
_ie = None      # CLIP SentenceTransformer(在线)
_te_offline = None   # BGE-m3 副本(离线:批量入库;懒加载)
_ie_offline = None   # CLIP 副本(离线;懒加载)
_device = None

# 模型首加载互斥:并发首请求(检索/预热/warmup 多线程)下,避免多个线程各自加载
# 数 GB 的 BGE-m3/CLIP/reranker 导致内存翻倍甚至 OOM。
_model_lock = threading.Lock()

# 推理互斥:同一模型实例的并发前向不安全(torch 无跨线程前向正确性保证),且
# CPU 下并发只是争抢同一批核。按【模型】各持一把锁:不同模型是独立 nn.Module,
# 可安全并行(GPU kernel 流水,CPU 预处理重叠)——encode(BGE-m3)与 rerank
# (reranker)不再互相排队,消除跨模型尾延迟放大。
ENCODE_LOCK = threading.Lock()   # BGE-m3 文本编码(单飞批在锁内整批执行)
IMAGE_LOCK = threading.Lock()    # CLIP 图像/文本编码
RERANK_LOCK = threading.Lock()   # BGE-reranker 交叉编码打分

# 双实例隔离(在线查询 vs 批量入库):入库用独立加载的模型副本 + 独立推理锁,
# 入库整批前向不再与在线查询抢同一个模型实例(彻底消除排队;代价是模型内存
# ×2,离线副本懒加载——从没入过库就不占)。CPU 部署下两实例仍共享 CPU 核,
# 离线实例让路逻辑改为看【在线实例】的排队情况,入库在高负载时自动降速。
# RETRIEVAL_DUAL_ENCODER=0 可退回单实例(B1 让路继续兜底隔离)。
_DUAL = os.getenv("RETRIEVAL_DUAL_ENCODER", "1") not in ("0", "false", "no")
INGEST_ENCODE_LOCK = threading.Lock()   # 离线 BGE-m3(入库专用)
INGEST_IMAGE_LOCK = threading.Lock()    # 离线 CLIP(入库专用)

# 攒批窗口/单批上限(env 可调):窗口越大合并率越高、单请求延迟略增;超上限切分分批。
_ENCODE_WINDOW_S = float(os.getenv("RETRIEVAL_ENCODE_WINDOW_MS", "20")) / 1000
_ENCODE_MAX_BATCH = int(os.getenv("RETRIEVAL_ENCODE_MAX_BATCH", "16"))

# 入库让路:批量(入库)编码在【批与批之间】检查在线单条请求是否在排队,有人等
# 就先歇 _INGEST_YIELD_S 再继续(锁没法中途打断,让路只能发生在两批之间)。
# 在线请求最多等一小批,不再等整批入库;入库总耗时略增(本就不赶时间)。0=不让路。
_INGEST_YIELD_S = float(os.getenv("RETRIEVAL_INGEST_YIELD_MS", "50")) / 1000


def _split_axis0(result, n):
    """一次批量 encode 的返回 -> n 个"单条调用形状"的结果(按轴0切片)。

    BGE-m3 返回 (dense ndarray n×1024, sparse list[dict]),CLIP 返回 ndarray n×512,
    均按轴0切;切出的第 i 份与"只编码第 i 条"的返回形状完全一致,调用方无感知。
    """
    if isinstance(result, tuple):
        parts = [_split_axis0(r, n) for r in result]
        return [tuple(p[i] for p in parts) for i in range(n)]
    return [result[i:i + 1] for i in range(n)]


class _Slot:
    """单条挂起请求:等待期挂在队列上,执行完回填 result(成功)或 error(失败)。"""
    __slots__ = ("text", "result", "error")

    def __init__(self, text):
        self.text = text
        self.result = None
        self.error = None

    @property
    def done(self):
        return self.result is not None or self.error is not None


def _join_axis0(results):
    """多个分批的原始返回拼接回一个整体(_split_axis0 的逆操作,批间拼接)。

    tuple 逐元素递归拼接(BGE-m3 的 (dense, sparse));ndarray 轴0拼接;
    其余(list,如 sparse)按序 extend。
    """
    if len(results) == 1:
        return results[0]
    if isinstance(results[0], tuple):
        return tuple(_join_axis0([r[k] for r in results])
                     for k in range(len(results[0])))
    if isinstance(results[0], np.ndarray):
        return np.concatenate(results, axis=0)
    out = []
    for r in results:
        out.extend(r)
    return out


class _BatchedEncoder:
    """单飞批合并 + 推理互斥包装器(leader-follower 模式)。

    并发到达的单条 encode 请求被攒成一次批量调用:
      - 首个到达者成为 leader:等一个攒批窗口(默认 20ms)收拢后来者,循环把
        pending 队列整批执行(持 INFER_LOCK),直到窗口内不再有新请求;
      - 后来者挂入队列后阻塞等待,结果按到达顺序切分回填(axis-0 切片),
        返回形状与"自己单独调用一次"完全一致。
    多条(批量入库)调用不经窗口,直接切分上限内整批持锁执行。
    失败语义:同批请求共享同一次调用——批量调用抛错则整批各自收到同一异常。
    """

    def __init__(self, inner, window_s=_ENCODE_WINDOW_S, max_batch=_ENCODE_MAX_BATCH,
                 lock=None, yield_check=None):
        self._inner = inner
        self._window_s = window_s
        self._max_batch = max(1, max_batch)
        self._infer_lock = lock or ENCODE_LOCK   # 本模型自己的推理锁
        # 让路判断来源:默认看自己的在线队列;离线(入库)实例传入
        # "在线实例的 _online_waiting",实现跨实例让路。
        self._yield_check = yield_check or self._online_waiting
        self._cond = threading.Condition()
        self._pending = []        # list[_Slot],_cond 保护
        self._inflight = False    # 已有 leader 在收集/执行

    def encode(self, texts):
        texts = list(texts)
        if len(texts) != 1:       # 已是批量:无窗口开销,直接整批持锁执行
            return self._run_batched(texts)
        with self._cond:
            slot = _Slot(texts[0])
            self._pending.append(slot)
            leader = not self._inflight
            self._inflight = True
        if leader:
            self._flush_loop()
        else:
            with self._cond:
                # 结果/异常由 leader 回填并 notify
                self._cond.wait_for(lambda: slot.done)
        with self._cond:
            if slot.error is not None:
                raise slot.error
            return slot.result

    def _flush_loop(self):
        while True:
            time.sleep(self._window_s)   # 攒批窗口:等后来者进队
            with self._cond:
                batch = self._pending
                self._pending = []
                if not batch:
                    self._inflight = False
                    return
            try:
                self._execute(batch)
            except BaseException as e:   # 防御:执行器自身异常不能让 follower 永久挂起
                with self._cond:
                    for s in batch:
                        s.error = e
                    self._cond.notify_all()

    def _execute(self, batch):
        try:
            raw = self._run_batched([s.text for s in batch])
            results = _split_axis0(raw, len(batch))
        except Exception as e:
            with self._cond:
                for s in batch:
                    s.error = e
                self._cond.notify_all()
            return
        with self._cond:
            for s, r in zip(batch, results):
                s.result = r
            self._cond.notify_all()

    def _online_waiting(self) -> bool:
        """是否有在线单条请求在排队/收集窗口中(入库让路的判断依据)。"""
        with self._cond:
            return bool(self._pending) or self._inflight

    def _run_batched(self, texts):
        """批量执行:返回与 inner.encode(texts) 同构的原始形状(超上限分批再拼接)。

        入库让路:批与批之间若发现在线单条请求在排队(_pending/_inflight),
        先歇 _INGEST_YIELD_S 让在线流量先行,再继续下一批(每批独立持锁,
        间隙天然是让路点)。
        """
        outs = []
        for i in range(0, len(texts), self._max_batch):
            chunk = texts[i:i + self._max_batch]
            if i and _INGEST_YIELD_S > 0 and self._yield_check():
                time.sleep(_INGEST_YIELD_S)
            with self._infer_lock:
                outs.append(self._inner.encode(chunk))
        return _join_axis0(outs)

    def __getattr__(self, name):
        """其余属性透传;可调用方法(如 CLIP 的 encode_image/encode_text)自动持推理锁。"""
        attr = getattr(self._inner, name)
        if not callable(attr):
            return attr

        def _locked(*a, **kw):
            with self._infer_lock:
                return attr(*a, **kw)
        _locked.__name__ = getattr(attr, "__name__", name)
        return _locked


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


def _ensure_snapshot_offline(repo_id):
    """只允许用本地快照,缺失/不完整直接抛异常(不联网)。

    用于 reranker 这类"可选增强"模型:在线下载在断网/镜像不可达时会长时间
    挂起并持有模型锁,拖垮整个检索服务;离线快速失败后上层可降级(按召回
    分数排序),而不是卡死。需要安装该模型时,先手动把权重放入缓存目录。
    """
    from huggingface_hub import snapshot_download
    cache_root = os.path.join(os.path.expanduser("~"), ".cache",
                              "huggingface", "hub_local")
    local_dir = os.path.join(cache_root, repo_id.replace("/", "__"))
    ignore = ["*.DS_Store", "imgs/*", "*.onnx", "onnx/*", "*.ot", "*.msgpack"]
    return snapshot_download(repo_id=repo_id, local_dir=local_dir,
                             ignore_patterns=ignore, local_files_only=True)


# ---------- 文本嵌入 BGE-m3 ----------
class TextEncoder:
    def __init__(self, model_name=C.TEXT_EMBED_MODEL):
        from FlagEmbedding import BGEM3FlagModel
        local = _ensure_snapshot(model_name)
        self.model = BGEM3FlagModel(local, use_fp16=(_device_str() == "cuda"),
                                    device=_device_str())
        # 兜底:实测 FlagEmbedding 1.4 传 device="cuda" 仍可能静默落在 CPU
        # (next(model.parameters()).device == cpu)。显式 .to(cuda) 强制上 GPU。
        if _device_str() == "cuda":
            try:
                import torch
                if torch.cuda.is_available() and \
                        next(self.model.model.parameters()).device.type != "cuda":
                    self.model.model.to("cuda")
            except Exception:
                pass

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


def get_text_encoder(offline=False):
    """BGE-m3 编码器单例。

    offline=False(默认):在线实例——用户查询/记忆嵌入走这里(单飞批合并)。
    offline=True:离线实例——批量入库专用,独立模型副本 + INGEST_ENCODE_LOCK,
    与在线实例完全并行;其让路判断看在线实例排队情况(共享 CPU 核时 polite)。
    离线实例懒加载:从未入库则不占第二份模型内存。RETRIEVAL_DUAL_ENCODER=0
    时退回单实例(在线离线共用,靠让路兜底隔离)。
    """
    global _te, _te_offline
    if not _DUAL:
        offline = False
    if offline:
        if _te_offline is None:
            with _model_lock:
                if _te_offline is None:
                    online = get_text_encoder()
                    _te_offline = _BatchedEncoder(
                        TextEncoder(), lock=INGEST_ENCODE_LOCK,
                        yield_check=online._online_waiting)
        return _te_offline
    if _te is None:
        with _model_lock:
            if _te is None:
                _te = _BatchedEncoder(TextEncoder(), lock=ENCODE_LOCK)
    return _te


# ---------- 图像嵌入 CLIP ----------
class ImageEncoder:
    def __init__(self, model_name=C.IMAGE_EMBED_MODEL):
        from sentence_transformers import SentenceTransformer
        local = _ensure_snapshot(model_name)
        self.model = SentenceTransformer(local, device=_device_str())
        # fp16:与 BGE-m3/reranker 同口径(cuda 下约 2x 提速、显存减半;
        # encode 输出统一向上转 float32,调用方无感知)
        if _device_str() == "cuda":
            try:
                self.model.half()
            except Exception:
                pass

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


def get_image_encoder(offline=False):
    """CLIP 编码器单例。offline=True 为入库专用独立副本(见 get_text_encoder)。"""
    global _ie, _ie_offline
    if not _DUAL:
        offline = False
    if offline:
        if _ie_offline is None:
            with _model_lock:
                if _ie_offline is None:
                    online = get_image_encoder()
                    _ie_offline = _BatchedEncoder(
                        ImageEncoder(), lock=INGEST_IMAGE_LOCK,
                        yield_check=online._online_waiting)
        return _ie_offline
    if _ie is None:
        with _model_lock:
            if _ie is None:
                _ie = _BatchedEncoder(ImageEncoder(), lock=IMAGE_LOCK)
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
        # reranker 为可选增强:仅用本地快照,缺失即快速失败(上层降级为召回顺序),
        # 不在请求路径上联网下载(断网时会长时间挂起并持模型锁)。
        local = _ensure_snapshot_offline(model_name)
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
        with _model_lock:
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
    img = os.path.join(r"D:\清洗文件\pdf\0001 半导体设备资料合集\ALD原子沉积资料\ALD原子沉积资料\Oxford ALD Operation Manual\auto\images",
                       sorted(os.listdir(r"D:\清洗文件\pdf\0001 半导体设备资料合集\ALD原子沉积资料\ALD原子沉积资料\Oxford ALD Operation Manual\auto\images"))[0])
    v = ie.encode([img])
    print(f"  image vec: {v.shape}  norm={np.linalg.norm(v[0]):.3f}")
