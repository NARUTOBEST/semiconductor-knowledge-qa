# -*- coding: utf-8 -*-
"""OpenAI 兼容 API 调用:多模态图描述 + 文本模型 LLM 兜底切分。

- key 从 env/env.env 的 OPENAI_API_KEY 读取(经 config.OPENAI_API_KEY 注入)。
- 图描述:多模态模型,extra_body thinking disabled 省 token。
- LLM 兜底:文本模型,给超长单句文本,返回切分后的多块(不重写原文,仅在句间/从句间切)。
- 缓存:per-PDF _client_descriptions.json,重跑跳过已描述的图。
"""
import os, json, base64, time, re

import config as C

try:
    from openai import OpenAI
except ImportError as e:
    raise RuntimeError("缺少 openai SDK,先装:pip install openai") from e


# ---------- key ----------
def load_api_key():
    """从 config.OPENAI_VISION_API_KEY 读取(图描述用火山方舟 key)。"""
    key = C.OPENAI_VISION_API_KEY
    if not key:
        raise RuntimeError("未配置 OPENAI_VISION_API_KEY(请在 env/env.env 设置)")
    return key


_client = None
def get_client():
    global _client
    if _client is None:
        # 统一 20s 读超时(图描述含图片上传,失败有 describe_image 的 3 次重试兜底)
        _client = OpenAI(api_key=load_api_key(), base_url=C.OPENAI_VISION_BASE_URL, timeout=20)
    return _client


# ---------- 图描述 ----------
_DESC_PROMPT = (
    "你是半导体设备/工艺文档的图片分析助手。请用简洁中文描述这张图/图表,便于后续检索:"
    "1) 图的类型(照片/示意图/曲线图/流程图/表格截图等);"
    "2) 展示的核心内容、关键标签/部件名/坐标轴/数值趋势;"
    "3) 若有文字标注,摘录关键文字。直接输出描述,不要寒暄,300字以内。"
)

def _img_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def describe_image(img_path, prompt=_DESC_PROMPT, retries=3):
    """调多模态模型描述一张图,返回描述文本。"""
    b64 = _img_b64(img_path)
    client = get_client()
    last = None
    for i in range(retries):
        try:
            resp = client.chat.completions.create(
                model=C.OPENAI_VISION_MODEL,
                messages=[{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": prompt},
                ]}],
                extra_body=C.OPENAI_VISION_EXTRA,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            last = e
            time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"describe_image 失败({img_path}): {last}")


# ---------- LLM 兜底切分 ----------
_SPLIT_PROMPT = (
    "下面是一段较长的文档文本(可能是一整句长句或无标点的连续文本)。"
    "请把它切成 {n} 段左右、每段约 {target} 字的语义连贯小块,"
    "只能在句号/分号/逗号/换行等边界切,不得改写、增删或翻译原文。"
    "只返回 JSON 数组,元素为切分后的字符串(原样拼接应等于原文),不要任何解释。"
    "文本:\n{text}"
)

def llm_split_text(text, target=500, retries=2):
    """LLM 兜底切分超长单句文本。返回 list[str](>1 块);失败返回 None。"""
    client = get_client()
    n = max(2, round(len(text) / target))
    prompt = _SPLIT_PROMPT.format(n=n, target=target, text=text)
    for i in range(retries):
        try:
            resp = client.chat.completions.create(
                model=C.OPENAI_TEXT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            raw = resp.choices[0].message.content.strip()
            raw = re.sub(r"^```(json)?|```$", "", raw, flags=re.M).strip()
            arr = json.loads(raw)
            if isinstance(arr, list) and len(arr) > 1:
                # 校验:拼接应≈原文(允许空白差异)
                joined = "".join(arr)
                if len(joined) >= len(text) * 0.9:
                    return [s.strip() for s in arr if s.strip()]
            return None
        except Exception:
            time.sleep(1.5 * (i + 1))
    return None


# ---------- 缓存 ----------
def load_desc_cache(auto_dir):
    p = os.path.join(auto_dir, C.OPENAI_DESC_FILENAME)
    if os.path.exists(p):
        try:
            return json.load(open(p, encoding="utf-8"))
        except Exception:
            return {}
    return {}

def save_desc_cache(auto_dir, cache):
    p = os.path.join(auto_dir, C.OPENAI_DESC_FILENAME)
    json.dump(cache, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

def describe_with_cache(img_relname, img_abspath, auto_dir, cache=None):
    """带缓存的图描述。img_relname 是 content_list 里的 img_path(相对 auto/)。"""
    if cache is None:
        cache = load_desc_cache(auto_dir)
    # 路径分隔符归一化:旧缓存用反斜杠(images\xxx),content_list 用正斜杠(images/xxx)
    key_norm = img_relname.replace("\\", "/")
    for k in (img_relname, key_norm, key_norm.replace("/", "\\")):
        if k in cache:
            return cache[k], cache
    desc = describe_image(img_abspath)
    cache[key_norm] = desc
    save_desc_cache(auto_dir, cache)
    time.sleep(C.OPENAI_CALL_INTERVAL if hasattr(C, "OPENAI_CALL_INTERVAL") else 0.3)
    return desc, cache


if __name__ == "__main__":
    # 自测:描述 Oxford 的第一张图,确认多模态模型 id 可用
    CLEAN = r"D:\清洗文件\0001 半导体设备资料合集\ALD原子沉积资料\ALD原子沉积资料"
    stem = "Oxford ALD Operation Manual"
    auto = os.path.join(CLEAN, stem, "auto")
    # 找第一张图
    imgs = sorted(f for f in os.listdir(os.path.join(auto, "images")))
    img = os.path.join(auto, "images", imgs[0])
    print(f"模型 id: {C.OPENAI_VISION_MODEL}")
    print(f"测试图: {imgs[0]}  ({os.path.getsize(img)/1024:.0f} KiB)")
    print("调用中...")
    try:
        desc = describe_image(img)
        print(f"\n✅ 描述成功,长度 {len(desc)} 字:\n{desc}")
    except Exception as e:
        print(f"\n❌ 失败: {e}")
