r"""给图像库里的人像/证件照打 content_type=portrait 标签,供 query_image 默认过滤。

规则(针对 LLM 中文描述,已验证精确区分 5 张人像 vs 2 张含人物的设备/示意图):
  description 匹配 (人像|肖像) 或 (人物 且 (男性|女性)) -> 判为人像。
  说明:真设备照(Fiji 操作员戴手套)和示意图(画人物示意尺寸)只含"人物"一个词、
  不含 人像/肖像/男性/女性,故不会被误伤。
打标签用 set_payload,不重新编码、不重新切块。
"""
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
_RAG = os.path.dirname(_HERE)
_PROJECT = os.path.dirname(_RAG)
_CONFIG = os.path.join(_PROJECT, "config")
for _p in (_HERE, _RAG, _CONFIG):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import config as C
from qdrant_client import QdrantClient

_HEAD = ("人像", "肖像")          # 直接判人像的词
_PERSON = "人物"                  # 需配合性别词才判人像
_GENDER = ("男性", "女性")


def is_portrait(text: str) -> bool:
    if not text:
        return False
    if any(k in text for k in _HEAD):
        return True
    if _PERSON in text and any(k in text for k in _GENDER):
        return True
    return False


def main():
    cl = QdrantClient(path=C.QDRANT_PATH)
    all_pts, off = [], None
    while True:
        r = cl.scroll(C.IMAGE_COLLECTION, limit=256, offset=off, with_payload=True)
        all_pts.extend(r[0]); off = r[1]
        if off is None:
            break
    print(f"图像点总数: {len(all_pts)}")

    portrait_ids = []
    for p in all_pts:
        desc = (p.payload.get("description") or "") + " " + (p.payload.get("caption") or "")
        if is_portrait(desc):
            portrait_ids.append(p.id)
            print(f"  [人像] {p.payload.get('source_stem')} p{p.payload.get('page_num')} "
                  f"{p.payload.get('chunk_id')}  type={p.payload.get('item_type')}")
            print(f"        {(p.payload.get('description') or '')[:70]}")
    print(f"\n命中人像: {len(portrait_ids)} 个")

    if portrait_ids:
        cl.set_payload(
            collection_name=C.IMAGE_COLLECTION,
            payload={"content_type": "portrait"},
            points=portrait_ids,
        )
        print(f"已打标签 content_type=portrait ({len(portrait_ids)} 个点)")
    else:
        print("无人像需要打标签")
    cl.close()


if __name__ == "__main__":
    main()
