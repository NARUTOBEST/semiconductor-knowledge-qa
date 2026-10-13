# -*- coding: utf-8 -*-
"""设备型号 token 抽取(查询侧与入库侧共用,同一套规则保证两侧词表一致)。

用途:设备/型号双路融合检索。入库时给每块 payload.devices 打"正文/标题里出现
过的型号 token";查询时从问题抽同样的 token,有命中则在 RRF 融合前并列加一条
devices 过滤召回路——被邻题文档淹没的设备块即使无过滤池排 100 名外也能进重排池,
无过滤两路同时保留,金标不丢(不做硬过滤:纯字母缩写 MBE/CSP/TAB 会误杀工艺类
金标,实测 1000 题仿真误杀 65%,故只用软融合)。

规则(实测调优,见 eval/runs/50x5_20260919_065506 分桶分析):文本先按非字母
数字切片段,逐片段判型——
  - 字母+数字混合(D4i / I12 / H4 / V12):设备/模组型号主力;
  - 全大写字母 >=3 位(FUJI / NXT / ASM / MBE):品牌与缩写;
  - 首字母大写 >=3 位且非常用英文词(Datacon / Savannah):品牌名;
  - 纯数字一律排除:报警码(1420/2412)与型号数字(2200)无法区分,报警码走
    exact_code_recall 兜底,型号数字在品牌 token(DATACON)覆盖下无增量。
查询与入库对同一片段产出同一 token,两侧词表天然一致。
"""
import re

_FRAG = re.compile(r"[0-9A-Za-z]+")
# 首字母大写品牌词判定时排除的常用英文词(英文手册句首词会污染标签,纯无害但
# 稀释过滤路信号,顺手挡掉最常见的)
_COMMON = {
    "THE", "AND", "FOR", "WITH", "FROM", "THIS", "THAT", "WHEN", "WHAT",
    "HOW", "ARE", "WAS", "NOT", "BUT", "ALL", "CAN", "HAS", "HAVE", "ITS",
    "INTO", "AFTER", "BEFORE", "DURING", "VALUE", "SETTING", "PAGE",
    "CHAPTER", "SECTION", "FIG", "TABLE", "REV", "PDF", "DOC", "TORR",
}


def _tokenize(text, *, max_tokens):
    """片段判型:先收型号/缩写类(高信号),再收首字母大写品牌词(低信号),
    保证截断时低信号的先被裁掉。返回规范化大写去重列表。"""
    frags = _FRAG.findall(text or "")
    mixed, brand = [], []
    for f in frags:
        has_d = any(c.isdigit() for c in f)
        has_l = any(c.isalpha() for c in f)
        u = f.upper()
        if has_d and has_l and not f[0].isdigit():      # D4i/I12/H4/E1;"4i"纯数字开头不要
            if u not in mixed:
                mixed.append(u)
        elif f.isupper() and len(f) >= 3 and u not in _COMMON:
            if u not in mixed:
                mixed.append(u)
        elif f[:1].isupper() and len(f) >= 3 and u not in _COMMON:
            if u not in brand:
                brand.append(u)
    out = mixed + [b for b in brand if b not in mixed]
    return out[:max_tokens]


def extract_device_tokens(text, *, max_tokens: int = 16):
    """查询侧:从问题抽型号 token,返回规范化大写列表(保序去重,截到 max_tokens)。"""
    return _tokenize(text, max_tokens=max_tokens)


def chunk_device_tags(payload_text: str, source_stem: str = "",
                      heading_path: str = "", *, max_tags: int = 48):
    """入库侧:给单个块计算 devices 标签 = 正文+标题+书名三处 token 的并集。

    正文为主(型号多在正文而非书名,书名-only 会在仿真里误杀工艺类金标),
    书名/标题作补充(设备手册章节拆分文件如 "H4 - Description" 靠它兜住)。
    空 tags 返回 None,调用方不打该字段。"""
    tags = _tokenize(
        f"{payload_text or ''}\n{heading_path or ''}\n{source_stem or ''}",
        max_tokens=max_tags)
    return tags or None
