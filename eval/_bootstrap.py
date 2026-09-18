# -*- coding: utf-8 -*-
"""eval 包公共引导:sys.path 注入、代理旁路、运行模式(在任何项目 import 前调用)。

用法:在 run_eval.py / make_groundtruth.py 顶部 `from eval import _bootstrap  # noqa`
(eval 作为包从项目根 `python -m eval.pipeline.pipeline_full100.run_eval` 运行;_bootstrap 先于业务 import 执行)。
"""
import os
import sys

_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in [
    _PROJECT,
    os.path.join(_PROJECT, "config"),
    os.path.join(_PROJECT, "RAG"),
    os.path.join(_PROJECT, "RAG", "pdf"),
    os.path.join(_PROJECT, "tools", "retrieval", "settings"),
    os.path.join(_PROJECT, "server"),
    os.path.join(_PROJECT, "context management"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# 评估要白盒直调 react_stream(不走 HTTP),但 judge 的 LLM / BGE-m3 嵌入仍可能走
# 本地网关(127.0.0.1:4000)与检索微服务(127.0.0.1:8002)。Clash 类代理会劫持
# 127.0.0.1,必须在 NO_PROXY 里旁路;云端 ARK(.volces.com)/hf-mirror 同样旁路。
_BYPASS = "127.0.0.1,localhost,.volces.com,.hf-mirror.com"
for _k in ("NO_PROXY", "no_proxy"):
    os.environ[_k] = (_BYPASS + "," + os.environ.get(_k, "")).strip(",")

# 评估跑完整链路(路由/质检/自适应检索),关闭经济模式。
os.environ.setdefault("ECONOMY_MODE", "0")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

PROJECT_ROOT = _PROJECT
EVAL_QA_PATH = os.path.join(_PROJECT, "web", "public", "eval-qa.json")
RESULTS_DIR = os.path.join(_PROJECT, "eval", "results")
