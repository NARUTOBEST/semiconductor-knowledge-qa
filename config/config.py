# -*- coding: utf-8 -*-
"""阶段二配置:路径、OpenAI 兼容 API、Qdrant、嵌入模型、分块参数。"""
import os
import json

# === 加载 env/env.env(.env 配置)===
_ENV_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "env", "env.env")
try:
    from dotenv import load_dotenv
    load_dotenv(_ENV_FILE)
except Exception:
    # dotenv 不可用时手动解析(仅 KEY=value,忽略注释/空行;不覆盖已存在的环境变量)
    if os.path.exists(_ENV_FILE):
        with open(_ENV_FILE, encoding="utf-8") as _f:
            for _line in _f:
                _line = _line.strip()
                if not _line or _line.startswith("#") or "=" not in _line:
                    continue
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

# === 路径 ===
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 自动定位项目根(stage2 的上级),跨平台
SRC_ROOT     = os.getenv("SRC_ROOT", r"D:\180-半导体设备相关资料！")   # 资料库根(已迁至 D 盘,可用环境变量覆盖)
CLEAN_ROOT   = r"D:\清洗文件\pdf"                                   # MinerU 产物根(已迁至 D 盘;pdf/ 下并列各类目,顶层预留 video/cad 等)
QDRANT_PATH  = os.path.join(PROJECT_ROOT, "qdrant_db")                 # Qdrant 本地存储(文件模式)
QDRANT_URL   = os.getenv("QDRANT_URL", "")                             # Qdrant 服务器地址(空=本地文件模式,设了=服务器模式)
RAG_DIR      = os.path.join(PROJECT_ROOT, "RAG")

# === 品牌 / 产品名(中性默认;部署时可用环境变量覆盖为具体公司名,如"XX 半导体设备知识库")===
BRAND_NAME  = os.getenv("BRAND_NAME", "半导体设备知识问答系统").strip() or "半导体设备知识问答系统"
BRAND_SHORT = os.getenv("BRAND_SHORT", "设备知识库助手").strip() or "设备知识库助手"

# === OpenAI 兼容 API --从 env/env.env 读取,缺失项回退 Claude Code 云接入(settings.json) ===
def _claude_llm_env() -> dict:
    """从 ~/.claude/settings.json 读取火山 ARK codingplan 云接入配置。

    该文件是 Claude Code 全局配置,取 ANTHROPIC_BASE_URL / ANTHROPIC_AUTH_TOKEN
    两项供本项目对话 LLM 复用(密钥不入本项目文件)。读取失败返回空 dict,
    不阻断启动——缺配置时调用 LLM 会报错,便于尽早暴露。
    """
    try:
        from pathlib import Path
        p = Path.home() / ".claude" / "settings.json"
        if not p.exists():
            return {}
        env = (json.loads(p.read_text(encoding="utf-8")) or {}).get("env") or {}
        base = env.get("ANTHROPIC_BASE_URL", "").rstrip("/")
        # settings.json 里是 Anthropic 通道根(/api/coding);OpenAI SDK 走 /v1/chat/completions
        if base and not base.endswith("/v1"):
            base = base + "/v1"
        return {
            "base": base,
            "key": env.get("ANTHROPIC_AUTH_TOKEN", ""),
        }
    except Exception:
        return {}


_CLAUDE_LLM = _claude_llm_env()

# 对话推理 (云:火山 ARK codingplan 接口,OpenAI 兼容;env 显式配置优先,其次 settings.json 回退)
OPENAI_BASE_URL      = os.getenv("OPENAI_BASE_URL", "") or _CLAUDE_LLM.get("base", "")  # 对话 API 基础地址
OPENAI_API_KEY       = os.getenv("OPENAI_API_KEY", "") or _CLAUDE_LLM.get("key", "")   # 对话 API 密钥
OPENAI_TEXT_MODEL    = os.getenv("OPENAI_TEXT_MODEL", "") or "deepseek-v4-flash"       # 中等/复杂推理主模型(ReAct / Plan-and-Execute / grounding)
OPENAI_FALLBACK_MODEL = os.getenv("OPENAI_FALLBACK_MODEL", "") or "doubao-seed-2.0-lite"  # 主模型不可用时的备用模型

# ============================================================
# 可拔插 LLM 网关(租用 GPU 服务器上的自建 vLLM/LiteLLM,OpenAI 兼容)
# ------------------------------------------------------------
# 配置 LLM_GATEWAY_BASE_URL 后,所有 LLM 调用改走自建网关(自有 GPU 算力,
# 不限流、不与他人争抢配额);未配置时回退到上方云端 ARK 接口。
# 业务代码只用 OpenAI 协议 + 逻辑模型名(main/light),换 GPU 服务器只需
# 改这两个环境变量,业务零改动。
LLM_GATEWAY_BASE_URL = os.getenv("LLM_GATEWAY_BASE_URL", "").strip()
LLM_GATEWAY_API_KEY  = os.getenv("LLM_GATEWAY_API_KEY", "").strip()
LLM_GATEWAY_ACTIVE   = bool(LLM_GATEWAY_BASE_URL)

# 网关上的逻辑模型名(由 LiteLLM 映射到 GPU 上的具体模型);仅网关模式使用。
GATEWAY_MODEL_MAIN  = os.getenv("GATEWAY_MODEL_MAIN", "") or "main"
GATEWAY_MODEL_LIGHT = os.getenv("GATEWAY_MODEL_LIGHT", "") or "light"

# 有效端点/密钥/模型:网关优先,缺省回退云端(各处 LLM 客户端统一读这几个)。
EFFECTIVE_LLM_BASE_URL = LLM_GATEWAY_BASE_URL or OPENAI_BASE_URL
EFFECTIVE_LLM_API_KEY  = LLM_GATEWAY_API_KEY or OPENAI_API_KEY
EFFECTIVE_FALLBACK_MODEL = (
    GATEWAY_MODEL_LIGHT if LLM_GATEWAY_ACTIVE else OPENAI_FALLBACK_MODEL
)

# 语义别名:按任务复杂度选模型
#   MODEL_MAIN  - 偏复杂的推理任务;MODEL_LIGHT - 简单/轻量任务(分类、改写、摘要、切分、守门)
MODEL_MAIN  = GATEWAY_MODEL_MAIN if LLM_GATEWAY_ACTIVE else OPENAI_TEXT_MODEL
MODEL_LIGHT = (GATEWAY_MODEL_LIGHT if LLM_GATEWAY_ACTIVE
               else (os.getenv("MODEL_LIGHT", "") or "doubao-seed-2.0-lite"))

# ------------------------------------------------------------
# 经济模式(ECONOMY_MODE):回退云端(单一受限账号)时自动开启。
# 目标:把"每次请求的 LLM 调用数"砍到最少,避免单账号限流/阻塞——
# 非关键的旁路 LLM(质检/覆盖判定/查询改写/路由分类/后台升迁)默认关闭,
# 只保留面向用户作答的核心调用;确定性的引用格式校验仍然保留。
# 任一开关都可用对应环境变量显式覆盖(1 开 / 0 关)。
_econ_env = os.getenv("ECONOMY_MODE", "").strip().lower()
if _econ_env in ("", "auto"):
    ECONOMY_MODE = not LLM_GATEWAY_ACTIVE     # auto:无网关即经济
else:
    ECONOMY_MODE = _econ_env in ("1", "true", "yes", "on")


def _feature_on(env_key: str, default_on: bool = True, *, off_in_economy: bool = False) -> bool:
    """读布尔特性开关:env 显式设置优先;否则经济模式下 off_in_economy 的特性默认关。"""
    raw = os.getenv(env_key)
    if raw is not None:
        return raw.strip().lower() not in ("0", "false", "no", "off")
    if off_in_economy and ECONOMY_MODE:
        return False
    return default_on


# 路由分类 LLM(经济模式关→规则预筛,未命中规则一律兜底 react,安全但多走 ReAct)
ROUTER_LLM_ENABLED        = _feature_on("ROUTER_LLM_ENABLED", True, off_in_economy=True)

# ---- 两级范式:按任务形态路由的模型 ----
# L1 simple=闲聊单轮直答(轻量,无工具);L2 react=知识问答 ReAct 检索循环。
TIER_MODEL_SIMPLE     = os.getenv("TIER_MODEL_SIMPLE", "") or MODEL_LIGHT
TIER_MODEL_REACT      = os.getenv("TIER_MODEL_REACT", "") or MODEL_MAIN

# ---- 复杂度路由器(阶段 3)----
ROUTER_TIMEOUT          = float(os.getenv("ROUTER_TIMEOUT", "8"))        # 分类 LLM 调用超时(s)
ROUTER_CONFIDENCE_MIN   = float(os.getenv("ROUTER_CONFIDENCE_MIN", "0.6"))  # 低于此置信度兜底 react
ROUTER_SHORT_LEN        = int(os.getenv("ROUTER_SHORT_LEN", "6"))        # 不超过该长度且无领域术语/复杂特征 -> 规则预筛 simple

# ---- 澄清反问(ambiguity -> ask back)----
# 路由前判定问题是否缺少关键实体(设备/型号/指代无法消解),命中则先反问而非硬答。
CLARIFY_ENABLED         = os.getenv("CLARIFY_ENABLED", "1") not in ("0", "false", "False")
CLARIFY_TIMEOUT         = float(os.getenv("CLARIFY_TIMEOUT", "8"))       # 澄清判定 LLM 调用超时(s)

# ---- 各推理范式运行参数(均可经环境变量覆盖)----
# 不启用独立旁路 LLM(无独立改写/grounding/coverage 节点):react 循环内由模型自行
# 决定检索与改写;质检门判空答案 + react 低置信 + simple↔react 一级升级。
def _int(env_key, default):
    return int(os.getenv(env_key, str(default)))


def _float(env_key, default):
    return float(os.getenv(env_key, str(default)))


TIER_CONFIG = {
    "simple": {
        "model": TIER_MODEL_SIMPLE,
        "max_steps": _int("TIER_SIMPLE_MAX_STEPS", 1),       # 单轮直答,无工具循环
        "max_total_seconds": _int("TIER_SIMPLE_MAX_TOTAL_SECONDS", 20),
    },
    # raglite 快路径:单一事实点 1 次检索 + 1 次主模型流式作答,不走 ReAct 循环。
    # 线上 tier 事件仍发 "react"(附 path=raglite),eval tier_ok / 前端零改动。
    "raglite": {
        "model": TIER_MODEL_REACT,
        "max_steps": 1,
        "max_total_seconds": _int("RAGLITE_MAX_TOTAL_SECONDS", 25),
    },
    "react": {
        "model": TIER_MODEL_REACT,
        # 2 = 1 次工具步 + 1 次强制作答步。复杂问题走 react 的占比已大幅下降,
        # 循环内 requery 由 reflect_node 在 step<max_steps-1 时触发(2 步时自然禁用)。
        "max_steps": _int("TIER_REACT_MAX_STEPS", 2),
        "max_total_seconds": _int("TIER_REACT_MAX_TOTAL_SECONDS", 25),
    },
}

# ---- raglite 快路径 ----
RAGLITE_ENABLED             = _feature_on("RAGLITE_ENABLED", True)
RAGLITE_MAX_TOTAL_SECONDS   = _int("RAGLITE_MAX_TOTAL_SECONDS", 25)  # 单次执行时长预算
RAGLITE_MAX_QUESTION_LEN    = _int("RAGLITE_MAX_QUESTION_LEN", 60)   # 超长问题走 react
RAGLITE_SEARCH_K            = _int("RAGLITE_SEARCH_K", 5)            # 检索条数(sources 事件用)
RAGLITE_SEARCH_SCORE_RATIO  = _float("RAGLITE_SEARCH_SCORE_RATIO", 0.4)
RAGLITE_PROMPT_CHUNKS       = _int("RAGLITE_PROMPT_CHUNKS", 3)       # 喂给模型的前几块(控制 prefill)
RAGLITE_CHUNK_CHARS         = _int("RAGLITE_CHUNK_CHARS", 800)       # 每块截断字符
# 答案解码上限:GLM 纯思考模型的 reasoning 计入 max_tokens(实测思考可烧
# 300-1500 token),上限太低会被思考耗尽 → 正文 0 token 空答案(实测 512 恰好
# 整 = reasoning 烧满)。放宽到 2000 给"思考+正文"留足空间,失控由看门狗兜底。
RAGLITE_ANSWER_MAX_TOKENS   = _int("RAGLITE_ANSWER_MAX_TOKENS", 2000)
# 投机检索:进入非 simple 路径时后台预发 search_text,与路由/分类重叠省 1.5~2.5s
RAGLITE_SPECULATIVE_SEARCH  = _feature_on("RAGLITE_SPECULATIVE_SEARCH", True)
# raglite 自适应重查:首轮 rerank 顶分低于 RETRIEVAL_CONFIDENT_SCORE 时,轻模型
# 改写查询再检索一次(与首轮按 chunk_id 去重合并,取分高者)。单步路径没有
# react 循环内的换词重查,这是其唯一的检索失败恢复手段;高置信题零开销。
RAGLITE_ADAPTIVE_REQUERY    = _feature_on("RAGLITE_ADAPTIVE_REQUERY", True)
# react 终答(不绑工具的那次调用)解码上限;工具调用步不设限防截断 tool-call JSON。
# 同上:GLM reasoning 计入上限,600 会被思考烧光出空答案,放宽到 2000。
REACT_ANSWER_MAX_TOKENS     = _int("REACT_ANSWER_MAX_TOKENS", 2000)
REACT_PRE_SEARCH             = _feature_on("REACT_PRE_SEARCH", True)  # react 首轮注入投机检索结果
# 单次流式调用总时长看门狗(秒):思考模型流式 reasoning 持续到达会绕过
# STREAM_TIMEOUT 空闲超时(实测单步思考可拖 5 分钟+),到点强制断流按截断处理
REACT_STREAM_DEADLINE_S      = _int("REACT_STREAM_DEADLINE_S", 45)
LLM_REACT_NO_THINK          = _feature_on("LLM_REACT_NO_THINK", True)  # react 终答关思考(忠实性由 grounding 校验兜底)
# 终答 grounding 后置校验:整段到手后用 light 模型逐句判"能否被检索资料蕴含",
# 删掉无支撑句再下发。校验调用套熔断器(见 ReAct/support/grounding.py)。
GROUNDING_CHECK             = _feature_on("GROUNDING_CHECK", True)
GROUNDING_TIMEOUT           = float(os.getenv("GROUNDING_TIMEOUT", "8"))    # 单次校验 LLM 超时(秒)
GROUNDING_BREAKER_THRESHOLD = _int("GROUNDING_BREAKER_THRESHOLD", 3)        # 连续失败→熔断 OPEN
GROUNDING_BREAKER_COOLDOWN  = float(os.getenv("GROUNDING_BREAKER_COOLDOWN", "60"))  # OPEN 冷却秒数,冷却后半开试探
GROUNDING_REMOVAL_CAP       = float(os.getenv("GROUNDING_REMOVAL_CAP", "0.4"))  # 判 false 句占比超此值→置信度不足
# 置信度门控(置信度 = 校验通过句占比):低于此线不下发模型答案,替换为
# "引导人工翻阅手册/联系 FAE"的提示文本(run20 策略:低置信答案宁可不答)。
GROUNDING_MIN_CONFIDENCE    = float(os.getenv("GROUNDING_MIN_CONFIDENCE", "0.6"))
# raglite 快路径也套 grounding 门控:token 缓冲到整段生成完再校验,低置信同样
# 替换为人工引导(0=回退旧直发行为,不校验)。
RAGLITE_GROUNDING_CHECK     = _feature_on("RAGLITE_GROUNDING_CHECK", True)
LLM_ENABLE_THINKING         = _feature_on("LLM_ENABLE_THINKING", False)  # Qwen3 思考模式;默认关(延迟),置 1 回退
# 融合分类:1=clarify+router 合并为单次 light 调用;0=回退旧两段式
ROUTER_FUSED                = _feature_on("ROUTER_FUSED", True)
# 质检低置信处理:1=整轮 redo(旧行为,+12s);0=passed+可见警告(循环内
# reflect_node 已做自适应 requery,整轮 redo 与之重复,是 42s 均耗的主要放大器)
QC_LOW_CONF_REDO            = _feature_on("QC_LOW_CONF_REDO", False)

# ---- 自适应检索(agentic:低置信时在循环内换关键词再检索)----
# rerank cross-encoder 相关分达到该值视为"命中对口资料";低于它则提示模型改写再查,
# 且质检门判 react 答案为低置信(触发至多 1 次换 thread 重进)。阈值按经验给默认值,
# 上线后用离线 eval(eval/run_eval.py)的分数分布校准。
RETRIEVAL_CONFIDENT_SCORE = _float("RETRIEVAL_CONFIDENT_SCORE", 0.5)
# 总开关:关闭后回到"只跑 max_steps、不注入低置信提示、质检只判空"的旧行为。
REACT_ADAPTIVE_RETRIEVAL = os.getenv(
    "REACT_ADAPTIVE_RETRIEVAL", "1").strip().lower() not in ("0", "false", "no", "off")

# 入库管线图片描述(RAG 清洗时为图块生成描述文本):火山方舟 Doubao-Seed-2.0-mini 推理点。
# 与对话模型同密钥(通用 key);未单独配置时回退到 OPENAI_BASE_URL / OPENAI_API_KEY。
OPENAI_VISION_BASE_URL = os.getenv("OPENAI_VISION_BASE_URL", OPENAI_BASE_URL)   # 方舟 API 基础地址
OPENAI_VISION_API_KEY  = os.getenv("OPENAI_VISION_API_KEY", OPENAI_API_KEY)     # 方舟密钥(默认通用 key)
OPENAI_VISION_MODEL    = os.getenv("OPENAI_VISION_MODEL", "") or "ep-20260728200449-fntbv"  # Doubao-Seed-2.0-mini 推理点
OPENAI_VISION_EXTRA    = json.loads(os.getenv("OPENAI_VISION_EXTRA", '{"thinking":{"type":"disabled"}}'))  # 关闭 thinking 省 token

# === 嵌入模型 ===
TEXT_EMBED_MODEL   = "BAAI/bge-m3"                           # dense 1024 + sparse
IMAGE_EMBED_MODEL  = "sentence-transformers/clip-ViT-B-32"  # CLIP 512
EMBED_DEVICE       = os.getenv("EMBED_DEVICE", "cpu")       # 嵌入模型设备:cpu(给 LLM 腾 GPU)/ cuda

# === Qdrant collection ===
TEXT_COLLECTION  = "ald_text"
IMAGE_COLLECTION = "ald_image"
TEXT_DENSE_DIM   = 1024
IMAGE_DENSE_DIM  = 512

# === 图片对象存储(火山引擎 TOS,S3 兼容;未配置则检索结果回退本地绝对路径) ===
# 检索结果把 payload 里 D:\清洗文件 下的图片绝对路径,转成 TOS 私有桶的"时效签名 HTTPS 链接"。
# 这样部署到虚拟机时无需复制清洗图片目录,前端/视觉模型直接用签名 URL 取图。
CLEAN_FILES_ROOT  = os.getenv("CLEAN_FILES_ROOT", r"D:\清洗文件")  # 图片文件根(payload 路径此前缀)
TOS_ENDPOINT      = os.getenv("TOS_ENDPOINT", "")        # 例:tos-cn-beijing.volces.com
TOS_REGION        = os.getenv("TOS_REGION", "")          # 例:cn-beijing
TOS_BUCKET        = os.getenv("TOS_BUCKET", "")          # 存储桶名
TOS_ACCESS_KEY    = os.getenv("TOS_ACCESS_KEY", "")
TOS_SECRET_KEY    = os.getenv("TOS_SECRET_KEY", "")
TOS_URL_EXPIRES   = int(os.getenv("TOS_URL_EXPIRES", "3600"))   # 签名链接有效期(秒),默认 1 小时
TOS_USE_PATH_STYLE = os.getenv("TOS_USE_PATH_STYLE", "0") == "1"  # MinIO 设 1;TOS 用虚拟主机风格(0)
TOS_KEY_PREFIX    = os.getenv("TOS_KEY_PREFIX", "qingxi").strip("/")  # 对象 key 前缀(桶内子目录)
TOS_ENABLED       = bool(TOS_ENDPOINT and TOS_BUCKET and TOS_ACCESS_KEY and TOS_SECRET_KEY)

# === 分块参数 ===
# 数值据已清洗语料实测:段落 P95≈713 字,max=800 可让 ~96% 段落整块不拆;
# target=500 为嵌入甜点(段落中位仅 45 字,需打包)。表行 P99≈259,不会被拆。
CHUNK_TARGET  = 500                       # 目标字数
CHUNK_MAX     = 800                       # 超过则按句/行切;单行仍超则确定性硬切(不调 LLM)
CHUNK_MIN     = 150                       # 小于则并入同段下一 chunk
FILTER_TYPES  = {"header", "footer", "page_number"}
IMAGE_TYPES   = {"image", "chart"}        # 进图库的类型(表格留 HTML,公式留 LaTeX)

# === 缓存 ===
OPENAI_DESC_FILENAME = "_ark_descriptions.json"   # per-PDF 图描述缓存,放 auto/ 下

# === 认证(JWT + SQLite)===
JWT_SECRET       = os.getenv("JWT_SECRET", "change-me-in-production")
JWT_ALGORITHM    = "HS256"
JWT_EXPIRE_HOURS = int(os.getenv("JWT_EXPIRE_HOURS", "24"))
AUTH_DB_PATH     = os.getenv("AUTH_DB_PATH", "") or os.path.join(PROJECT_ROOT, "auth.db")
# 首个管理员引导令牌:注册首个用户时,请求头 X-Bootstrap-Token 须与此一致才授予 admin;
# 留空(默认)则任何人都无法通过自助注册获得 admin(首个用户也是普通 user),
# 杜绝"抢先注册即提权"。部署方注入一次性令牌,创建管理员后即可停用。
ADMIN_BOOTSTRAP_TOKEN = os.getenv("ADMIN_BOOTSTRAP_TOKEN", "")
# /metrics 内部抓取密钥:监控系统以请求头 X-Internal-Token 携带此值可免登录拉取指标;
# 留空则只允许 admin JWT 访问(指标含每用户用户名等敏感数据,默认不匿名开放)。
METRICS_INTERNAL_TOKEN = os.getenv("METRICS_INTERNAL_TOKEN", "")

# === 速率限制 ===
# 50 用户压测可经环境变量覆盖(如 RATE_LIMIT_GLOBAL_CONCURRENT=16)
RATE_LIMIT_PER_USER_CONCURRENT = int(os.getenv("RATE_LIMIT_PER_USER_CONCURRENT", "1"))   # 每用户同时请求数(防并发刷)
RATE_LIMIT_GLOBAL_CONCURRENT = int(os.getenv("RATE_LIMIT_GLOBAL_CONCURRENT", "8"))   # 全局并发上限(10用户设8,留余量给VL模型)
RATE_LIMIT_QUEUE_TIMEOUT = int(os.getenv("RATE_LIMIT_QUEUE_TIMEOUT", "30"))      # 排队等待超时(秒)

# === 重排(Cross-Encoder)===
RERANK_MODEL       = "BAAI/bge-reranker-v2-m3"
RERANK_RECALL_K    = int(os.getenv("RERANK_RECALL_K", "48"))   # 送入重排的候选数(BGE-m3 多取;48 候选 GPU 重排 <20ms)
RERANK_POOL_MIN    = int(os.getenv("RERANK_POOL_MIN", "8"))    # ratio 过滤后重排池保底(RRF 分 Top-Heavy,ratio 可能把 80 压到 1~2 个)
RERANK_MAX_CONTENT = 800     # 重排时每条文档最大字符数(与 CHUNK_MAX=800 对齐,略留余量)
# 动态截断:重排后按绝对分截断(而非固定 top-k)。rerank 分 0~1(normalize=True),
# τ=0.5 为 5×100 评测扫描的均衡值(见 eval/results_concurrent/tau_sweep.jsonl)。
RERANK_TAU    = float(os.getenv("RERANK_TAU", "0.3"))  # τ 复扫(250 题,GPU):0.4 时 R@3=上限 0.836;0.3 兼容表格块(重排低分高相关,run20 id10 探针)
RERANK_MAX_K  = int(os.getenv("RERANK_MAX_K", "8"))    # 动态截断上限(4→8:run20 id10 探针显示目标表格块排在 4 名之外;P@3 分母不受影响,后端 prompt 自行截断)
RERANK_MIN_K  = int(os.getenv("RERANK_MIN_K", "3"))    # 保底 3 块:P@k 分母固定 k,pool<3 直接压死精度
# sparse 词面保底:重排后把 sparse top-N 词面命中块追加到结果尾部(去重,评分<τ)。
# 救 cross-encoder 对表格/码表块的系统性低分(run20 id10 探针:答案表 dense 第10、重排<0.3)。
RETRIEVAL_SPARSE_RESCUE    = os.getenv("RETRIEVAL_SPARSE_RESCUE", "1").strip().lower() not in ("0", "false", "no", "off")
RETRIEVAL_SPARSE_RESCUE_K  = int(os.getenv("RETRIEVAL_SPARSE_RESCUE_K", "2"))

# === 上下文管理(Context Management) ===
# 控制发给 LLM 的 messages 体积。当前仅截断过长的工具结果。
CONTEXT_TOOL_RESULT_MAX_CHARS = int(
    os.getenv("CONTEXT_TOOL_RESULT_MAX_CHARS", "800"))  # 工具结果回传 LLM 时的截断字符数

# === 检索微服务 ===
RETRIEVAL_SERVICE_URL = os.getenv("RETRIEVAL_SERVICE_URL", "http://127.0.0.1:8002")
# 服务间鉴权:非空时,检索微服务(:8002)所有端点(除 /health)要求请求头
# X-Internal-Token 与此值一致,否则 403(防内网任意主体投毒 Qdrant/打爆 CPU)。
# 留空 = 鉴权关闭(本地开发默认);主服务/MCP桥/embed_http/长期记忆等调用方
# 自动携带同值。另配合 RETRIEVAL_HOST=127.0.0.1 做网络层收口(容器内网部署)。
RETRIEVAL_INTERNAL_TOKEN = os.getenv("RETRIEVAL_INTERNAL_TOKEN", "")


def _bool_env(key, default=False):
    """读布尔型环境变量:1/true/yes/on 为 True,其余为 False。"""
    val = os.getenv(key)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


# === 熔断器(按工具维度)===
CIRCUIT_BREAKER_ENABLED = _bool_env("CIRCUIT_BREAKER_ENABLED", True)
# 连续失败达到该阈值则熔断(OPEN);FATAL 类错误不计入
CIRCUIT_BREAKER_FAILURE_THRESHOLD = int(
    os.getenv("CIRCUIT_BREAKER_FAILURE_THRESHOLD", "5"))
# OPEN 持续(秒)后半开试探一次
CIRCUIT_BREAKER_COOLDOWN_SECONDS = float(
    os.getenv("CIRCUIT_BREAKER_COOLDOWN_SECONDS", "30"))

# 同一轮 ReAct 内多个无依赖工具调用的并发上限(线程池 fan-out);1=串行
TOOL_MAX_PARALLEL = int(os.getenv("TOOL_MAX_PARALLEL", "4"))

# 工具故障自适应:把熔断/本轮故障的工具从 schema 摘掉并在 system 提示中告知替代策略
TOOL_HEALTH_ADAPT_ENABLED = os.getenv(
    "TOOL_HEALTH_ADAPT_ENABLED", "1") not in ("0", "false", "False")

# === Agent 记忆系统(三层)===
# 三层记忆落在不同后端:
#   - 工作记忆 = LangGraph checkpoint,用 langgraph-checkpoint-redis 的 RedisSaver(Redis);
#   - 短期记忆 = 会话事件流水(mem:* 键,Redis,见 memories/storage/short/short_term.py);
#   - 长期记忆 = 用户个人偏好,PostgreSQL + pgvector,按 user 哈希分表
#     (见 memories/storage/long/)。长期记忆是旁路增强:PG 不可用全程降级,不影响聊天。
#
# 运行环境:
#   - 本地开发(Windows):Redis 跑在 WSL2,经 127.0.0.1:6379 访问。
#     风险:WSL2 重启后 netsh 端口转发规则可能失效,需重设 portproxy;生产勿依赖 WSL2 转发。
#   - Ubuntu Server / docker-compose:Redis 为 compose 内 redis 服务,REDIS_HOST=redis。
REDIS_HOST     = os.getenv("REDIS_HOST", "127.0.0.1")
REDIS_PORT     = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")
REDIS_DB       = int(os.getenv("REDIS_DB", "0"))

# Redis 连接 URL(供 RedisSaver.from_conn_string 用)。显式 REDIS_URL 优先;
# 未设则由 host/port/password/db 拼出 redis://[:pwd@]host:port/db。
def _build_redis_url() -> str:
    explicit = os.getenv("REDIS_URL", "").strip()
    if explicit:
        return explicit
    auth = f":{REDIS_PASSWORD}@" if REDIS_PASSWORD else ""
    return f"redis://{auth}{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"

REDIS_URL = _build_redis_url()

# 短期记忆流水的滚动 TTL(天);与 lifecycle.DEFAULT_TTL_DAYS 对齐,可 env 覆盖。
MEMORY_TTL_DAYS = int(os.getenv("MEMORY_TTL_DAYS", "30"))

# === 追踪记录(trace 包;独立 Redis 键空间,供测试/运维复查 agent 工作流)===
# 与"短期对话记忆"分离:短期记忆只存 user/assistant 问答喂模型;追踪存储记录
# tool_call/tool_result/error/error_trace/done(含完整 trace),供事后复查。
TRACE_STORE_ENABLED = os.getenv("TRACE_STORE_ENABLED", "1") not in ("0", "false", "False")
TRACE_TTL_DAYS = int(os.getenv("TRACE_TTL_DAYS", "14"))
# 单条事件 payload 上限(字符);超长(如 done 的完整 trace)截断兜底,防 Redis 膨胀。
TRACE_MAX_PAYLOAD_CHARS = int(os.getenv("TRACE_MAX_PAYLOAD_CHARS", "20000"))
# 写入失败【先重试】:单条操作最多尝试次数(含首次)与重试退避基数(秒,指数退避)。
TRACE_RETRY_MAX = int(os.getenv("TRACE_RETRY_MAX", "3"))
TRACE_RETRY_BACKOFF = float(os.getenv("TRACE_RETRY_BACKOFF", "0.05"))
# 重试用尽后【熔断开路】:冷却期内所有追踪操作直接跳过(不碰 Redis、不阻塞聊天);
# 冷却后半开探测,成功即闭合恢复。
TRACE_CIRCUIT_COOLDOWN = float(os.getenv("TRACE_CIRCUIT_COOLDOWN", "30"))

# === 长期记忆(用户偏好;PostgreSQL + pgvector,按 user 哈希分表)===
# 旁路增强:开关关闭或 PG/嵌入服务不可用时,偏好不抽取、不注入,聊天不受影响。
LONG_MEM_ENABLED = os.getenv("LONG_MEM_ENABLED", "1") not in ("0", "false", "False")

# PG 连接:显式 LONG_PG_URI 优先;否则由 POSTGRES_* 拼出 postgresql://user:pwd@host:port/db。
LONG_PG_URI = os.getenv("LONG_PG_URI", "").strip()
_POSTGRES_HOST = os.getenv("POSTGRES_HOST", "127.0.0.1")
_POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
_POSTGRES_DB   = os.getenv("POSTGRES_DB", "memory_long")
_POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
_POSTGRES_PWD  = os.getenv("POSTGRES_PASSWORD", "")
if not LONG_PG_URI:
    _pg_auth = f"{_POSTGRES_USER}:{_POSTGRES_PWD}@" if _POSTGRES_PWD else f"{_POSTGRES_USER}@"
    LONG_PG_URI = f"postgresql://{_pg_auth}{_POSTGRES_HOST}:{_POSTGRES_PORT}/{_POSTGRES_DB}"

# 分表数:按 blake2b(username) % SHARD_COUNT 落到 long_mem_00..SHARD-1。改此值需重建表。
LONG_MEM_SHARD_COUNT = max(1, int(os.getenv("LONG_MEM_SHARD_COUNT", "16")))
# 每轮注入的语义相关偏好条数上限。
LONG_MEM_TOP_K = max(1, int(os.getenv("LONG_MEM_TOP_K", "5")))
# 无 key 的偏好按向量近邻去重:余弦距离 < (1 - 该阈值) 视为同一条(阈值=相似度)。
LONG_MEM_DUP_COSINE = float(os.getenv("LONG_MEM_DUP_COSINE", "0.9"))
# BGE-m3 dense 维度(与检索微服务 /embed_text 一致)。
LONG_MEM_EMBED_DIM = 1024

# ============================================================
# 记忆工作闭环(memory-loop):后台独立记忆图,答案定稿后经管道异步执行。
# ① 存储节点(短期事实表 + 升迁门写长期 PG;PG 宕机走本地 spool)
# ② 摘要压缩节点(阈值触发:session-memory.md + Auto-Compact)
# ③ 兜底维护(WAL 回填 + NULL 向量 backfill + PG spool 重放 + 记忆欠账补做 work_spool:
#    Redis 宕机期的事实表欠账 / 升迁门 LLM 宕机期的判定欠账,恢复后补跑完整沉淀)。
# 同会话任务管道内串行;下一轮请求入口 wait_idle 等待(有界超时兜底放行)。
# 全程不占请求流:done 立即发;旁路 LLM 在经济模式下默认关。
# ============================================================
MEM_LOOP_ENABLED = _feature_on("MEM_LOOP_ENABLED", True)  # 闭环总开关
# 入口等待门:下一轮请求最多等上一轮记忆链这么久(超时放行,幂等重做留给下一轮)
MEM_WAIT_IDLE_TIMEOUT     = float(os.getenv("MEM_WAIT_IDLE_TIMEOUT", "10.0"))
# 跨进程会话锁 TTL(多 worker 部署下同一会话记忆任务全局互斥;持有者崩溃后
# 该会话记忆最长被阻塞 TTL 秒,须大于单任务最长耗时含升迁门退避重试)
MEM_LOOP_LOCK_TTL         = float(os.getenv("MEM_LOOP_LOCK_TTL", "300"))

# ---- 节点一:短期事实表(Redis,与事件流水 mem:* 分离,前缀 memf:)----
MEM_FACT_TTL_SECONDS     = max(60, int(os.getenv("MEM_FACT_TTL_SECONDS", "86400")))   # TTL 24h
MEM_FACT_MAX_PER_THREAD  = max(10, int(os.getenv("MEM_FACT_MAX_PER_THREAD", "10000")))  # 容量上限
MEM_FACT_SEMANTIC_DEDUP  = _feature_on("MEM_FACT_SEMANTIC_DEDUP", False)  # embedding 语义去重(默认关,用规范化指纹)
MEM_FACT_DUP_COSINE      = float(os.getenv("MEM_FACT_DUP_COSINE", "0.9"))
MEM_FACT_DEDUP_SCAN      = max(1, int(os.getenv("MEM_FACT_DEDUP_SCAN", "50")))  # 语义去重比对近期条数

# ---- 存储节点:升迁门(LLM 判定是否升迁长期记忆)----
# 后台执行不占用户等待:超时/重试放宽回正常值(同步链时代的紧缩与
# MEM_CHAIN_BUDGET 整链预算一并退役);持续故障由熔断器兜底。
MEM_PROMOTE_ENABLED       = _feature_on("MEM_PROMOTE_ENABLED", True, off_in_economy=True)
MEM_PROMOTE_IMPORTANCE    = float(os.getenv("MEM_PROMOTE_IMPORTANCE", "0.6"))  # 升迁重要性阈值
MEM_RETRY_MAX             = max(0, int(os.getenv("MEM_RETRY_MAX", "2")))       # 升迁门 LLM 最多重试
MEM_RETRY_MAX_SLEEP       = float(os.getenv("MEM_RETRY_MAX_SLEEP", "2.0"))     # 指数退避 cap(秒)
MEM_BREAKER_FAIL_THRESHOLD = max(1, int(os.getenv("MEM_BREAKER_FAIL_THRESHOLD", "3")))
MEM_BREAKER_COOLDOWN      = float(os.getenv("MEM_BREAKER_COOLDOWN", "300"))     # 熔断冷却(秒)
MEM_GATE_LLM_TIMEOUT      = float(os.getenv("MEM_GATE_LLM_TIMEOUT", "6.0"))     # 升迁门单模型超时(s)

# ---- 节点二:会话摘要 session-memory.md + Auto-Compact ----
MEM_SESSION_DIR = os.getenv("MEM_SESSION_DIR", "").strip() or os.path.join(
    PROJECT_ROOT, "memories_data", "sessions")
MEM_SUMMARY_FIRST_TOKENS      = int(os.getenv("MEM_SUMMARY_FIRST_TOKENS", "10000"))     # 首摘阈值
MEM_SUMMARY_UPDATE_TOKEN_TOOL = int(os.getenv("MEM_SUMMARY_UPDATE_TOKEN_TOOL", "5000"))  # 更新条件A:token 增长
MEM_SUMMARY_UPDATE_TOOL_CALLS = int(os.getenv("MEM_SUMMARY_UPDATE_TOOL_CALLS", "3"))     # 更新条件A:工具调用增长
MEM_SUMMARY_UPDATE_TOKEN_TEXT = int(os.getenv("MEM_SUMMARY_UPDATE_TOKEN_TEXT", "2000"))  # 更新条件B:纯文本轮 token 增长
MEM_MODEL_CONTEXT_TOKENS      = int(os.getenv("MEM_MODEL_CONTEXT_TOKENS", "40960"))      # 模型上下文窗口
MEM_COMPACT_RESERVE_MIN       = int(os.getenv("MEM_COMPACT_RESERVE_MIN", "13000"))       # 预留缓冲下限
MEM_COMPACT_RESERVE_RATIO     = float(os.getenv("MEM_COMPACT_RESERVE_RATIO", "0.10"))    # 预留缓冲比例
MEM_KEEP_RECENT_DEFAULT       = int(os.getenv("MEM_KEEP_RECENT_DEFAULT", "20000"))       # compact 保留原文(默认)
MEM_KEEP_RECENT_MIN           = int(os.getenv("MEM_KEEP_RECENT_MIN", "10000"))
MEM_KEEP_RECENT_MAX           = int(os.getenv("MEM_KEEP_RECENT_MAX", "40000"))
MEM_MIN_TEXT_MESSAGES         = int(os.getenv("MEM_MIN_TEXT_MESSAGES", "5"))             # 保留区最少文本消息
MEM_RULE_KEEP_TURNS           = int(os.getenv("MEM_RULE_KEEP_TURNS", "6"))               # 规则截断降级保留轮次
MEM_SUMMARY_LLM_TIMEOUT       = float(os.getenv("MEM_SUMMARY_LLM_TIMEOUT", "8.0"))
MEM_SUMMARY_LLM_RETRIES       = max(0, int(os.getenv("MEM_SUMMARY_LLM_RETRIES", "2")))
MEM_SUMMARY_MAX_CHARS         = int(os.getenv("MEM_SUMMARY_MAX_CHARS", "4000"))          # 摘要硬截断
MEM_FILE_LOCK_TIMEOUT         = float(os.getenv("MEM_FILE_LOCK_TIMEOUT", "5.0"))         # 文件锁等待(s)
MEM_SUMMARY_VERSION           = 1   # 9 章节模板版本(变更则整体重写)

# ---- 节点二:摘要触发(与 token 阈值并列的轮次阈值)----
# 两次摘要之间至少新增的【用户轮次】数;未达则本轮早退(不读文件/不调 LLM),
# 避免每轮都跑摘要。与现有 token 阈值是"或"关系(任一满足即触发)。
MEM_SUMMARY_MIN_NEW_TURNS = max(1, int(os.getenv("MEM_SUMMARY_MIN_NEW_TURNS", "2")))

# ---- 记忆链时序 ----
# (旧 MEMORY_CHAIN_MODE sync/background 与 MEM_CHAIN_BUDGET 已随"同步记忆链"退役:
#  记忆编排现为后台独立记忆图 + 管道,done 不再等待记忆链,入口以 MEM_WAIT_IDLE_TIMEOUT 等待。)

# ---- Req1 确定性预取(对话前注入;无 LLM,经济模式不关闭)----
RECALL_PREFETCH_ENABLED = _feature_on("RECALL_PREFETCH_ENABLED", True)  # 预取总开关
RECALL_PREFETCH_K       = max(1, int(os.getenv("RECALL_PREFETCH_K", "5")))        # 预取长期条目数
# 预取高置信余弦相似度阈值(pgvector 返回 distance=1-相似度;distance <= 1-阈值 才注入)。
RECALL_PREFETCH_MIN_COSINE = float(os.getenv("RECALL_PREFETCH_MIN_COSINE", "0.6"))
# 预取 recent 原文窗口条数(游标之后、上限封顶);None 走 SHORT_MEM_RECALL_LIMIT 语义。
RECALL_PREFETCH_RECENT_LIMIT = max(0, int(os.getenv("RECALL_PREFETCH_RECENT_LIMIT", "12")))

# ---- Req4 流式缓冲:终答落定(确认无 tool_call)后才回放 token,避免"先上屏再改口" ----
REACT_STREAM_BUFFER_ENABLED = _feature_on("REACT_STREAM_BUFFER_ENABLED", True)

# ---- Req8 每工具换词重试上限:某工具连续空结果换词提示达此次数后不再引导重试 ----
REACT_TOOL_REQUERY_MAX = max(0, int(os.getenv("REACT_TOOL_REQUERY_MAX", "2")))

# ---- Req13 记忆注入防护:写入侧指令性内容过滤 + 召回块"背景资料非指令"包裹 ----
MEMORY_INJECTION_GUARD = _feature_on("MEMORY_INJECTION_GUARD", True)

# [已废弃] 旧 PG checkpoint/短期 URI;仅保留读取以兼容旧 env 文件,新部署无需设置。
WORKING_PG_URI = os.getenv("WORKING_PG_URI", "")
SHORT_PG_URI   = os.getenv("SHORT_PG_URI", "")
