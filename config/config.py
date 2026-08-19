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
SRC_ROOT     = os.path.join(PROJECT_ROOT, "180-半导体设备相关资料！")   # 源 PDF 根
CLEAN_ROOT   = r"D:\清洗文件"                                       # MinerU 产物根(已迁至 D 盘省空间)
QDRANT_PATH  = os.path.join(PROJECT_ROOT, "qdrant_db")                 # Qdrant 本地存储(文件模式)
QDRANT_URL   = os.getenv("QDRANT_URL", "")                             # Qdrant 服务器地址(空=本地文件模式,设了=服务器模式)
RAG_DIR      = os.path.join(PROJECT_ROOT, "RAG")

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
OPENAI_TEXT_MODEL    = os.getenv("OPENAI_TEXT_MODEL", "") or "deepseek-v4-flash"       # 首选文本对话模型(medium ReAct)
OPENAI_FALLBACK_MODEL = os.getenv("OPENAI_FALLBACK_MODEL", "") or "doubao-seed-2.0-lite"  # 备用文本对话模型(主模型不可用时自动切换)

# ---- 三级范式:按复杂度路由的模型(阶段 2/3;9.1 补齐各 tier 的步数/时长/质检深度)----
TIER_MODEL_SIMPLE  = os.getenv("TIER_MODEL_SIMPLE", "") or "doubao-seed-2.0-lite"   # simple 单轮直答(lite)
TIER_MODEL_MEDIUM  = os.getenv("TIER_MODEL_MEDIUM", "") or OPENAI_TEXT_MODEL         # medium ReAct
TIER_MODEL_COMPLEX = os.getenv("TIER_MODEL_COMPLEX", "") or OPENAI_TEXT_MODEL        # complex Plan-and-Execute

# ---- 复杂度路由器(阶段 3)----
ROUTER_TIMEOUT          = float(os.getenv("ROUTER_TIMEOUT", "8"))        # 分类 LLM 调用超时(s)
ROUTER_CONFIDENCE_MIN   = float(os.getenv("ROUTER_CONFIDENCE_MIN", "0.6"))  # 低于此置信度兜底 medium
ROUTER_SHORT_LEN        = int(os.getenv("ROUTER_SHORT_LEN", "6"))        # 不超过该长度且无领域术语/复杂特征 -> 规则预筛 simple

# ---- 各推理范式运行参数(阶段 9.1/9.3,均可经环境变量覆盖)----
# quality_depth:
#   light   - simple:只判空/过短,不调 grounding LLM;
#   standard- medium:grounding 校验 + 失败时判断是否升级;
#   deep    - complex:grounding + 计划步骤覆盖度(uncovered_steps)联合判定。
def _int(env_key, default):
    return int(os.getenv(env_key, str(default)))


TIER_CONFIG = {
    "simple": {
        "model": TIER_MODEL_SIMPLE,
        "max_steps": _int("TIER_SIMPLE_MAX_STEPS", 1),       # 单轮直答,无工具循环
        "max_total_seconds": _int("TIER_SIMPLE_MAX_TOTAL_SECONDS", 20),
        "plan_enabled": False,                                # simple 不做规划
        "quality_depth": "light",
    },
    "medium": {
        "model": TIER_MODEL_MEDIUM,
        "max_steps": _int("TIER_MEDIUM_MAX_STEPS", 6),
        "max_total_seconds": _int("TIER_MEDIUM_MAX_TOTAL_SECONDS", 60),
        "plan_enabled": os.getenv("TIER_MEDIUM_PLAN_ENABLED", "1") not in ("0", "false", "False"),
        "quality_depth": "standard",
    },
    "complex": {
        "model": TIER_MODEL_COMPLEX,
        "max_steps": _int("TIER_COMPLEX_MAX_STEPS", 4),       # 计划步数上限(generate_plan max_steps)
        "max_total_seconds": _int("TIER_COMPLEX_MAX_TOTAL_SECONDS", 120),
        "plan_enabled": True,                                 # complex 必经规划
        "quality_depth": "deep",
    },
}

# 在线多模态 (vLLM 实例 2: Qwen2.5-VL-7B-Instruct-AWQ, 端口 8001)
OPENAI_VL_BASE_URL   = os.getenv("OPENAI_VL_BASE_URL", OPENAI_BASE_URL)   # 多模态 API 基础地址
OPENAI_VL_API_KEY    = os.getenv("OPENAI_VL_API_KEY", OPENAI_API_KEY)     # 多模态 API 密钥
OPENAI_VL_MODEL      = os.getenv("OPENAI_VL_MODEL", "")                   # 多模态对话模型

# 清洗图描述 (火山方舟 Doubao, 未配时回退到对话配置)
OPENAI_VISION_BASE_URL = os.getenv("OPENAI_VISION_BASE_URL", OPENAI_BASE_URL)   # 图描述 API 基础地址
OPENAI_VISION_API_KEY  = os.getenv("OPENAI_VISION_API_KEY", OPENAI_API_KEY)     # 图描述 API 密钥
OPENAI_VISION_MODEL    = os.getenv("OPENAI_VISION_MODEL", "")                   # 多模态推理端点
OPENAI_VISION_EXTRA    = json.loads(os.getenv("OPENAI_VISION_EXTRA", "{}"))     # 多模态附加参数

# === 嵌入模型 ===
TEXT_EMBED_MODEL   = "BAAI/bge-m3"                           # dense 1024 + sparse
IMAGE_EMBED_MODEL  = "sentence-transformers/clip-ViT-B-32"  # CLIP 512
EMBED_DEVICE       = os.getenv("EMBED_DEVICE", "cpu")       # 嵌入模型设备:cpu(给 LLM 腾 GPU)/ cuda

# === Qdrant collection ===
TEXT_COLLECTION  = "ald_text"
IMAGE_COLLECTION = "ald_image"
TEXT_DENSE_DIM   = 1024
IMAGE_DENSE_DIM  = 512

# === 分块参数 ===
CHUNK_TARGET  = 500                       # 目标字数
CHUNK_MAX     = 600                       # 超过则切;单 item 超且无边界 -> LLM 兜底
CHUNK_MIN     = 150                       # 小于则并入同段下一 chunk
FILTER_TYPES  = {"header", "footer", "page_number"}
IMAGE_TYPES   = {"image", "chart"}        # 进图库的类型(表格留 HTML,公式留 LaTeX)

# === 缓存 ===
OPENAI_DESC_FILENAME = "_ark_descriptions.json"   # per-PDF 图描述缓存,放 auto/ 下

# === 认证(JWT + SQLite)===
JWT_SECRET       = os.getenv("JWT_SECRET", "change-me-in-production")
JWT_ALGORITHM    = "HS256"
JWT_EXPIRE_HOURS = int(os.getenv("JWT_EXPIRE_HOURS", "24"))
AUTH_DB_PATH     = os.path.join(PROJECT_ROOT, "auth.db")

# === 速率限制 ===
RATE_LIMIT_PER_USER_CONCURRENT = 1   # 每用户同时请求数(防并发刷)
RATE_LIMIT_GLOBAL_CONCURRENT = 8   # 全局并发上限(10用户设8,留余量给VL模型)
RATE_LIMIT_QUEUE_TIMEOUT = 30      # 排队等待超时(秒)

# === 重排(Cross-Encoder)===
RERANK_MODEL       = "BAAI/bge-reranker-v2-m3"
RERANK_RECALL_K    = 20      # 召回候选数(BGE-m3 多取)
RERANK_MAX_CONTENT = 512     # 重排时每条文档最大字符数(截断)

# === 答案 Grounding(幻觉检测)===
GROUNDING_FAITHFULNESS_THRESHOLD = 0.5  # 忠实度检测:仅对 score>此值的 chunk 调 LLM 检查

# === 上下文管理(Context Management) ===
# 控制发给 LLM 的 messages 体积。当前仅截断过长的工具结果。
CONTEXT_TOOL_RESULT_MAX_CHARS = int(
    os.getenv("CONTEXT_TOOL_RESULT_MAX_CHARS", "800"))  # 工具结果回传 LLM 时的截断字符数

# === 检索微服务 ===
RETRIEVAL_SERVICE_URL = os.getenv("RETRIEVAL_SERVICE_URL", "http://127.0.0.1:8002")

# === Agent 记忆系统 (PostgreSQL 多库分表 + Redis 可选缓存)===
# 运行环境:Python 在 Windows,PG/Redis 在 WSL2,经 127.0.0.1 端口转发访问。
# 风险提示:WSL2 重启后 netsh 端口转发规则会失效,需重新执行 netsh interface portproxy 命令;
#         生产环境不要依赖 WSL2 端口转发方案。
# 三库独立,禁止跨库 join;每个库都需单独安装 pgvector 扩展。
WORKING_PG_URI = os.getenv("WORKING_PG_URI", "")  # 工作记忆库(LangGraph checkpoint 表)
SHORT_PG_URI   = os.getenv("SHORT_PG_URI", "")    # 短期记忆库(session_events 流水)
LONG_PG_URI    = os.getenv("LONG_PG_URI", "")     # 长期记忆库(long_term_memories + vector)

REDIS_HOST     = os.getenv("REDIS_HOST", "127.0.0.1")
REDIS_PORT     = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")
REDIS_DB       = int(os.getenv("REDIS_DB", "0"))
# 记忆召回网关
LONG_MEMORY_RECALL_TOP_K = int(os.getenv("LONG_MEMORY_RECALL_TOP_K", "5"))
LONG_MEMORY_RECALL_MAX_TOKENS = int(os.getenv("LONG_MEMORY_RECALL_MAX_TOKENS", "1200"))
# 长期记忆向量维度(BGE-m3 dense)
LONG_MEMORY_EMBED_DIM = 1024
# 升迁守门员模型:萃取前先判断本轮是否值得升迁,空则复用 OPENAI_TEXT_MODEL
LONG_MEMORY_GATE_MODEL = os.getenv("LONG_MEMORY_GATE_MODEL", "") or OPENAI_TEXT_MODEL
# 长期记忆语义去重阈值:写入前与同用户同类型既有记忆比对余弦相似度,>= 此值视为重复跳过(BGE-m3)
LONG_MEMORY_DEDUP_THRESHOLD = float(os.getenv("LONG_MEMORY_DEDUP_THRESHOLD", "0.92"))
