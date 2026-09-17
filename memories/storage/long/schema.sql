-- 长期记忆(用户偏好)PostgreSQL schema。幂等(IF NOT EXISTS)。
-- 由 memories/storage/long/long_term.py:setup() 执行:
--   1) 下方 SHARD_TABLE 分隔标记【之前】的静态部分(扩展 + user_profile)直接执行一次;
--   2) 分隔标记【之后】的分片表模板对 long_mem_00 .. long_mem_(N-1) 逐张执行,
--      表名占位符由 setup 替换(表名来自程序自身的分片枚举,非用户输入)。
-- 注意:分隔标记字符串在本文件中只能出现一次(setup 用 str.partition 取首个出现位置),
--       且模板内除表名占位符外不得出现花括号(会被 str.format 误解析)。

CREATE EXTENSION IF NOT EXISTS vector;

-- 用户画像(非分片:每用户 1 行,稳定设置/一句话画像/关注点)
CREATE TABLE IF NOT EXISTS user_profile (
    username      TEXT PRIMARY KEY,
    display_prefs JSONB NOT NULL DEFAULT '{}'::jsonb,  -- 语言/语气/格式等稳定设置
    summary       TEXT NOT NULL DEFAULT '',            -- 一句话用户画像
    top_interests JSONB NOT NULL DEFAULT '[]'::jsonb,   -- 关注领域列表
    fact_count    INTEGER NOT NULL DEFAULT 0,           -- 名下偏好条目数
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- @@SHARD_TABLE@@
-- 下面是【单个分片表】的 DDL 模板,{table} 由 setup 逐分片替换为 long_mem_00/01/...
CREATE TABLE IF NOT EXISTS {table} (
    id            BIGSERIAL PRIMARY KEY,
    username      TEXT NOT NULL,
    category      TEXT NOT NULL,            -- language/role/expertise/topic_interest/tone/format/constraint/fact
    key           TEXT,                     -- 结构化去重槽位(可空;空=纯语义事实,按向量近邻去重)
    content       TEXT NOT NULL,            -- 面向 LLM 可读的偏好陈述
    embedding     vector(1024),             -- BGE-m3 dense 向量
    importance    REAL NOT NULL DEFAULT 0.5,-- 重要度 0~1
    status        TEXT NOT NULL DEFAULT 'active',  -- active / superseded / archived
    source_thread TEXT,                     -- 来源会话
    hit_count     INTEGER NOT NULL DEFAULT 0,
    last_hit_at   TIMESTAMPTZ,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
-- 结构化偏好去重:同 (用户,类别,key) 唯一(key 为 NULL 时不约束,改由向量近邻去重)
CREATE UNIQUE INDEX IF NOT EXISTS {table}_uniq_key
    ON {table} (username, category, key) WHERE key IS NOT NULL;
-- 语义召回:余弦距离近邻(ivfflat;空表/小数据走顺序扫描,功能正确)
CREATE INDEX IF NOT EXISTS {table}_emb
    ON {table} USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
-- 按用户 + 状态过滤
CREATE INDEX IF NOT EXISTS {table}_user_status
    ON {table} (username, status);
