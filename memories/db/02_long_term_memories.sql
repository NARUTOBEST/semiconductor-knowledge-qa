-- =====================================================================
-- 长期记忆库 agent_long_db:用户长期记忆表
-- 用途:跨 thread 沉淀事实/偏好/实体,带 pgvector 向量供语义召回
-- 运行方式:psql -d agent_long_db -f 02_long_term_memories.sql
-- 前提:本库已执行 CREATE EXTENSION IF NOT EXISTS vector;
-- 向量维度 1024 (BGE-m3 dense),与 config.LONG_MEMORY_EMBED_DIM 一致
-- =====================================================================

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS long_term_memories (
    id            BIGSERIAL PRIMARY KEY,
    user_id       TEXT NOT NULL,
    thread_id     TEXT,
    memory_type   TEXT NOT NULL DEFAULT 'fact',
    content       TEXT NOT NULL,
    embedding     vector(1024),
    meta          JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ivfflat 向量索引(余弦距离;BGE-m3 已 L2 归一,内积等价)
-- lists 为聚类中心数,经验值 ~ 行数/1000。当前每用户记忆量仅几十到几百条,
-- lists=10 更合适;数据量增长后可通过迁移调大。建索引后 ANALYZE 助优化器选计划。
CREATE INDEX IF NOT EXISTS idx_long_term_memories_embedding
    ON long_term_memories
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 10);
ANALYZE long_term_memories;

-- 按用户拉取/去重/过滤
CREATE INDEX IF NOT EXISTS idx_long_term_memories_user
    ON long_term_memories (user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_long_term_memories_type
    ON long_term_memories (memory_type);
CREATE INDEX IF NOT EXISTS idx_long_term_memories_thread
    ON long_term_memories (thread_id);

-- 内容去重(同一用户下相同文本只保留一条)
CREATE UNIQUE INDEX IF NOT EXISTS uq_long_term_memories_user_content
    ON long_term_memories (user_id, md5(content));

COMMENT ON TABLE long_term_memories IS
    '长期记忆:升迁流水线从短期流水萃取的事实/偏好;召回网关走向量检索+重排+裁剪后注入 LLM';
