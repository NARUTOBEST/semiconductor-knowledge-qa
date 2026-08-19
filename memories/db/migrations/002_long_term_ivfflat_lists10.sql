-- database: agent_long_db
-- ivfflat lists 从 100 降到 10:每用户长期记忆仅几十到几百条,lists=100 偏高,
-- 小数据量下优化器易放弃索引走顺序扫描。ivfflat 的 lists 无法 ALTER,需重建索引。
-- 仅由 migrate runner 应用一次(记录在 schema_migrations);新库已在 02 DDL 用 lists=10。

DROP INDEX IF EXISTS idx_long_term_memories_embedding;

CREATE INDEX idx_long_term_memories_embedding
    ON long_term_memories
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 10);

ANALYZE long_term_memories;
