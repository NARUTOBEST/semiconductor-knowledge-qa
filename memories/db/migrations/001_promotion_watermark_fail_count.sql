-- database: agent_short_db
-- 升迁水位表增加连续失败计数:防止 LLM 永久故障时 backlog 无限增长、每轮重放。
-- 幂等(ADD COLUMN IF NOT EXISTS),可重复执行。

ALTER TABLE promotion_watermark
    ADD COLUMN IF NOT EXISTS fail_count      INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS last_attempt_at TIMESTAMPTZ;

COMMENT ON TABLE promotion_watermark IS
    '升迁水位:每 thread 已升迁到长期记忆的最大 session_events.seq;fail_count 为连续失败次数';
