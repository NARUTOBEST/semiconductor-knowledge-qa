-- =====================================================================
-- 短期记忆库 agent_short_db:升迁水位表
-- 用途:记录每个 thread 已升迁到长期记忆的最大 session_events.seq,
--       使 promote_thread 只升迁新增事件(增量升迁),避免每轮重放全量流水。
-- 运行方式:psql -d agent_short_db -f 03_short_promotion_watermark.sql
-- 幂等:CREATE TABLE IF NOT EXISTS + ON CONFLICT,可重复执行。
-- 说明:长期库 (user_id, md5(content)) 唯一索引仍是最终去重兜底;
--       水位只用于减少重复 LLM 调用与重放开销。
-- =====================================================================

CREATE TABLE IF NOT EXISTS promotion_watermark (
    thread_id       TEXT PRIMARY KEY,
    last_seq        INTEGER NOT NULL DEFAULT 0,
    promoted_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- 升迁连续失败计数(LLM 不可用等):达到上限后强制推进水位,避免 backlog 无限增长
    fail_count      INTEGER NOT NULL DEFAULT 0,
    last_attempt_at TIMESTAMPTZ
);

-- 幂等升级:旧库补列(已存在则跳过)
ALTER TABLE promotion_watermark ADD COLUMN IF NOT EXISTS fail_count INTEGER NOT NULL DEFAULT 0;
ALTER TABLE promotion_watermark ADD COLUMN IF NOT EXISTS last_attempt_at TIMESTAMPTZ;

COMMENT ON TABLE promotion_watermark IS
    '升迁水位:每 thread 已升迁到长期记忆的最大 session_events.seq;fail_count 为连续失败次数';
