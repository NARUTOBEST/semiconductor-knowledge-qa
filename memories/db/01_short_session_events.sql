-- =====================================================================
-- 短期记忆库 agent_short_db:会话事件流水表
-- 用途:审计 / 回放 / 溯源,只追加(append-only),不 UPDATE/DELETE 业务数据
-- 运行方式:psql -d agent_short_db -f 01_short_session_events.sql
-- 前提:本库已执行 CREATE EXTENSION IF NOT EXISTS vector;
--       (短期表本身不用 vector 类型,但按约束每个库都装扩展)
-- =====================================================================

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS session_events (
    id           BIGSERIAL PRIMARY KEY,
    thread_id    TEXT NOT NULL,
    session_id   TEXT,
    user_id      TEXT,
    seq          INTEGER NOT NULL,
    event_type   TEXT NOT NULL,
    payload      JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 按 thread 顺序回放/溯源
CREATE INDEX IF NOT EXISTS idx_session_events_thread_seq
    ON session_events (thread_id, seq);

-- 按用户/会话筛选
CREATE INDEX IF NOT EXISTS idx_session_events_user
    ON session_events (user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_session_events_session
    ON session_events (session_id, created_at DESC);

-- 事件类型筛选(status/token/tool_call/tool_result/error/done ...)
CREATE INDEX IF NOT EXISTS idx_session_events_type
    ON session_events (event_type);

-- 升迁任务按时间窗口扫描未处理事件
CREATE INDEX IF NOT EXISTS idx_session_events_created
    ON session_events (created_at);

-- 同一 thread 内事件序号唯一,防止重复写入
CREATE UNIQUE INDEX IF NOT EXISTS uq_session_events_thread_seq
    ON session_events (thread_id, seq);

COMMENT ON TABLE session_events IS
    '短期记忆:会话事件流水,只追加;任务结束后由升迁流水线异步萃取到长期记忆库';
