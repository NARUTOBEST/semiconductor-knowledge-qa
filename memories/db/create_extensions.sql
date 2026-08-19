-- =====================================================================
-- 三个库都必须安装 pgvector 扩展(各自独立实例,禁止跨库 join)
-- 在三个 database 上分别执行本文件:
--   psql -d agent_working_db -f create_extensions.sql
--   psql -d agent_short_db   -f create_extensions.sql
--   psql -d agent_long_db    -f create_extensions.sql
-- =====================================================================
CREATE EXTENSION IF NOT EXISTS vector;
