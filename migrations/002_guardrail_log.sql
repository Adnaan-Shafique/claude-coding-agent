-- Adds the guardrail log to a database that already has the post-001 schema.
-- Fresh installs get this from init.sql and do not need to run it.
--
--   psql -h <host> -U <user> -d <database> -v ON_ERROR_STOP=1 -f migrations/002_guardrail_log.sql

BEGIN;

-- ------------------------------------------------------------------
-- Guardrail log: one row per request that was blocked or flagged.
--
-- user_id is ON DELETE SET NULL rather than CASCADE, and the username is
-- denormalised alongside it, so deleting an account does not erase the
-- record of what it did. question is stored REDACTED — see
-- guardrails.redact_secrets() — because a log of blocked credential
-- pastes would otherwise be the densest collection of secrets in the
-- system. Uploaded file content is never stored, only its name.
-- ------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS coding_agent_schema.blocked_queries (
    id              SERIAL PRIMARY KEY,
    user_id         INTEGER REFERENCES coding_agent_schema.users(id) ON DELETE SET NULL,
    username        TEXT NOT NULL,
    conversation_id INTEGER REFERENCES coding_agent_schema.conversations(id) ON DELETE SET NULL,
    action          TEXT NOT NULL CHECK (action IN ('blocked', 'flagged')),
    category        TEXT NOT NULL,
    rules           TEXT NOT NULL,
    detail          TEXT,
    question        TEXT NOT NULL,
    file_name       TEXT,
    client_ip       TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS blocked_queries_created_idx
    ON coding_agent_schema.blocked_queries (created_at DESC);
CREATE INDEX IF NOT EXISTS blocked_queries_user_idx
    ON coding_agent_schema.blocked_queries (user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS blocked_queries_category_idx
    ON coding_agent_schema.blocked_queries (category, created_at DESC);

COMMIT;
