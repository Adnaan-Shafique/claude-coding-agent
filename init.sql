-- Forge Code — full schema for a NEW database, on PostgreSQL with pgvector.
--
--   psql -h <host> -U <user> -d <database> -v ON_ERROR_STOP=1 -f init.sql
--
-- Pass ON_ERROR_STOP=1 so psql aborts on the first problem instead of ploughing on.
--
-- Re-running this against a database it created is safe: every statement is guarded by
-- IF NOT EXISTS. Running it against a database created by an EARLIER version of the
-- schema is NOT safe to rely on — IF NOT EXISTS skips those tables entirely, so they
-- keep their old column sets and the new indexes then fail. The guard below detects
-- exactly that case and stops with a pointer to the migration instead.

CREATE SCHEMA IF NOT EXISTS coding_agent_schema;
CREATE EXTENSION IF NOT EXISTS vector;

-- ------------------------------------------------------------------
-- Guard: refuse to run on a database built by the pre-approval schema.
-- ------------------------------------------------------------------
DO $guard$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_schema = 'coding_agent_schema' AND table_name = 'users'
    ) AND NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'coding_agent_schema' AND table_name = 'users' AND column_name = 'is_active'
    ) THEN
        RAISE EXCEPTION
            'This database already has an OLDER version of the Forge Code schema. %',
            'Do not run init.sql against it — run migrations/001_auth_and_scoping.sql instead, '
            'which adds the new columns in place and keeps your existing accounts working.';
    END IF;
END
$guard$;

-- ------------------------------------------------------------------
-- Users. Accounts self-register but start inactive; an admin approves
-- them before login succeeds.
-- ------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS coding_agent_schema.users (
    id            SERIAL PRIMARY KEY,
    username      TEXT UNIQUE NOT NULL,
    email         TEXT,
    password_hash TEXT NOT NULL,
    is_active     BOOLEAN NOT NULL DEFAULT FALSE,
    is_admin      BOOLEAN NOT NULL DEFAULT FALSE,
    approved_by   INTEGER REFERENCES coding_agent_schema.users(id) ON DELETE SET NULL,
    approved_at   TIMESTAMPTZ,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Case-insensitive uniqueness: "Alice" and "alice" must not both exist.
CREATE UNIQUE INDEX IF NOT EXISTS users_username_lower_idx
    ON coding_agent_schema.users (LOWER(username));

-- ------------------------------------------------------------------
-- Conversations (sidebar history, pruned after CONVERSATION_RETENTION_DAYS).
-- ------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS coding_agent_schema.conversations (
    id         SERIAL PRIMARY KEY,
    user_id    INTEGER NOT NULL REFERENCES coding_agent_schema.users(id) ON DELETE CASCADE,
    title      TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS conversations_user_created_idx
    ON coding_agent_schema.conversations (user_id, created_at DESC);

CREATE TABLE IF NOT EXISTS coding_agent_schema.messages (
    id              SERIAL PRIMARY KEY,
    conversation_id INTEGER NOT NULL REFERENCES coding_agent_schema.conversations(id) ON DELETE CASCADE,
    question        TEXT NOT NULL,
    answer          TEXT,
    file_name       TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS messages_conversation_created_idx
    ON coding_agent_schema.messages (conversation_id, created_at ASC);

-- ------------------------------------------------------------------
-- Memory. Short-term is the last-N sliding window used as prompt context
-- and is scoped to one conversation so context does not bleed between
-- chats. Long-term is one rolling summary per user.
-- ------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS coding_agent_schema.short_term_memory (
    id              SERIAL PRIMARY KEY,
    user_id         INTEGER NOT NULL REFERENCES coding_agent_schema.users(id) ON DELETE CASCADE,
    conversation_id INTEGER REFERENCES coding_agent_schema.conversations(id) ON DELETE CASCADE,
    question        TEXT NOT NULL,
    answer          TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS short_term_user_conv_idx
    ON coding_agent_schema.short_term_memory (user_id, conversation_id, created_at ASC);

CREATE TABLE IF NOT EXISTS coding_agent_schema.long_term_memory (
    id         SERIAL PRIMARY KEY,
    user_id    INTEGER UNIQUE NOT NULL REFERENCES coding_agent_schema.users(id) ON DELETE CASCADE,
    summary    TEXT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ------------------------------------------------------------------
-- Feedback and the golden/flagged example stores that feed it back into
-- the prompt. Both are scoped per user: one developer's private code must
-- never surface as an "example" in another developer's prompt.
-- ------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS coding_agent_schema.feedback (
    id         SERIAL PRIMARY KEY,
    user_id    INTEGER NOT NULL REFERENCES coding_agent_schema.users(id) ON DELETE CASCADE,
    question   TEXT NOT NULL,
    answer     TEXT NOT NULL,
    vote       TEXT NOT NULL CHECK (vote IN ('up', 'down')),
    reason     TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS feedback_user_idx ON coding_agent_schema.feedback (user_id);

CREATE TABLE IF NOT EXISTS coding_agent_schema.golden_examples (
    id         SERIAL PRIMARY KEY,
    user_id    INTEGER NOT NULL REFERENCES coding_agent_schema.users(id) ON DELETE CASCADE,
    question   TEXT NOT NULL,
    answer     TEXT NOT NULL,
    embedding  vector(384),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS golden_examples_user_idx ON coding_agent_schema.golden_examples (user_id);

CREATE TABLE IF NOT EXISTS coding_agent_schema.flagged_answers (
    id         SERIAL PRIMARY KEY,
    user_id    INTEGER NOT NULL REFERENCES coding_agent_schema.users(id) ON DELETE CASCADE,
    question   TEXT NOT NULL,
    answer     TEXT NOT NULL,
    reason     TEXT,
    embedding  vector(384),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS flagged_answers_user_idx ON coding_agent_schema.flagged_answers (user_id);
