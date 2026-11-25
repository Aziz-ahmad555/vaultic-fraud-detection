-- Vaultic platform schema v1 (Sprint 3 step 3.1, D231). Applied once by `python -m vaultic.infra.db
-- migrate`; never edit an applied migration (the runner refuses a changed file): add a new one instead.
-- Nothing in these tables feeds a model, a metric or a research table (D208, D230).

-- People who can log in (3.2 adds the login itself). Passwords are stored only as a hash.
CREATE TABLE users (
    id            BIGSERIAL PRIMARY KEY,
    username      TEXT NOT NULL UNIQUE CHECK (length(username) BETWEEN 1 AND 60),
    password_hash TEXT NOT NULL,
    role          TEXT NOT NULL CHECK (role IN ('analyst', 'admin')),
    active        BOOLEAN NOT NULL DEFAULT TRUE,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- An alert under review (its meaning is settled in 3.3, Q10).
CREATE TABLE cases (
    id             BIGSERIAL PRIMARY KEY,
    transaction_id BIGINT NOT NULL,
    status         TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'closed')),
    assignee       BIGINT REFERENCES users (id),
    opened_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    closed_at      TIMESTAMPTZ,
    CHECK ((status = 'closed') = (closed_at IS NOT NULL))
);
CREATE INDEX cases_transaction ON cases (transaction_id);

-- A system decision for one transaction: the frozen E12 decisions stay in the read-only bundle;
-- rows here come from the stream scorer (3.5) or a recorded copy.
CREATE TABLE decisions (
    id             BIGSERIAL PRIMARY KEY,
    transaction_id BIGINT NOT NULL,
    source         TEXT NOT NULL CHECK (source IN ('frozen_e12', 'stream')),
    action         TEXT NOT NULL CHECK (action IN ('allow', 'monitor', 'step_up', 'hold', 'block')),
    risk           DOUBLE PRECISION CHECK (risk IS NULL OR (risk >= 0 AND risk <= 1)),
    model_version  TEXT NOT NULL,
    decided_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX decisions_transaction ON decisions (transaction_id);

-- The audit log, with exactly the columns of the SQLite log that its hash chain covers (D208, D209):
-- id is assigned by the store, time is ISO text, so a migrated entry keeps its hash. Append-only by
-- trigger: tamper-evident, not tamper-proof (someone with database rights can drop the triggers).
CREATE TABLE audit_log (
    id        BIGINT PRIMARY KEY,
    "time"    TEXT NOT NULL,
    analyst   TEXT NOT NULL,
    role      TEXT NOT NULL,
    action    TEXT NOT NULL,
    target    TEXT,
    "before"  TEXT,
    "after"   TEXT,
    detail    TEXT NOT NULL,
    prev_hash TEXT NOT NULL,
    hash      TEXT NOT NULL
);
CREATE INDEX audit_time ON audit_log ("time");
CREATE INDEX audit_analyst ON audit_log (analyst);

CREATE FUNCTION audit_append_only() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'the audit log is append-only';
END
$$;
CREATE TRIGGER audit_no_update_or_delete BEFORE UPDATE OR DELETE ON audit_log
    FOR EACH ROW EXECUTE FUNCTION audit_append_only();
CREATE TRIGGER audit_no_truncate BEFORE TRUNCATE ON audit_log
    FOR EACH STATEMENT EXECUTE FUNCTION audit_append_only();

-- The copilot's full record of one request, linked to its single audit entry (3.3).
CREATE TABLE copilot_audit (
    id         BIGSERIAL PRIMARY KEY,
    audit_id   BIGINT NOT NULL REFERENCES audit_log (id),
    record     JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX copilot_audit_entry ON copilot_audit (audit_id);
