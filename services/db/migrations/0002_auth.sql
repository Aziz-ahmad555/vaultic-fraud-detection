-- Refresh tokens for the login (Sprint 3 step 3.2, D235). Only the sha256 of a token is stored. A
-- token is revoked when it is rotated (replaced_by points to its successor), on logout, or with its
-- whole family when an already rotated token is presented again (reuse detection).
CREATE TABLE refresh_tokens (
    id          TEXT PRIMARY KEY,
    user_id     BIGINT NOT NULL REFERENCES users (id),
    token_hash  TEXT NOT NULL UNIQUE,
    family_id   TEXT NOT NULL,
    issued_at   TIMESTAMPTZ NOT NULL,
    expires_at  TIMESTAMPTZ NOT NULL,
    revoked_at  TIMESTAMPTZ,
    replaced_by TEXT REFERENCES refresh_tokens (id),
    CHECK (expires_at > issued_at)
);
CREATE INDEX refresh_tokens_family ON refresh_tokens (family_id);
CREATE INDEX refresh_tokens_user ON refresh_tokens (user_id);
