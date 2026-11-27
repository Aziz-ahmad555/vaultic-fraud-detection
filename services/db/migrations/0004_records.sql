-- Records in PostgreSQL (Sprint 3 step 3.3, D240): the audit log's truncation anchor, persistent analyst
-- labels, and the copilot records made append-only and covered by their linked audit entry (Q13).
-- Nothing in these tables feeds a model, a metric or a research table (D208, D230).

-- The truncation anchor of the audit log (as <db>.anchor.json next to the SQLite log): one row, updated
-- in the same transaction as every append. Someone with database rights can rewrite the log and this row
-- together; tamper-evident, not tamper-proof.
CREATE TABLE audit_anchor (
    one     BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (one),
    entries BIGINT NOT NULL CHECK (entries >= 0),
    hash    TEXT NOT NULL
);

-- Q13: a copilot record is exactly the content of its hash-chained audit entry plus that record's
-- sha256; verify() recomputes it from the entry. One row per entry, append-only like audit_log.
ALTER TABLE copilot_audit ADD COLUMN record_sha256 TEXT NOT NULL CHECK (length(record_sha256) = 64);
ALTER TABLE copilot_audit ADD CONSTRAINT copilot_audit_one_per_entry UNIQUE (audit_id);
DROP INDEX copilot_audit_entry;  -- the unique constraint's index replaces it

CREATE FUNCTION records_append_only() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION '% is append-only', TG_TABLE_NAME;
END
$$;
CREATE TRIGGER copilot_audit_no_update_or_delete BEFORE UPDATE OR DELETE ON copilot_audit
    FOR EACH ROW EXECUTE FUNCTION records_append_only();
CREATE TRIGGER copilot_audit_no_truncate BEFORE TRUNCATE ON copilot_audit
    FOR EACH STATEMENT EXECUTE FUNCTION records_append_only();

-- Analyst labels (roadmap 11.2), one row per label action, linked to its audit entry and written in the
-- same transaction (fail-closed, D209). A change is a new row; the current label is the latest row.
-- "unsure" is stored but never enters any training.
CREATE TABLE feedback_labels (
    id             BIGSERIAL PRIMARY KEY,
    audit_id       BIGINT NOT NULL UNIQUE REFERENCES audit_log (id),
    transaction_id BIGINT NOT NULL,
    label          TEXT NOT NULL CHECK (label IN ('fraud', 'legit', 'unsure')),
    analyst        TEXT NOT NULL,
    role           TEXT NOT NULL,
    labelled_at    TEXT NOT NULL
);
CREATE INDEX feedback_labels_transaction ON feedback_labels (transaction_id, id);
CREATE TRIGGER feedback_labels_no_update_or_delete BEFORE UPDATE OR DELETE ON feedback_labels
    FOR EACH ROW EXECUTE FUNCTION records_append_only();
CREATE TRIGGER feedback_labels_no_truncate BEFORE TRUNCATE ON feedback_labels
    FOR EACH STATEMENT EXECUTE FUNCTION records_append_only();
