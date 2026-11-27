-- Fixes from the review of PR #20 (Sprint 3 step 3.3a, D241); 0004 is applied elsewhere and not edited.

-- B2: only the stream scorer's decisions are stored (Q10, D240): the frozen E12 decisions stay in the
-- read-only bundle (D129). The table is append-only like the other record tables. Fails (and rolls back)
-- if a frozen_e12 row already exists, rather than deleting it.
ALTER TABLE decisions ADD CONSTRAINT decisions_stream_only CHECK (source = 'stream');
CREATE TRIGGER decisions_no_update_or_delete BEFORE UPDATE OR DELETE ON decisions
    FOR EACH ROW EXECUTE FUNCTION records_append_only();
CREATE TRIGGER decisions_no_truncate BEFORE TRUNCATE ON decisions
    FOR EACH STATEMENT EXECUTE FUNCTION records_append_only();

-- A1: label entries with an id above this boundary must each have their feedback_labels row (verify()).
-- Entries at or below it are exempt: those written before this migration (e.g. by the frozen demo route)
-- and, after `migrate_audit`, every entry copied from the SQLite log. One row; same trust class as the
-- audit anchor (someone with database rights can change it).
CREATE TABLE audit_label_boundary (
    one      BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (one),
    after_id BIGINT NOT NULL CHECK (after_id >= 0)
);
INSERT INTO audit_label_boundary (one, after_id) SELECT TRUE, COALESCE(MAX(id), 0) FROM audit_log;
