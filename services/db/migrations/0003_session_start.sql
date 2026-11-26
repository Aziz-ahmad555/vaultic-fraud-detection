-- The login time of each session, carried by every refresh token of its family (Sprint 3 step 3.2a,
-- D238, B1): a session lasts at most 12 hours from login, however often it is refreshed. Existing rows
-- take their own issue time. 0002 is applied and is not edited.
ALTER TABLE refresh_tokens ADD COLUMN family_started_at TIMESTAMPTZ;
UPDATE refresh_tokens SET family_started_at = issued_at WHERE family_started_at IS NULL;
ALTER TABLE refresh_tokens ALTER COLUMN family_started_at SET NOT NULL;
