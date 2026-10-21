# Audit log (Phase C a)

## What it is
A persistent record of who did what in the dashboard: analyst labels, Test Lab scores and reveals,
Attack Lab runs, refused threshold changes, exports and a few client-side actions (D208). Stored in
SQLite behind one interface (`vaultic.audit.store.AuditStore`), viewable and verifiable from the
Settings page.

## Why we built it
A fraud console needs accountability for human actions (who labelled a case, who exported data).
It records actions only and never feeds a model, a metric or a decision (D208).

## How it works (simple)
Each entry stores time (UTC), analyst, role, action, target, before / after and detail, plus a
SHA-256 hash of the previous entry's hash and its own content (a hash chain). UPDATE and DELETE are
refused by database triggers. An anchor file holds the latest count and hash, so a removed tail is
detected. `GET /api/audit/verify` walks the chain and reports the first broken or missing entry
(D208, D209).

## Key results (with sources)
No research numbers: this is a product feature. Its behaviour is fixed by decisions and tests:
- Tamper-evident, not tamper-proof: detects edits, removals, reordering and truncation; does not
  protect against someone with file access who rewrites the whole chain and the anchor
  (`research/decisions.md`, D209).
- Failure policy: labels are fail-closed (no audit entry, no label change; HTTP 503); scores,
  reveals, Attack Lab runs and exports are fail-open with a visible "audit failed" warning (D209).
- The analyst name and role are self-declared by the browser (attribution, not proof) until sign-in
  is added (D209).

## Limitations
- No authentication yet; the actor is self-declared.
- Tamper-evident only (see above).
- SQLite file on the local disk; PostgreSQL is planned behind the same interface.

## Screenshots to take
1. Settings page: the audit log table with entries from a Test Lab session.
2. The chain check result ("verified" with the entry count).
3. A Test Lab result showing the small "not recorded in the audit log" warning (fail-open).
