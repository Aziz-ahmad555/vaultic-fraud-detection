# Sprint 3 (platform): status and handoff

Updated 2026-11-27, after step 3.3 was committed and pushed. The plan is in
`docs/plans/sprint3_platform_plan.md` and the user guide is in `docs/PLATFORM.md`.

The work is on branch `sprint3-platform`, in the worktree `.worktrees/sprint3-platform`. The branch's upstream
is `origin/sprint3-platform`, never `phase0-1`.

## Merged into phase0-1

| PR | Step | Merge commit | Notes |
|---|---|---|---|
| #16 | 3.1: Compose, PostgreSQL schema v1, plain-SQL migrations | `80c184b` | Merged before its PR CI finished and before the review (D232) |
| #17 | 3.1a: fixes from the post-merge review | `039db47` | |
| #18 | 3.2: login with role checks | `6502feb` | CI was green before the merge, but it was merged before the review (D237) |

## Merged later

| PR | Step | Merge commit |
|---|---|---|
| #19 | 3.2a: fixes from the post-merge review of PR #18, plus D239 | `0e7a5e0` |

## Open on sprint3-platform (pushed, not merged, no PR yet): step 3.3

Step 3.3 puts the audit log, analyst labels and copilot records in PostgreSQL behind `VAULTIC_STORE`, with
a verified migration from the SQLite log (D240, which answers Q10 and Q13). The branch was fast-forwarded
to `0e7a5e0` first.

| Commit | What |
|---|---|
| `b9b1f5d` | D240, logged alone |
| `f6bb20f` | 3.3 code, migration 0004, tests, PLATFORM.md |
| (this commit) | This note |

**What 3.3 adds**
- `src/vaultic/audit/pg_store.py`: `PostgresAuditStore` (same chain, advisory-lock appends, anchor row,
  copilot records derived from their entry, `verify()` checks chain, anchor and records).
- `src/vaultic/infra/stores.py`: the `VAULTIC_STORE` switch (fails closed), `LabelStore`, `DecisionStore`
  (stream decisions only).
- `src/vaultic/infra/migrate_audit.py`: SQLite to PostgreSQL, dry run, verified before commit.
- `services/db/migrations/0004_records.sql`: `audit_anchor`, `feedback_labels`, `copilot_audit`
  append-only with `record_sha256` and one row per entry.
- `services/api/platform_server.py`: persistent label route with the PostgreSQL store, `GET
  /api/feedback/{tid}` (501 with the SQLite log), the store banner at start-up. `services/api/auth.py`:
  its role rule.
- `src/vaultic/audit/store.py`: `default_store()` follows `VAULTIC_STORE`. `tests/conftest.py`: the tests
  always use the files store.
- E20-frozen files (`demo_server.py`, the copilot) unchanged.

**Checks for 3.3** (2026-11-27, vaultic-postgres healthy on 127.0.0.1:5442):
- **Disk:** C 9.4 GB free at the start (below the D239 floor). With Aziz's OK, about 7 GB of Visual
  Studio installer leftovers in `%TEMP%\wwy5ek4r` were deleted (C 16.4 GB); Visual Studio was later
  uninstalled in another session. Before the final checks: C 58.7 GB, E 22.0 GB.
- **Full pytest** (`VAULTIC_DATA_DIR` and `VAULTIC_ENV_FILE` set; 44 min): **1083 passed, 1 failed, 4 skipped**. The failure was `test_docs_clean`: this note had control characters where Windows-path backslashes had been read as escapes; fixed. Then, on the black-formatted files: `test_docs_clean`, `test_records_pg`, `test_auth`, `test_audit_store`, `test_infra_db`, `test_copilot_api` and `test_demo_server` gave **172 passed, 0 skipped** (every PostgreSQL integration test ran). The 4 skips of the full run are the usual environment skips (torch, dice_ml twice, `.venv-torch`).
- **Dashboard** (unchanged in 3.3): typecheck OK; vitest 125/125 (26 files); build OK.
- **Freeze, code rows:** `frozen_configs`, `frozen_feedback`, `frozen_final`, `frozen_stage2` unchanged. **Data rows:** feedback 10/10, stage2 20/20, final 16/18 (the two D234 paths). **E20:** all 9 files of `EXP-320.yaml` byte-identical (LF sha256).
- **Demo check, migration:** a copy of the real audit log (`E:\dev-cache\audit\audit.sqlite`, 14 entries,
  2 copilot entries) went into a throwaway schema: the dry run wrote nothing, the copy verified (14 entries,
  2 copilot records), and a second run was refused (target not empty). The schema was dropped; the real
  log and the real database were not touched.
- **Frozen files:** none of the changed files is in `EXP-320.yaml` or a `research/frozen_*.md` list;
  `demo_server.py` blob sha256 still `55163d9a…`.
- **Not done:** the real `vaultic` database still has 0002, 0003 and now 0004 pending (Aziz migrates); the
  real audit log is not migrated (needs Aziz's go-ahead); no dashboard change (the Case page does not yet
  show a stored label); no live uvicorn run with `VAULTIC_STORE=postgres` against the real database.

## Decisions (each logged alone in research/decisions.md)

| # | What |
|---|---|
| D230 | Scope: the platform adds Redis and Kafka (amends D31; SHOULD) |
| D231 | 3.1 design: compose project `vaultic`, ports 5442 / 6390 / 9094 (amends D14), `vaultic-*` volumes, 4 GB WSL cap, plain-SQL migrations, schema v1, package `vaultic.infra` |
| D232 | Process: PR #16 was merged before its PR CI finished and before the review (declared deviation) |
| D233 | 3.1a fixes: read-only `status`; unlock never hides the error; the D231 profile wording corrected by a dated note; no restart policy; bounded lock wait; COMMIT / ROLLBACK / CONCURRENTLY refused |
| D234 | Note: two `frozen_final.md` data rows point under the deleted `phase7-prep` worktree (identical copies are in `experiments/runs`); the routine freeze check covers code rows only, so step reports name both checks |
| D235 | 3.2 design: JWT access token (15 min) plus a rotated refresh token, Argon2id, fail-closed role table, no-login demo mode (Q8), seeded test users (Q9) |
| D236 | 3.2 wiring in `services/api/platform_server.py`, because `services/api/demo_server.py` is E20-frozen and stays byte-identical |
| D237 | Process: PR #18 was merged before the independent review (declared deviation, review finding A6) |
| D238 | 3.2a design: fixes for A1–A5, B1, B4 and B7; B2, B3, B5 and B6 documented |
| D239 | Drives: no work on C; C and E each keep at least 10 GB free, checked before heavy jobs (`tools/disk_space.py`; the test suite stops by itself); all temp files and caches under `E:\dev-cache` |
| D240 | 3.3 design: PostgreSQL records behind `VAULTIC_STORE`; Q10 (decisions = stream decisions + labels; case = alert under review); Q13 (`copilot_audit` append-only and covered by its entry's hash); verified migration. NEEDS CONFIRMATION in the PR review |

Next free number: **D241**.

## Review findings for PR #18 and how each was fixed (D238)

| Finding | Fix | Test |
|---|---|---|
| A1 auth mode fails open (`.env` not read; typos give demo mode) | `auth_mode()` reads `.env`. Only an empty value or `jwt` is accepted, and anything else refuses the start. The mode is printed at start-up. | `test_a1_*` (2) |
| A2 per-user lockout beaten by parallel requests | `RateLimiter.attempt` counts the attempt under a lock before the password is checked. A success gives its attempt back. | 30 parallel wrong guesses check only 5 passwords; `test_a2_*` (2) |
| A3 `X-Forwarded-For` trusted, so the per-IP limit can be bypassed | `python -m vaultic.infra.serve` turns proxy headers off and trusts no forwarded IPs. Every documented uvicorn command has `--no-proxy-headers --forwarded-allow-ips ""`. | A real uvicorn server test, plus a test of the docs commands |
| A4 `*` CORS origin accepted | `*`, a malformed origin or an empty list refuses the start | `test_a4` |
| A5 two tabs log each other out; false reuse entries | A token rotated less than 10 s ago gets 409, nothing is revoked, and the dashboard retries once. A refresh after logout gets 401 with no reuse entry. | `test_a5_*`, the PostgreSQL integration test, a dashboard test |
| A6 merged before the review | D237 | — |
| B1 no absolute session limit | A session ends 12 h after login however often it is refreshed: `refresh_tokens.family_started_at`, migration 0003. | `test_b1_*` and the PostgreSQL integration test |
| B2 logout leaves the access token valid for up to 15 min | Documented in PLATFORM.md | — |
| B3 the cookie's missing Secure flag isn't documented | Documented in PLATFORM.md | — |
| B4 a failed login stores the typed username | The typed name is stored only if it names a user | `test_b4_*` |
| B5 the rate limiter is in memory | Documented: per process, reset on restart, anyone can lock out a known user for 15 min | — |
| B6 the CSRF header relies on the CORS allow-list | Documented; A4 enforces the allow-list | — |
| B7 `--root-path` could skip the role check | Every request with a root path is refused (500); `serve` sets `root_path=""` | `test_b7_*` |

## Open questions for later steps

- **Q10 (3.3):** answered in D240 (needs Aziz's confirmation).
- **Q11 (3.4):** whether Redis is used for the cache and sessions only.
- **Q12 (3.5):** the parity size, 10,000 or 2,000 rows; give both timings.
- **Q13 (3.3):** answered in D240: both append-only and covered by its entry's hash (needs confirmation).

## Exact next steps

1. **Aziz:** open the 3.3 PR from the compare link
   `https://github.com/Aziz-ahmad555/vaultic-fraud-detection/compare/phase0-1...sprint3-platform`, using the
   text in `E:\old work\fyp 11\pr_sprint3_3_to_phase0-1.md`.
2. **Wait for CI to pass.** Then run the independent review in a NEW Claude Code session, and fix or answer
   every finding **before** merging (D232, D237). Confirm or change D240 (Q10, Q13) in that review.
3. **Aziz merges.** Claude never merges.
4. **Optional, after the merge:** `python -m vaultic.infra.db migrate` on the real database (applies 0002,
   0003, 0004), then, only if Aziz says go, the dry run and the real `python -m vaultic.infra.migrate_audit
   --from E:\dev-cache\audit\audit.sqlite`, then `VAULTIC_STORE=postgres` in `.env`.
5. **Next is step 3.4** (Redis cache and sessions). It starts with a plan and the answer to Q11, and its own
   decision (D241 onwards) is logged alone before any code.
6. **Before every heavy job:** run `python tools/disk_space.py` and stop if C or E is below 10 GB (D239).
