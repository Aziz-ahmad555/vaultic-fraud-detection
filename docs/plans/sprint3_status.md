# Sprint 3 (platform): status and handoff

Written 2026-11-26. The plan is `docs/plans/sprint3_platform_plan.md` and the user guide is `docs/PLATFORM.md`.
Branch `sprint3-platform` (worktree `.worktrees/sprint3-platform`); its upstream is `origin/sprint3-platform`,
never `phase0-1`.

## Merged into phase0-1

| PR | Step | Merge commit | Notes |
|---|---|---|---|
| #16 | 3.1 Compose, PostgreSQL schema v1, plain-SQL migrations | `80c184b` | Merged before its PR CI finished and before review: D232 |
| #17 | 3.1a post-merge review fixes | `039db47` | |
| #18 | 3.2 login with role checks | `6502feb` | CI green before merge, but merged before review: D237 |

## Open on sprint3-platform (not merged)

- **Step 3.2a**, the fixes from the post-merge review of PR #18:
  - **Committed and pushed:** only the two decisions, D237 (`d88a049`) and D238 (`fbde546`).
  - **In the working tree, not committed yet:** the 3.2a code, tests and documentation:
    - `src/vaultic/infra/auth.py` and `src/vaultic/infra/serve.py` (new);
    - `services/api/auth.py` and `services/api/platform_server.py`;
    - `services/db/migrations/0003_session_start.sql` (new);
    - `tests/test_auth.py`, `docs/PLATFORM.md`, `dashboard/src/api/auth.ts` and its test.
- **Test state when this note was written:**
  - Auth tests: 33 passed. The one documentation check that failed before PLATFORM.md was updated now has
    its update.
  - A full suite, dashboard and freeze run was in progress.
  - **The PostgreSQL integration tests have not run for 3.2a.** Docker Desktop started, but its engine
    did not come up. Its run folder shows `sailor-ingest.sock` beside `sailor-ingest.sock.stale`, the same
    symptom as the earlier crash, so it is left to Aziz.
- **No PR is open for 3.2a yet.**

## Decisions so far (each logged alone in research/decisions.md)

| # | What |
|---|---|
| D230 | Scope: platform adds Redis and Kafka (amends D31; SHOULD) |
| D231 | 3.1 design: compose project `vaultic`, ports 5442 / 6390 / 9094 (amends D14), `vaultic-*` volumes, 4 GB WSL cap, plain-SQL migrations, schema v1, package `vaultic.infra` |
| D232 | Process: PR #16 merged before its PR CI finished and before review (declared deviation) |
| D233 | 3.1a fixes: read-only `status`; unlock never hides the error; D231 profile wording corrected by a dated note; no restart policy; lock wait; COMMIT / ROLLBACK / CONCURRENTLY refused |
| D234 | Note: two `frozen_final.md` rows under the deleted `phase7-prep` worktree (identical copies in `experiments/runs`); the routine freeze check covers code rows only, so step reports name both checks |
| D235 | 3.2 design: JWT access (15 min) + rotated refresh token, Argon2id, fail-closed role table, no-login demo mode (Q8), seeded test users (Q9) |
| D236 | 3.2 wiring in `services/api/platform_server.py`, because `services/api/demo_server.py` is E20-frozen and stays byte-identical |
| D237 | Process: PR #18 merged before the independent review (declared deviation, review finding A6) |
| D238 | 3.2a design: fixes A1–A5, B1, B4, B7; B2–B7 documented |

Next free number: **D239**.

## Review findings of PR #18 and their state (D238)

| Finding | Fix | State |
|---|---|---|
| A1 auth mode fail-open | `.env` read first; only empty or `jwt` accepted; mode printed at start-up | Coded and tested; not committed |
| A2 lockout beaten by parallel requests | Attempt counted under a lock before the password check | Coded; 30 parallel guesses check 5 passwords |
| A3 X-Forwarded-For trusted | `--no-proxy-headers` in every command; `python -m vaultic.infra.serve` | Coded; real uvicorn test |
| A4 `*` CORS origin accepted | `*` and malformed origins refuse the start | Coded and tested |
| A5 two tabs log each other out / false reuse entries | 10 s grace (409, dashboard retries once); after logout no reuse entry | Coded and tested (backend and dashboard) |
| A6 merged before review | D237 | Logged |
| B1 no absolute session limit | 12 h from login; migration 0003 | Coded and tested (memory store); PostgreSQL integration not yet run |
| B4 typed username stored on failure | Stored only if it names a user | Coded and tested |
| B7 `--root-path` could skip the check | Any request with a root path refused | Coded and tested |
| B2, B3, B5, B6 | Documented in PLATFORM.md (logout and the 15-minute access token, cookie not Secure, in-memory limiter, CSRF relies on strict CORS) | Written |

Open questions for later steps:
- **Q10 (3.3):** what "decisions" and "cases" mean.
- **Q11 (3.4):** Redis for cache and sessions only.
- **Q12 (3.5):** parity size, 10,000 or 2,000; give both timings.
- **Q13 (3.3):** whether `copilot_audit` should be hash-chained or append-only.

## Exact next steps

1. **Finish the 3.2a run:** the full pytest with `VAULTIC_DATA_DIR`, the dashboard typecheck, vitest and
   build, and the freeze checks (code rows and data rows; expected: `frozen_final.md` 16/18 as in D234).
   Confirm that `services/api/demo_server.py` still has sha256 `55163d9a…`.
2. **Commit the 3.2a code** (backend and dashboard) and push `sprint3-platform`.
3. **Once Aziz has Docker running again:** `docker compose --env-file "<main checkout>/.env" up -d`, then
   run the integration tests with `VAULTIC_ENV_FILE` set:
   `pytest tests/test_infra_db.py tests/test_auth.py -m integration` (7 tests, including migration 0003 and
   the PostgreSQL login flow).
4. **Write the 3.2a PR text** (`E:\old work\fyp 11\pr_sprint3_2a_to_phase0-1.md`) and give the compare
   link `phase0-1...sprint3-platform`.
5. **Stop.** Aziz opens the PR, waits for CI and runs the independent review **before** merging, to avoid
   another D232 / D237.
6. **After 3.2a is merged:** step 3.3 (PostgreSQL stores and audit migration), starting with Q10 and Q13.

The real `vaultic` database is untouched by 3.2 and 3.2a: `0002_auth.sql` and `0003_session_start.sql` are
pending there until Aziz migrates.
