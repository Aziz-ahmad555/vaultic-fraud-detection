# Sprint 3 (platform): status and handoff

Updated 2026-11-26, after step 3.2a was committed and pushed. The plan is in
`docs/plans/sprint3_platform_plan.md` and the user guide is in `docs/PLATFORM.md`.

The work is on branch `sprint3-platform`, in the worktree `.worktrees/sprint3-platform`. The branch's upstream
is `origin/sprint3-platform`, never `phase0-1`.

## Merged into phase0-1

| PR | Step | Merge commit | Notes |
|---|---|---|---|
| #16 | 3.1: Compose, PostgreSQL schema v1, plain-SQL migrations | `80c184b` | Merged before its PR CI finished and before the review (D232) |
| #17 | 3.1a: fixes from the post-merge review | `039db47` | |
| #18 | 3.2: login with role checks | `6502feb` | CI was green before the merge, but it was merged before the review (D237) |

## Open on sprint3-platform (pushed, not merged, no PR yet)

The open branch holds step 3.2a, the fixes from the post-merge review of PR #18, plus the D239 drive rule.

| Commit | What |
|---|---|
| `d88a049` | D237, logged alone |
| `fbde546` | D238, logged alone |
| `b19b4c2` | First version of this note |
| `f12a2d6` | D239, logged alone |
| `82d98ef` | D239 code: `tools/disk_space.py`, temp files on E in `tests/conftest.py`, the CLAUDE.md rule, tests |
| `e050302` | 3.2a backend: A1–A5, B1, B4, B7, migration `0003_session_start.sql`, `vaultic.infra.serve`, PLATFORM.md |
| `f8bfe65` | 3.2a dashboard: a refresh that gets 409 is retried once |

**Checks for 3.2a** (2026-11-26, vaultic-postgres running and healthy on 127.0.0.1:5442):
- **Disk check:** C 16.9 GB free, E 22.2 GB free; both are above the 10 GB floor.
- **Full pytest** (with `VAULTIC_DATA_DIR` and `VAULTIC_ENV_FILE` set): **1054 passed, 1 failed, 4 skipped**.
  - **The failure:** the PostgreSQL integration test `test_login_refresh_reuse_and_logout_on_postgresql`.
    - It still expected an immediate second refresh to count as reuse. Under A5 that case now correctly
      gets 409.
    - It had never run for 3.2a, because Docker was down when 3.2a was written.
    - The fix: the test now checks the 409 inside the grace period and reuse after it. It also checks the
      12-hour limit (B1) on PostgreSQL with migration 0003.
  - **After the fix:** `tests/test_auth.py`, `tests/test_infra_db.py` and `tests/test_disk_space.py` gave
    61 passed with 0 skipped, so all 7 integration tests ran. After black reformatted the file, the
    committed version gave 59 passed for the auth and infra tests.
  - **Not re-run:** the other 1,000-odd tests, which the fix does not touch.
  - **The 4 skips:** torch, dice_ml twice, and `.venv-torch`. These are the usual environment skips.
- **Dashboard:** typecheck OK; vitest 125/125 (26 files); build OK.
- **Freeze, code rows:** `frozen_configs`, `frozen_feedback`, `frozen_final` and `frozen_stage2` are all
  unchanged.
- **Freeze, data rows (size and sha256):**
  - feedback 10/10;
  - stage2 20/20;
  - final 16/18. The two missing rows are the D234 paths under the deleted `phase7-prep` worktree.
- **E20's frozen `services/api/demo_server.py`:**
  - The committed blob has sha256 `55163d9a…eff3` at `6502feb` and at HEAD.
  - The Windows working copy has CRLF line endings, so a raw hash of the file on disk differs.
  - The E20 guard tests pass.
- **The real `vaultic` database:** migration 0001 is applied; `0002_auth.sql` and `0003_session_start.sql`
  are pending. The tests use throwaway schemas, and Aziz migrates the real database.

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

Next free number: **D240**.

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

- **Q10 (3.3):** what "decisions" and "cases" mean.
- **Q11 (3.4):** whether Redis is used for the cache and sessions only.
- **Q12 (3.5):** the parity size, 10,000 or 2,000 rows; give both timings.
- **Q13 (3.3):** whether `copilot_audit` should be hash-chained or append-only.

## Exact next steps

1. **Aziz:** open the 3.2a PR from the compare link
   `https://github.com/Aziz-ahmad555/vaultic-fraud-detection/compare/phase0-1...sprint3-platform`, using the
   text in `E:\old work\fyp 11\pr_sprint3_2a_to_phase0-1.md`.
2. **Wait for CI to pass.** Then run the independent review in a NEW Claude Code session, and fix or answer
   every finding **before** merging, to avoid another D232 / D237.
3. **Aziz merges.** Claude never merges.
4. **Optional, after the merge:** migrate the real database with
   `python -m vaultic.infra.db migrate` (with `VAULTIC_ENV_FILE` set), which applies 0002 and 0003. Then seed
   the test users with `python -m vaultic.infra.users seed`.
5. **Next is step 3.3,** the PostgreSQL stores and the audit migration. It starts with a plan and the answers
   to Q10 and Q13, and its own decision (D240 onwards) is logged alone before any code.
6. **Before every heavy job:** run `python tools/disk_space.py` and stop if C or E is below 10 GB (D239).
   Run one heavy job at a time.
