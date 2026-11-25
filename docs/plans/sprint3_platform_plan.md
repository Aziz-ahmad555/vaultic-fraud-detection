# Sprint 3: platform (plan only)

Branch `sprint3-platform`, from `phase0-1` at `220a499` (the E20 merge, PR #15). **No code until Aziz
approves this plan.** Roadmap Phase 11 ("Real-time platform"), milestone M10.

## What exists today (checked on this branch)

- **Backend.** `services/api/demo_server.py` (FastAPI, about 900 lines) serves the EXP-200-final test period
  from `data/interim/demo_final_bundle.parquet` (92,427 transactions), read-only.
  - **Decisions:** the system's decisions are the frozen E12 ones from that bundle (D129), never written.
  - **Feedback:** analyst labels (`POST /api/feedback/{tid}`) live in memory for the session only.
  - **Optional routers:** Test Lab (`VAULTIC_TESTLAB=1`), copilot (`VAULTIC_COPILOT=1`), attack lab.
- **Identity (D208).** The dashboard sends a self-chosen analyst name and role in the headers
  `X-Vaultic-Analyst` / `X-Vaultic-Role`. There is **no login and no access control**; the role is a label.
- **Audit (D208, D209).** `vaultic.audit.store.AuditStore` is a protocol with one implementation,
  `SQLiteAuditStore`, on drive E. It is hash-chained with a truncation anchor: tamper-evident, not
  tamper-proof. D208 already planned "PostgreSQL later behind the same interface".
  - **Copilot audit:** the copilot writes one entry per request into the same log. There is no separate
    copilot table.
- **Live scoring (Test Lab, D182, D183).** `src/vaultic/live` computes every feature of a transaction at time
  t online, with the research pipeline's own code.
  - **Strictly earlier rows:** the history store answers strictly-before-t queries, and the daily window graph
    covers days [d - 30, d) in both the offline features and the live store.
  - **Parity gate:** 2,000 random test transactions; features within 1e-6 and scores within 1e-12 of the
    stored EXP-200-final run. It passed, and took **26 to 40 minutes**.
  - **Latency, measured:** features + scoring p50 472 ms (p95 577 ms); full HTTP request p50 825 ms.
  - **Cost:** about 4 GiB of RAM and a store built at start-up (about 4 minutes, then cached on E).
- **Docker.**
  - `docker-compose.yml` has PostgreSQL 16 only (host port 55432, D14), with no schema and nothing using it.
  - Docker Desktop is installed (client 29.7.2) but not running.
  - Its disk image `docker_data.vhdx` (4.4 GB) is on **drive C**.
- **CI.** `.github/workflows/ci.yml` runs lint and `pytest` on ubuntu-latest with no services. The IEEE-CIS
  data is not in CI.
- **Laptop:** 15.85 GiB RAM, CPU only. D225c measured 15.5 GiB peak with the Test Lab and Ollama together.
- **Scope record.** D31 lists the platform as SHOULD: "simplified platform (FastAPI + Postgres + replay
  script, **no Kafka**)", and README milestone M10 says the same. Redis is not mentioned. **Steps 3.4 and 3.5
  go beyond D31** (question Q1).

## Rules for every step

- **The no-Docker demo keeps working:** `uvicorn services.api.demo_server:app` with no environment variables
  behaves as now. Platform features switch on only with explicit settings (`VAULTIC_PLATFORM=1`,
  `VAULTIC_DB_URL`, ...), and a test pins the demo mode.
- **No frozen research file or result changes:** the freeze check on the four `research/frozen_*.md` records
  runs at the end of every step. Nothing written by the platform ever feeds a model, a metric or a research
  table (as D208 for the audit log).
- **Each design choice is a decision entry** (D230 onward), logged alone before its code. Each step ends with:
  - the full `pytest` with `VAULTIC_DATA_DIR`;
  - the dashboard typecheck, vitest and build when the dashboard changes;
  - the freeze check;
  - a PR, and an independent review in a new session (D32).
- **Honest wording.** Never "production-ready", "bank-grade", "secure", "PCI-DSS compliant", "real-time"
  without a measured latency, or "scalable" without a measurement. Auth is described as "login with role
  checks for a local demo".
- **One heavy job at a time:** image pulls, the Test Lab store build, the parity run, benchmarks.
- **Ask first:** every download (images, packages) and every persistent setting.
- **Storage:** all data, volumes and caches on drive E. `.env` stays untouched by me; only `.env.example`
  gets new keys with placeholders.
- **Living beside the Personal AI Agent project (Aziz, 2026-11-25).** That project uses host ports 6379
  (Redis) and 5433 (PostgreSQL), so both stacks must run at the same time without conflict:
  - its own compose project name: `name: vaultic` in `docker-compose.yml`, so containers, networks and
    volumes are prefixed `vaultic`;
  - named volumes `vaultic-postgres`, `vaultic-redis` and `vaultic-kafka`;
  - its own host ports, each set in `.env` with a default:

    | service | host port | `.env` key |
    |---|---|---|
    | PostgreSQL | 5442 | `VAULTIC_POSTGRES_PORT` |
    | Redis | 6390 | `VAULTIC_REDIS_PORT` |
    | Kafka | 9094 | `VAULTIC_KAFKA_PORT` |

  - all bound to 127.0.0.1;
  - Vaultic never touches the other project's containers, volumes or ports.

  The PostgreSQL default changes from D14's 55432 to 5442, logged as a decision amending D14 in 3.1. The old
  `POSTGRES_PORT` key is still read as a fallback, so an existing `.env` keeps working.

---

## 3.1 Docker Compose skeleton: PostgreSQL schema, migrations, health checks, memory limits

**What changes**
- `docker-compose.yml` gets the project name `vaultic`, the named volume `vaultic-postgres`, and PostgreSQL
  on host port `${VAULTIC_POSTGRES_PORT:-5442}` (see the rules above; amends D14).
- It also gets **profiles**: `db` (PostgreSQL only, the default), `cache` (+ Redis), `stream` (+ Kafka) and
  `full`.
  - In 3.1 only PostgreSQL is filled in; the others are added by their steps.
  - Each service gets a health check and a memory limit (`mem_limit`, below) and binds to 127.0.0.1 only.
- **Schema v1** as plain numbered SQL migrations (`services/db/migrations/0001_init.sql`, ...), applied by a
  small runner (`python -m vaultic.platform.db migrate`). The runner records each migration's name and
  sha256 in `schema_migrations` and refuses a changed migration. Tables:
  - `users`: id, username, password_hash, role, active, created_at;
  - `cases`: id, transaction_id, status, assignee, opened_at, closed_at;
  - `decisions`: id, transaction_id, source (`frozen_e12` / `stream`), action, risk, model_version, decided_at;
  - `audit_log`: the same columns and hash chain as the SQLite log;
  - `copilot_audit`: one row per copilot request, linked to its `audit_log` entry.
- **Volumes on drive E** (question Q2).
- **Memory budget** (question Q3): PostgreSQL 512 MB (`shared_buffers` 128 MB), with Redis and Kafka limits
  added by their steps.

**Files:** `docker-compose.yml`, `services/db/migrations/*.sql`, `src/vaultic/platform/__init__.py`,
`src/vaultic/platform/db.py` (connection + runner), `.env.example` (new keys, placeholders),
`docs/PLATFORM.md` (start/stop), `requirements.txt` (psycopg, pinned, after approval).

**Tests**
- Unit tests of the migration runner on a throwaway schema:
  - applies in order, is idempotent, and refuses an edited migration.
  - The integration tests are skipped when no database is reachable (marker `integration`).
- A test that the compose file has a memory limit, a health check and a 127.0.0.1 binding on every service,
  and no service without a profile except PostgreSQL.
- The same test checks for the project name `vaultic` and volumes prefixed `vaultic-`, and that every host
  port comes from a `VAULTIC_*_PORT` key with a default that is not 5433, 5432, 6379 or 9092 (the other
  project's ports and the usual ones).
- The demo-mode test, plus the full suite and the freeze check.

**Demo check:** with the Personal AI Agent stack running, `docker compose up -d` starts Vaultic without a
port or name clash, and `docker compose ps` shows `vaultic-postgres-1` "healthy" on 127.0.0.1:5442; `python -m
vaultic.platform.db migrate` then `\dt` lists the five tables; the dashboard without Docker still works as
before.

**Risks**
- Docker's data lives on C until it is moved (Q2).
- Windows bind mounts are slow and PostgreSQL rejects their permissions, hence named volumes inside an
  E-drive disk image.
- Port clashes: a native PostgreSQL uses 5432 (D14) and the Personal AI Agent uses 5433 and 6379, so
  Vaultic uses 5442, 6390 and 9094.
  - `up` checks each port first and stops with a clear message if it is taken, rather than failing halfway.
  - An existing `.env` with `POSTGRES_PORT=55432` keeps working through the fallback.

**Estimate:** 2 working sessions (about 6 to 8 h), plus image pulls.

## 3.2 Auth: JWT login, roles, hashed passwords, role checks on every route, login page

**What changes**
- `POST /api/auth/login` (username + password) returns a **short-lived access token** (JWT, HS256, 15
  minutes) and a **refresh token** (opaque random, 8 hours, stored hashed in PostgreSQL so it can be revoked;
  Redis takes over in 3.4). Plus `POST /api/auth/refresh` and `POST /api/auth/logout`.
- **Passwords** are hashed with Argon2id (argon2-cffi). Login failures are rate-limited per username and IP,
  in memory.
- **Roles:** `analyst` and `admin`. A FastAPI dependency `require(role)` sits on **every** `/api/*` route.
  The exceptions are `/api/auth/login`, `/api/auth/refresh` and a health route, listed explicitly.
  - **Admin only:** audit export, user management, the Attack Lab run.
  - **Analyst:** reading cases, feedback labels, the copilot.
  - The actor in the audit log comes from the token, not from the headers.
- **Platform mode only** (`VAULTIC_AUTH=jwt`, which needs PostgreSQL). The no-Docker demo keeps the D208
  headers and says on screen "demo mode, no login" (question Q8).
- **Dashboard:** a login page, the access token kept in memory, and the refresh token in an httpOnly,
  SameSite=Strict cookie. On a 401 it refreshes once, then goes back to the login page. Settings shows the
  logged-in user instead of the free-text name.
- **Test users only from seed files:** `services/db/seed/users.example.json` holds usernames and roles,
  committed with no passwords. The seed command takes each password from a git-ignored E-drive file or an
  environment variable, or generates one and writes it to a git-ignored file on E (question Q9). No real
  credential appears anywhere. The JWT signing key comes from `.env` (`VAULTIC_JWT_SECRET`), with a
  placeholder in `.env.example`.

**Files:** `src/vaultic/platform/auth.py`, `services/api/auth.py` (router), `services/api/demo_server.py`
(dependency wiring, mode switch), `services/db/migrations/0002_auth.sql`,
`services/db/seed/users.example.json`, `dashboard/src/pages/Login.tsx`, `dashboard/src/api/auth.ts`,
`dashboard/src/api/identity.ts`, `requirements.txt` (PyJWT, argon2-cffi, after approval).

**Tests**
- **Route coverage:** a test walks `app.routes` and fails if any `/api/*` route lacks a role dependency (the
  public list is explicit), so a new route cannot be added unprotected.
- **Tokens:** an expired, tampered or wrong-algorithm token gives 401, and `alg=none` is refused.
- **Roles:** an analyst on an admin route gives 403.
- **Refresh:** a refresh token after logout is refused, and so is a reused (rotated) refresh token.
- **Passwords:** the stored hash is Argon2id and never the password, and the password never appears in a log
  or an audit entry.
- **Demo mode:** unchanged with `VAULTIC_AUTH` unset.
- **Dashboard (vitest):** the login form, the 401 refresh then logout flow, the visible demo-mode label.

**Demo check:** log in as the seeded analyst; open the audit export, which gives 403; log in as admin, which
works; the audit log shows the logged-in names; wait out the 15 minutes and the session refreshes silently;
log out, and the old refresh token fails.

**Risks**
- Cookie and CORS settings between Vite (5173) and the API (8000) on localhost.
- A forgotten route; the coverage test is the guard.
- Overclaiming: no "secure"; there is no TLS on localhost, which is stated.

**Estimate:** 4 sessions.

## 3.3 Decisions, audit entries and copilot audit records in PostgreSQL, with a migration path

**What changes**
- **Audit log:** `PostgresAuditStore` implements the existing `AuditStore` protocol (append, query, count,
  verify, export).
  - Same hash chain (same `entry_hash` function) and same fail-closed / fail-open policy (D209).
  - Appends are serialised with a transaction-level advisory lock.
  - The truncation anchor moves to a one-row `audit_anchor` table updated in the same transaction. Its limit
    is stated: someone with database rights can rewrite both, as with the SQLite file and its anchor.
- **Copilot audit:** the copilot keeps writing its one `audit_log` entry, and its full record (question,
  route, kept / removed facts and reasons, timings) also goes to `copilot_audit`, linked to that entry (Q10).
- **Decisions and labels**, as defined by Q10:
  - analyst labels (`fraud` / `legit` / `unsure`, with time and analyst, roadmap 11.2) become persistent in
    `feedback_labels` instead of in memory; "unsure" never enters any training;
  - stream decisions go to `decisions` (source `stream`, from 3.5);
  - the frozen E12 decisions stay in the read-only parquet bundle and are **not** copied or changed.
- **Migration from the files:** `python -m vaultic.platform.migrate_audit --from <sqlite> --to <db url>`.
  - It first verifies the SQLite chain and its anchor, then copies every entry in order, then verifies the
    PostgreSQL chain and compares the entry count, the last hash and a full-export sha256.
  - It refuses a non-empty target and never deletes the SQLite file. A dry-run mode prints what would happen.
  - The real audit DB is migrated only with Aziz's go-ahead; it is a persistent change.
- **Store choice:** `VAULTIC_DB_URL` set means PostgreSQL stores; unset means the SQLite / in-memory
  behaviour of today.

**Files:** `src/vaultic/audit/pg_store.py`, `src/vaultic/platform/migrate_audit.py`,
`src/vaultic/platform/stores.py` (labels, decisions, copilot audit), `services/api/demo_server.py` and
`services/api/copilot.py` (store selection), `services/db/migrations/0003_records.sql`.

**Tests**
- The existing audit-store tests run against **both** stores (a parametrised fixture; PostgreSQL ones only
  when reachable). Same chain on both, `verify()` catches an edited, deleted or reordered row and a truncated
  tail.
- **Migration:** a synthetic 500-entry SQLite log migrates with an identical last hash and export digest; a
  broken source chain is refused; a non-empty target is refused.
- Labels survive an API restart.
- "unsure" labels are excluded by the label reader.
- The copilot writes exactly one `audit_log` entry plus one linked `copilot_audit` row.
- Demo mode unchanged.

**Demo check:** label a transaction, restart the API and the label is still there; run the migration dry-run
and then the real one on a copy of the audit DB; `/api/audit/verify` says intact on PostgreSQL; ask the
copilot a question and see both records.

**Risks**
- Two processes appending at once (the advisory lock and a concurrency test cover it).
- Timestamp and precision differences changing a hash: the hash covers the same canonical JSON as today, and
  a test compares SQLite and PostgreSQL hashes for the same entries.

**Estimate:** 3 sessions.

## 3.4 Redis: cache and session store, degrading to slower without it

**What changes**
- **Sessions:** refresh tokens and their revocation list move from PostgreSQL (3.2) to Redis with a TTL.
  PostgreSQL stays the fallback when Redis is down, so login and refresh still work, just with one more query.
- **Cache:** read-mostly responses keyed by request, with a short TTL:
  - day summaries;
  - what-if threshold sets (today an in-process dict, D128);
  - network graphs;
  - the copilot's cached summaries.
- **Down mode:** each Redis call has a short timeout (about 50 ms) and a circuit breaker. On error the code
  goes straight to the source and marks the response `X-Vaultic-Cache: bypass`. `/api/health` reports
  `redis: down`, and the dashboard shows a small "cache off" note.
- **Not in this step:** Redis does **not** hold online features. The roadmap (11.2) puts rolling counters
  there, but the Test Lab already computes features with the research code (the basis of parity), and a
  second feature path would need its own parity proof (question Q11).

**Files:** `src/vaultic/platform/cache.py`, `src/vaultic/platform/auth.py` (session backend),
`services/api/demo_server.py` (cache on the chosen routes), `docker-compose.yml` (redis, profile `cache`,
host port `${VAULTIC_REDIS_PORT:-6390}`, volume `vaultic-redis`, 128 MB, `maxmemory-policy allkeys-lru`, no
AOF), `requirements.txt` (redis-py, after
approval).

**Tests**
- With Redis up: cache hit and miss, TTL expiry, and a cached response byte-identical to the uncached one.
- **Redis stopped mid-run (`docker compose stop redis`):** every route still answers with the same body, the
  bypass header is set, and login / refresh / logout still work.
- The circuit breaker opens and closes.
- Responses carrying `whatIf` are never served under another setting's key, which keeps D128's statelessness.

**Demo check:** time a day summary twice (the second is a cache hit, latency shown in the response headers);
stop Redis; the dashboard keeps working and shows "cache off"; start Redis and the cache comes back.

**Risks**
- A stale cache after data changes. The data are read-only, so the risk is low; keys include the bundle
  hash.
- Caching a per-user response: only user-independent routes are cached, and a test lists them.

**Estimate:** 2 sessions.

## 3.5 Kafka (single-node KRaft): replay producer, scoring consumer, parity, no future information

**What changes**
- **Kafka:** single node in KRaft mode (profile `stream`, 1 GB limit, heap 512 MB, volume `vaultic-kafka`). The
  host listener is on `${VAULTIC_KAFKA_PORT:-9094}`; the internal listener is reachable only on the compose
  network. Topics
  `tx.raw`, `tx.scored`, `alerts` and `feedback` (roadmap 11.1), 1 partition each so order is kept, with
  `replication.factor=1`.
- **Replay producer** (`python -m vaultic.platform.replay`): reads the test-period transactions in
  `TransactionDT` order (ties broken by `TransactionID`) and publishes the raw IEEE-CIS columns only. No label,
  no derived feature and no score leaves the producer.
  - Speed: x1, x100, x1000, or as fast as possible.
  - Window: a start day and either a row count or an end day.
  - Each message carries a sequence number.
- **Scoring consumer** (`python -m vaultic.platform.scorer`): reads `tx.raw` in order and scores each
  transaction with the Test Lab engine (the frozen stage-2 system, MVAF, E12 decision). It writes
  `tx.scored` and `decisions` (source `stream`), and sends `alerts` for step-up or higher.
- **Point in time, enforced by construction:** the consumer's history holds only the pre-replay history
  (everything before the replay's first event) plus the events it has already consumed. An event's features
  are computed **before** the event is added. It checks that event times never decrease and refuses a
  message with an earlier `TransactionDT` than the last one.
- **Labels:** a label enters the history only when its_time + L <= t (L = 30, the label-delay curtain).
- **Parity test** (`python -m vaultic.platform.stream_parity`): replay N test-period transactions through
  Kafka, then compare, per transaction, every feature (1e-6) and the MVAF / B5 scores (1e-12, as D183) with
  the offline frozen-run values. It writes `research/tables/platform/stream_parity.json` (counts, largest
  differences, failing columns; no row data). The parity run is a heavy job: about 2 to 3.5 h for 10,000
  transactions at the Test Lab's measured speed (question Q12).
- **No-leakage tests:**
  1. a unit test with a hand-made 10-row stream, where an event's features change when an **earlier** event
     is changed, never when a **later** one is changed or removed;
  2. the consumer refuses out-of-order messages;
  3. a label before its curtain time is invisible;
  4. the history store, asked at time t, holds no row with time >= t (an assertion inside the store,
     switched on in tests and in the parity run).
  `tests/test_leakage.py` keeps passing.

**Files:** `src/vaultic/platform/replay.py`, `scorer.py` and `stream_parity.py`,
`src/vaultic/live/store.py` (an incremental "only what was consumed" mode next to the existing one, checked
against it), `docker-compose.yml` (kafka), `tests/test_stream_*.py`, `requirements.txt` (confluent-kafka,
after approval).

**Demo check:** start the full stack; replay day 151 at x1000; the dashboard's live page shows scored
transactions and alerts arriving; `tx.scored` count = messages sent; a stream-sourced decision appears in
`decisions`; the parity JSON says passed.

**Risks**
- **Memory:** Kafka plus the Test Lab (about 4 GiB) plus PostgreSQL and Redis fit only without the copilot
  (Q3).
- **Throughput:** about 1 to 2 transactions per second per scorer, CPU-bound. A full 92,427-transaction
  replay would take many hours, so the demo replays a window and the measured rate is stated as measured.
- **The incremental history store must reproduce the full store exactly:** a test compares the two on the
  parity sample, and the parity gate is the proof.
- **Kafka client wheels on Windows:** confluent-kafka ships wheels; checked before approval.

**Estimate:** 5 sessions, plus the parity run (heavy, about 2 to 3.5 h).

## 3.6 One command, README, integration tests in CI, measured latency and throughput

**What changes**
- **One command** (`tools/platform.ps1 up | down | status`, with an equal `make platform` for CI and Linux):
  1. start compose with the chosen profile and wait for the health checks;
  2. migrate;
  3. seed the test users (if not seeded);
  4. start the API (host venv) and the dashboard;
  5. print the URLs and the seeded user names, but **not** the passwords, which are only in the git-ignored E
     file.
  `down` stops everything; data stays in the volumes unless `--wipe` is given.
  - The script runs every compose command with `-p vaultic`.
  - `--wipe` removes only `vaultic-*` volumes, after listing them and asking.
  - It never stops or removes another project's containers or volumes.
- **README:** a "Platform (Sprint 3)" section covering the requirements, the one command, the modes (demo
  without Docker, platform with Docker), what is measured and what is not, and the limits.
- **CI:** a second job, `integration`, with PostgreSQL, Redis and Kafka as GitHub Actions service containers
  on synthetic data (CI has no IEEE-CIS data):
  - migrations;
  - auth and route coverage;
  - both audit stores and the migration;
  - Redis-down behaviour;
  - a 200-event synthetic replay through Kafka with the no-leakage tests.
  The real-data parity stays a local run, recorded in `research/tables/platform/`.
- **Measurements:** `python -m vaultic.platform.bench` on the laptop, Docker running and the copilot off,
  recorded with the commit and the machine:
  - end-to-end latency per event (producer send to `tx.scored`), p50 / p95 / p99 over a fixed window;
  - sustained throughput (events per second) for 1 scorer;
  - API latency for the main routes with and without Redis;
  - login latency.
  Results go to `research/tables/platform/bench.json` and a short table in the README. They are reported as
  "measured on one laptop CPU", not as a capacity claim. The large load test stays dropped (D31).

**Files:** `tools/platform.ps1`, `Makefile`, `README.md`, `docs/PLATFORM.md`, `.github/workflows/ci.yml`,
`src/vaultic/platform/bench.py`, `tests/integration/*`.

**Tests:** the integration job green on the PR; a dry-run test of the script's commands; the benchmark
script on synthetic data in CI (numbers not asserted, only that it runs and writes its file).

**Demo check:** with the Personal AI Agent stack running, from a fresh terminal, `tools/platform.ps1 up`
(no clash), log in, replay a day, watch alerts, stop Redis
(still works), `tools/platform.ps1 down`; the no-Docker demo still starts as before.

**Risks**
- CI time: service containers add a few minutes.
- Kafka start-up in CI is slow, so the health check waits.
- Benchmark numbers vary with the laptop's load; the conditions are stated, three runs are taken and the
  median of the runs is reported.

**Estimate:** 3 sessions, plus the benchmark (heavy, about 1 h).

**Total:** about 19 working sessions plus the heavy runs (parity 2 to 3.5 h, benchmark about 1 h, image
pulls).

---

## Questions to decide before 3.1

1. **Q1, scope (amends D31).** D31 says "simplified platform (FastAPI + Postgres + replay script, no
   Kafka)" and does not list Redis; README M10 says the same. Your steps 3.4 and 3.5 add Redis and Kafka. Do
   you approve amending D31 (logged alone as D230), keeping it SHOULD priority, behind the MUST research
   work?
2. **Q2, Docker storage on drive E.** Docker Desktop keeps images and volumes in `docker_data.vhdx`, now
   4.4 GB on **C**.
   - **Recommended:** you move it in Docker Desktop (Settings, Resources, Advanced, "Disk image location") to
     `E:\docker` before any image is pulled. It is a persistent setting, so you make it; then use named
     volumes, which then live on E.
   - **Alternative:** bind mounts to `E:\dev-cache\platform\...`. They are slower on Windows, and PostgreSQL's
     data directory has permission problems on them.
3. **Q3, memory.** I propose:
   - a cap for the Docker/WSL VM of 4 GB in `C:\Users\Aziz Ahmad\.wslconfig` (a persistent setting; you
     approve, or set it yourself);
   - container limits: PostgreSQL 512 MB, Redis 128 MB, Kafka 1 GB;
   - the API with the Test Lab on the host (about 4.5 GiB).

   The full demo then needs about 7 to 8 GiB and runs **without the copilot**, because Ollama (about 5 GiB)
   plus the Test Lab already reached 15.5 GiB (D225c). Is that budget right, and is "copilot off in the full
   stack" acceptable?
4. **Q4, downloads (one list to approve).**
   - **Images:** `postgres:16-alpine` (already in compose), `redis:7-alpine`, `apache/kafka:3.9.1` (KRaft);
     versions pinned by digest after the pull.
   - **Python, pinned with the legacy constraints:** `psycopg[binary]` 3.2, `PyJWT` 2.x, `argon2-cffi`,
     `redis` 5.x, `confluent-kafka` 2.x.
   - **Dashboard:** no new npm package planned.
   - Each is pulled only at the start of its step.
5. **Q5, migrations.** Plain numbered SQL files and a small runner (recommended: no new dependency, easy to
   review), or Alembic?
6. **Q6, where the API runs.** Recommended: on the host (the venv, Windows paths, Test Lab caches on E), with
   only PostgreSQL, Redis and Kafka in Docker. A containerised API would duplicate the 4 GiB Test Lab state.
   Agree?
7. **Q7, PR granularity.** Recommended: one PR per step (3.1 to 3.6) from `sprint3-platform` into `phase0-1`,
   each with its own independent review, merged in order before the next step starts. Or one PR for the
   whole sprint?

**Decided at their step (listed now so nothing is a surprise):**
- **3.2:**
  - Q8: does the no-Docker demo stay login-free, with the D208 headers and a visible "demo mode, no login"
    label? Recommended: yes.
  - Q9: test-user passwords: generated into a git-ignored file on E (recommended), or set by you in `.env`?
- **3.3:**
  - Q10: what "decisions" and "cases" mean. Recommended:
    - decisions = stream-scored decisions plus analyst labels;
    - frozen E12 decisions stay in the read-only bundle;
    - a case = an alert under review (open / closed, assignee);
    - copilot records in their own table linked to the single hash-chained audit entry.

    Also: when to migrate the real audit DB.
- **3.4:** Q11: Redis for cache and sessions only (as you wrote), not the roadmap's online-feature counters
  (11.2); recorded as a deviation from the roadmap. Agree?
- **3.5:**
  - Q12: parity size: 10,000 transactions as the roadmap says (about 2 to 3.5 h, one heavy job), or 2,000 as
    D183 (about 30 to 40 min)?
  - the replay window for the demo (recommended: day 151, as fast as possible, about 3,000 events).

**Not in this sprint:** TLS, multiple Kafka brokers or partitions, horizontal scaling, the large load test
(dropped, D31), Redis online features (unless Q11 changes), any change to a frozen research file or result.
