# Vaultic platform (Sprint 3)

The platform adds PostgreSQL (step 3.1), login with roles (3.2), database-backed records (3.3), Redis
(3.4) and Kafka replay and scoring (3.5) around the existing demo. The scope is D230, which amends D31,
and the plan is `docs/plans/sprint3_platform_plan.md`.

**The no-Docker demo is unchanged.** `uvicorn services.api.demo_server:app` works without any of this.
Nothing the platform stores feeds a model, a metric or a research table.

## What runs where

- **On the host:** the API (`services/api/demo_server.py`, the venv) and the dashboard.
- **In Docker** (compose project `vaultic`): only PostgreSQL, Redis and Kafka (D231, Q6).

| service | image | host port (`.env` key, default) | memory limit | step |
|---|---|---|---|---|
| PostgreSQL | `postgres:16.16-alpine` | `VAULTIC_POSTGRES_PORT`, 5442 | 512 MB | 3.1 |
| Redis | `redis:7.4.11-alpine` (profile `cache`) | `VAULTIC_REDIS_PORT`, 6390 | 128 MB | 3.4 |
| Kafka (KRaft) | added in 3.5 | `VAULTIC_KAFKA_PORT`, 9094 | 1 GB | 3.5 |

Every port is bound to 127.0.0.1. The ports, the project name `vaultic` and the `vaultic-*` volumes keep
this stack apart from other local compose projects, for example one using PostgreSQL 5433 and Redis 6379.
Both can run at once.

Port fallback: if `VAULTIC_POSTGRES_PORT` is not set, the older `POSTGRES_PORT` key (D14, whose old
default was 55432) is used, and 5442 only when neither is set (D231).

## One-time setup

1. **Docker Desktop's disk image on drive E.** In Docker Desktop: Settings, Resources, Advanced, "Disk
   image location". The named volumes (`vaultic-postgres`, ...) live inside that image.
2. **Memory cap for the Docker VM:** `C:\Users\Aziz Ahmad\.wslconfig` contains

   ```ini
   [wsl2]
   memory=4GB
   ```

   - It applies to every WSL2 distribution (docker-desktop and Ubuntu), so other compose projects share the
     4 GB.
   - It takes effect after `wsl --shutdown`, which stops Docker Desktop and all running containers.
   - **To undo:** delete the file (or its `memory` line), then run `wsl --shutdown`.
3. **`.env`** (never committed): set `POSTGRES_PASSWORD`, and optionally `POSTGRES_USER`, `POSTGRES_DB` and
   the `VAULTIC_*_PORT` keys. See `.env.example`.
   - **From a git worktree,** which has no `.env` of its own, point both tools at the main checkout's file:
     - `docker compose --env-file "<main checkout>/.env" ...`
     - `VAULTIC_ENV_FILE="<main checkout>/.env" python -m vaultic.infra.db ...`

## Start, migrate, stop

```bash
docker compose up -d
docker compose ps
python -m vaultic.infra.db migrate
python -m vaultic.infra.db status
docker compose down
```

- `docker compose ps` should show `vaultic-postgres-1` as healthy.
- `migrate` applies `services/db/migrations/*.sql` once each, in order.
- `status` lists the applied and the pending migrations.
- `docker compose down` stops the stack and keeps the data in the `vaultic-postgres` volume.
- **No restart policy (D233):** Vaultic's containers never start on their own, after a reboot or a Docker
  Desktop restart. They start only with `docker compose up`.
- `status` is read-only: it never creates anything. On an empty database it says "no migrations table"
  and lists every migration as pending.

## Migrations

- Plain SQL files `services/db/migrations/NNNN_name.sql`, numbered from 0001 without gaps.
- Each runs once, in one transaction, under an advisory lock.
- **Lock wait (D233):** a second runner waits at most 30 seconds for the lock (it retries every 0.5 s), then
  stops with "another migration runner holds the lock" instead of hanging.
- **One transaction per file (D233):** a file containing `COMMIT`, `ROLLBACK` or `CONCURRENTLY` is refused.
  The check ignores case and skips `--` and `/* */` comments. Such statements would end the runner's
  transaction or cannot run inside one, so a half-applied file could not be rolled back.
- An error while releasing the lock never hides the error that stopped a migration.
- `schema_migrations` records each file's name and sha256.
- **Never edit an applied migration:** the runner refuses a changed or missing applied file, a gap or a
  duplicate number. Add a new file instead.

**Schema v1** (`0001_init.sql`):
- `users`
- `cases`
- `decisions`
- `audit_log`: the SQLite log's columns, append-only by trigger; tamper-evident, not tamper-proof (D209).
- `copilot_audit`: made append-only and covered by its linked entry's hash in `0004_records.sql` (step
  3.3, Q13, below).

Later migrations: `0002_auth.sql` (refresh tokens, 3.2), `0003_session_start.sql` (12-hour session limit,
3.2a), `0004_records.sql` (audit anchor, labels, copilot records, 3.3), `0005_decisions_guard.sql`
(decisions stream-only and append-only, the label boundary; 3.3a, D241).

## Cache (step 3.4, D244)

Redis is a cache only. It makes the platform faster; it never decides anything, and the platform works
the same without it, only slower.

```bash
docker compose --profile cache up -d        # PostgreSQL + Redis
VAULTIC_CACHE=redis VAULTIC_AUTH=jwt python -m vaultic.infra.serve --port 8000
curl http://127.0.0.1:8000/api/health       # {"api": "up", "redis": "up", "postgres": "up"}
```

- **`VAULTIC_CACHE`** (environment or `.env`): empty = no Redis (as before); `redis` = Redis on
  127.0.0.1:`VAULTIC_REDIS_PORT` (6390), with `REDIS_PASSWORD` if set. Any other value refuses the start.
- **Container:** Redis 7.4.11, pinned by digest; 128 MB limit, at most 100 MB of data with `allkeys-lru`
  eviction; no RDB and no AOF, so nothing is written to disk and there is no volume; no restart policy.
- **What Redis holds (Q11):**
  - **Responses** of the user-independent GET routes: day summary, alerts, network, queues, drift,
    what-if status, model lab, meta and range. The key is the bundle's sha256, the path and the sorted
    query string, so a what-if setting or a label delay never gets another one's answer. Entries live
    5 minutes; only 200 JSON responses up to 2 MB are kept.
  - **Copies of new refresh-token records.** PostgreSQL stays the source of truth: the rotation, the
    reuse check and the audit all happen there. A rotation or logout deletes the copies.
  - **No online features.** The roadmap's Redis counters (11.2) are not built; the features come from the
    research code, the basis of the parity proof.
- **Headers:** every cached route answers with `X-Vaultic-Cache: hit`, `miss` or `bypass` and a
  `Server-Timing: app;dur=<ms>` header. Other routes carry neither.
- **Redis down:** every call has a 50 ms timeout and a circuit breaker (3 failures open it for 10 s, then
  one trial). Every route still answers with the same body, marked `bypass`; login, refresh and logout
  keep working through PostgreSQL. `/api/health` says `"redis": "down"` and the dashboard shows a small
  "Cache off" note.
- **Why a Redis outage cannot bring back a revoked session:** copies are written only for new tokens,
  never on a read. If a delete fails (Redis down during a logout), the session cache is marked stale and
  skipped until all its keys are flushed. A copy that is still "active" can only reach PostgreSQL's
  conditional rotation, which refuses it.
- **Limits:** the cache is per Redis instance and is lost when Redis stops, which is harmless. A cached
  response can be up to 5 minutes old; the bundle is read-only, so it cannot change underneath. The
  copilot's cached summaries stay in their file cache on drive E (D224).

## Records in PostgreSQL (step 3.3, D240; fixes 3.3a, D241)

**Run it with the platform server (D241, B3).** With PostgreSQL, start the API with `python -m
vaultic.infra.serve` (`services/api/platform_server.py`). `services/api/demo_server.py` is for the no-Docker
demo only. Plain `uvicorn services.api.demo_server:app` with `VAULTIC_STORE=postgres` would write its audit
log to PostgreSQL but keep the frozen in-memory label route: labels are not checked, stay in memory and get
no `feedback_labels` row (since 3.3a that route answers 503 on PostgreSQL), and no store banner is printed.

**The switch.** `VAULTIC_STORE` in the environment or `.env` (as `VAULTIC_AUTH`) chooses the stores:

| `VAULTIC_STORE` | Audit log | Analyst labels | Copilot records |
|---|---|---|---|
| empty (default) | SQLite on drive E, as before | in memory, for the session only | inside the audit entry |
| `postgres` | `audit_log` + `audit_anchor` | `feedback_labels`, persistent | `copilot_audit`, linked to the entry |

Anything else refuses the start. With `postgres`, an unreachable database or **any** pending migration
(or an applied one that changed or is missing) also refuses the start (D241, B1): there is never a silent
fallback to SQLite, which would split the log in two. The
server prints the active store at start-up, under the login mode.

**Audit log.** The same hash chain as the SQLite log (the same function, so a migrated entry keeps its
hash) and the same policies: labels fail-closed, everything else fail-open (D209). Appends are serialised
by a transaction-level advisory lock, across threads and processes. The truncation anchor is the one row of
`audit_anchor`, written in the same transaction as each append. `/api/audit/verify` also returns
`records` (copilot records and labels checked).
- **NUL characters (D241, A2).** PostgreSQL text and JSON cannot hold U+0000. The PostgreSQL store writes
  every NUL in a text field as the six characters `\u0000` *before* hashing, and the copilot record
  does the same in every string of its JSON, so the stored text is the hashed text and `verify()` still
  matches. A copilot question containing NUL is therefore audited, not lost.

**Copilot records (Q13).** A `copilot_audit` row is derived only from its hash-chained entry (kind,
target, question, analyst, role, time and the entry's `after`, which holds the route, kept and removed
facts with reasons and the timing), plus that record's sha256. The row is written in the same transaction
as the entry. `verify()` recomputes every row from its entry and needs exactly one row per copilot entry.
The table is append-only by trigger. The copilot's code is E20-frozen and unchanged.

**Labels.** `fraud`, `legit` or `unsure`, with the analyst and the time. Anything else gets 422 and
nothing is written. The audit entry and the label row are written in one transaction, so it is both or
neither. A change is a new row; the current label is the latest. `GET /api/feedback/{tid}` (analyst role)
returns the current label and the history. Without `postgres` it answers 501. Nothing reads labels into a
model, a metric or a research table (D208).
- **Known limitation (D241, B5).** Neither label route (the frozen demo route nor the platform route) checks
  that the transaction id exists in the bundle, as in the demo: a label for an unknown id is stored and
  audited like any other.
- **Every label entry has its row (D241, A1).** `verify()` fails if a label entry above the stored
  boundary (`audit_label_boundary`) has no `feedback_labels` row. Entries at or below it are exempt: those
  copied from the SQLite log, and any written before migration 0005. The store refuses a label action
  written without its row, so the frozen demo server's in-memory label route answers 503 on PostgreSQL:
  use `python -m vaultic.infra.serve` (the platform server) for labels.
- **Training labels (D241, A4).** `LabelStore.training_labels(as_of, label_delay_days)` needs both. It
  returns, per transaction, the latest label at or before `as_of` minus the delay (by label time; the
  project's delay is 30 days, CLAUDE.md rule 4), and only fraud or legit: "unsure" never reaches training.

**Decisions (Q10).** `decisions` holds the stream scorer's decisions only (source `stream`, written from
step 3.5). The frozen E12 decisions stay in the read-only bundle and are never copied. Since 0005 the
database itself refuses a `frozen_e12` row and any UPDATE, DELETE or TRUNCATE of `decisions` (D241, B2).
A `case` is an alert under review; the table has no writer yet.

**Moving the SQLite log to PostgreSQL** (only with Aziz's go-ahead; a persistent change).
**Stop the API server first** (D241, A3): a server still writing to the SQLite log would make it grow
apart from its copy.

```powershell
python -m vaultic.infra.db migrate
python -m vaultic.infra.migrate_audit --from "<path to audit.sqlite>" --dry-run
python -m vaultic.infra.migrate_audit --from "<path to audit.sqlite>"
```

- The SQLite file is opened read-only and verified first; a broken chain or anchor, an entry containing
  NUL (its hash covers the raw character, which PostgreSQL cannot store) or an already migrated log is
  refused.
- A target with any audit entry, anchor, copilot record or label is refused.
- Every entry is copied with its id and hashes, plus the copilot records derived from it. In the same
  transaction, the entry count, the last hash and the sha256 of the full CSV export are compared with the
  source and the copy is verified; any difference rolls everything back. "Verified before and after"
  means: the SQLite source before the copy, and the PostgreSQL copy inside the migration's transaction,
  before it commits (D241, wording note), so nothing is committed that did not verify.
- **The old log is closed (D241, A3).** Before the commit, the marker `<log>.migrated.json` (target
  without password, entry count, last hash, time) is written next to the SQLite log, and the log is
  re-read under SQLite's write lock. If it grew or changed during the copy, the marker is removed and
  everything rolls back. After a migration, the SQLite store refuses every append (and export) while the
  marker exists, so a server started on the old log cannot split it from the copy.
- The SQLite file itself is never changed or deleted. Keep it, with its marker, as the record of the old
  log.
- Labels from the SQLite era were session-only and are not turned into label rows; their audit entries are
  copied like every other entry.

**Migration 0004 and a non-empty `copilot_audit` (D241, B4).** 0004 adds `copilot_audit.record_sha256` as
NOT NULL with no default. That works only because the table was empty when 0004 ran, which is true for the
real database (nothing wrote `copilot_audit` before 3.3). 0004 is not edited (the runner refuses a changed
applied migration). A database whose `copilot_audit` already had rows would refuse 0004 and roll it back;
it would need a new migration that fills the column first.

**Limits.** Tamper-evident, not tamper-proof: someone with database rights can drop the triggers and
rewrite the log, the anchor and the records together (a test shows it).

## Login (step 3.2, D235; fixes 3.2a, D238)

Login with role checks for a local demo. There is no TLS on localhost, and it is not described as secure.

**Where it runs:** `services/api/platform_server.py` (D236). It wraps the demo backend, because
`services/api/demo_server.py` is part of E20's frozen code and stays byte-identical.
`uvicorn services.api.demo_server:app` still runs the plain no-Docker demo exactly as before: no login and
no role table. The dashboard treats a backend without `/api/auth/mode` as the no-login demo mode.

**Two modes of the platform server:**
- **No-login demo mode** (`VAULTIC_AUTH` unset).
  - The role table applies, so a route without a rule is refused.
  - Names and roles are self-declared in Settings and sent as the D208 headers.
  - The roles are labels and are not enforced.
  - The dashboard shows "Demo mode: no login; names and roles are self-declared".
- **Login on** (`VAULTIC_AUTH=jwt`): needs PostgreSQL with `0002_auth.sql` applied, and
  `VAULTIC_JWT_SECRET` (at least 32 characters) in `.env`. The API refuses to start without them.
  - The demo mode is off: the self-declared headers are ignored, and a request with only them gets 401.
  - The audit log records the logged-in user.

**Tokens:**
- **Access token:** a JWT (HS256) valid 15 minutes, sent as `Authorization: Bearer` and kept in the
  dashboard's memory.
- **Refresh token:** valid 8 hours, in an httpOnly cookie (SameSite=Strict, path `/api/auth`). It rotates
  on every refresh, and only its sha256 is stored.
  - **Session limit (B1):** a session lasts at most **12 hours from login**, however often it is
    refreshed. A refresh token never outlives its session.
  - **Two tabs (A5):** a rotated token presented again within 10 seconds is a concurrent refresh. It gets
    409 and nothing is revoked; the dashboard retries once with the newer cookie.
  - **Reuse:** a rotated token presented later ends the whole session family, and that is audited as
    reuse.
  - **After logout:** a refresh is refused, without a reuse entry.
  - **Logout (B2):** ends the refresh token at once. An access token already issued stays valid until it
    expires, **at most 15 minutes**.
  - **CSRF (B6):** `/api/auth/refresh` and `/api/auth/logout` also need the header `X-Vaultic-CSRF: 1`.
    This protects them only together with the strict CORS allow-list below. A page that CORS allowed
    could set the header.
  - **Not Secure (B3):** the cookie is not marked Secure, because the demo runs over plain http on
    localhost. Current browsers may accept Secure cookies on localhost, but that is not relied on and was
    not tested.
- **CORS (A4):** with login on, only the origins in `VAULTIC_CORS_ORIGINS` (default
  `http://127.0.0.1:5173,http://localhost:5173`) get CORS headers, with credentials. A `*` or any value
  that is not `scheme://host[:port]` (no path, no trailing slash) refuses the start.
- **Rate limit (A2, B5):**
  - 5 attempts per username, or 20 per client IP, in 15 minutes give 429. Each attempt is counted
    **before** the password is checked, under a lock, so parallel guesses cannot all be checked. A
    successful login gives its attempt back.
  - The limiter is in memory: per process, reset on restart, with one entry per distinct username or IP
    seen in the window.
  - Anyone can lock a known username out for 15 minutes. That is by design.
- **Client IP (A3):** never taken from proxy headers. Every documented start command turns them off
  (`--no-proxy-headers --forwarded-allow-ips ""`, or `python -m vaultic.infra.serve`).
- **Root path (B7):** a request with a root path (`--root-path`) is refused, because the routed path would
  differ from the checked one.

**Passwords:** Argon2id only; never logged, audited or returned. Every failure says "invalid username or
password". A failed login stores the typed username in the audit log only when it names an existing user
(B4), so a password typed into the username field is never stored.

**Mode at start-up (A1):** `VAULTIC_AUTH` is read from the environment and `.env`. It must be empty
(no-login demo mode) or `jwt` (login on); any other value refuses the start. The server prints
`Vaultic login: ON (jwt)` or `Vaultic login: OFF - no-login demo mode`.

**Roles:**
- `analyst` and `admin`; admin includes analyst.
- Admin only: the audit export and verify, and the Attack Lab run and suite.
- Every `/api` route has one rule in `ROUTE_ROLES` (`services/api/auth.py`). A route without a rule is
  refused, and a test fails if one is missing.

**Test users:**
- The seed file `services/db/seed/users.example.json` holds usernames and roles only: `analyst1` and
  `admin1`.
- `python -m vaultic.infra.users seed` takes each password from `.env` (`VAULTIC_SEED_PASSWORD_ANALYST1`,
  `VAULTIC_SEED_PASSWORD_ADMIN1`, at least 12 characters), or generates one and prints it once. It is
  never written to a file.
- Existing users are left alone unless you pass `--reset-password`.

**Start with login on**, from the main checkout (from a worktree, also set `VAULTIC_ENV_FILE`):

```bash
python -m vaultic.infra.db migrate
python -m vaultic.infra.users seed
VAULTIC_AUTH=jwt python -m vaultic.infra.serve --port 8000
```

or, with uvicorn directly:

```bash
VAULTIC_AUTH=jwt uvicorn services.api.platform_server:app --port 8000 --no-proxy-headers --forwarded-allow-ips ""
```

The frozen `demo_server.py` docstring still shows its old uvicorn command. It runs the no-login demo,
which has no login limit to protect.

- Then run the dashboard with `VITE_DATA_SOURCE=http` and open it at `http://127.0.0.1:5173`. Use
  127.0.0.1, not localhost, so the page and the API are the same site and the SameSite cookie works.
- **Not yet:** user management in the API (use the seed command), Redis sessions (3.4) and TLS.

## Tests

`python -m pytest tests/test_infra_db.py tests/test_compose.py tests/test_auth.py tests/test_records_pg.py`
- The tests marked `integration` run the real migrations in a throwaway schema when PostgreSQL is
  reachable, and are skipped otherwise.
- CI gets the database services in step 3.6.
