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
| Redis | added in 3.4 | `VAULTIC_REDIS_PORT`, 6390 | 128 MB | 3.4 |
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
- `copilot_audit`: **not hash-chained or append-only yet.** Only `audit_log` is. Whether it should be is
  open question Q13 for step 3.3.

## Login (step 3.2, D235)

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
  - Presenting an already used refresh token ends the whole session family; this is audited.
  - Logout ends the session.
  - `/api/auth/refresh` and `/api/auth/logout` also need the header `X-Vaultic-CSRF: 1`.
- **Rate limit:** 5 failed logins per username, or 20 per client IP, in 15 minutes give 429.

**Passwords:** Argon2id only; never logged, audited or returned. Every failure says "invalid username or
password".

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
VAULTIC_AUTH=jwt uvicorn services.api.platform_server:app --port 8000
```

- Then run the dashboard with `VITE_DATA_SOURCE=http` and open it at `http://127.0.0.1:5173`. Use
  127.0.0.1, not localhost, so the page and the API are the same site and the SameSite cookie works.
- **Not yet:** user management in the API (use the seed command), Redis sessions (3.4) and TLS.

## Tests

`python -m pytest tests/test_infra_db.py tests/test_compose.py`
- The tests marked `integration` run the real migrations in a throwaway schema when PostgreSQL is
  reachable, and are skipped otherwise.
- CI gets the database services in step 3.6.
