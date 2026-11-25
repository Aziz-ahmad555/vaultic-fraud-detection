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

## Tests

`python -m pytest tests/test_infra_db.py tests/test_compose.py`
- The tests marked `integration` run the real migrations in a throwaway schema when PostgreSQL is
  reachable, and are skipped otherwise.
- CI gets the database services in step 3.6.
