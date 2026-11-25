"""Seed the test users of the local demo (Sprint 3 step 3.2, D235, Q9).

    python -m vaultic.infra.users seed [--reset-password]

The users and their roles come from services/db/seed/users.example.json, which is committed and holds
no passwords. Each password is read from the environment or .env (VAULTIC_SEED_PASSWORD_<USERNAME>,
for example VAULTIC_SEED_PASSWORD_ANALYST1, at least 12 characters). If it is absent, one is generated
and printed once to the terminal; it is never written to a file. A user that already exists is left
unchanged, unless --reset-password is given. Passwords are stored only as Argon2id hashes.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
from collections.abc import Mapping
from pathlib import Path

from vaultic.infra import db
from vaultic.infra.auth import ROLES, hash_password

SEED_FILE = db.REPO / "services" / "db" / "seed" / "users.example.json"
MIN_PASSWORD = 12
USERNAME = re.compile(r"^[a-z][a-z0-9_.-]{0,59}$")


class SeedError(Exception):
    pass


def seed_users(path: Path = SEED_FILE) -> list[dict]:
    users = json.loads(path.read_text("utf-8"))["users"]
    for u in users:
        if not USERNAME.match(u.get("username", "")) or u.get("role") not in ROLES:
            raise SeedError(f"bad seed entry {u!r}: lowercase username, role analyst or admin")
        if "password" in u:
            raise SeedError("passwords are never stored in the seed file")
    return users


def env_key(username: str) -> str:
    return "VAULTIC_SEED_PASSWORD_" + re.sub(r"[^A-Z0-9]", "_", username.upper())


def password_for(username: str, env: Mapping[str, str]) -> tuple[str, bool]:
    """(password, generated?) from the environment, else a new random one."""
    given = env.get(env_key(username))
    if given:
        if len(given) < MIN_PASSWORD:
            raise SeedError(f"{env_key(username)} must be at least {MIN_PASSWORD} characters")
        return given, False
    return secrets.token_urlsafe(18), True


def seed(conn, users: list[dict], env: Mapping[str, str], reset: bool = False, out=print) -> dict:
    """Create the missing users (or reset their passwords with `reset`); returns counts."""
    counts = {"created": 0, "reset": 0, "unchanged": 0}
    for u in users:
        row = conn.execute(
            "SELECT id, role FROM users WHERE username = %s", (u["username"],)
        ).fetchone()
        if row is not None and not reset:
            counts["unchanged"] += 1
            out(f"{u['username']} ({row[1]}): exists, unchanged")
            continue
        password, generated = password_for(u["username"], env)
        h = hash_password(password)
        if row is None:
            conn.execute("INSERT INTO users (username, password_hash, role) VALUES (%s, %s, %s)",
                         (u["username"], h, u["role"]))  # fmt: skip
            counts["created"] += 1
            what = "created"
        else:
            conn.execute("UPDATE users SET password_hash = %s, role = %s WHERE id = %s",
                         (h, u["role"], row[0]))  # fmt: skip
            counts["reset"] += 1
            what = "password reset"
        if generated:
            out(
                f"{u['username']} ({u['role']}): {what}; password (shown once, stored nowhere): {password}"
            )
        else:
            out(f"{u['username']} ({u['role']}): {what}; password from {env_key(u['username'])}")
    return counts


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("command", choices=["seed"])
    p.add_argument("--reset-password", action="store_true")
    a = p.parse_args(argv)
    try:
        url = db.db_url()  # also loads .env
        with db.connect(url) as conn:
            s = db.status(conn)
            if s["pending"]:
                raise SeedError(f"apply the migrations first: pending {s['pending']}")
            counts = seed(conn, seed_users(), os.environ, a.reset_password)
        print(f"{counts} on {db.redact(url)}")
    except (SeedError, db.MigrationError) as e:
        raise SystemExit(f"refused: {e}") from None


if __name__ == "__main__":
    main()
