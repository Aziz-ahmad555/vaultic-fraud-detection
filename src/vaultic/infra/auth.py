"""Login with role checks for the local Vaultic demo (Sprint 3 step 3.2, D235).

- Access token: a JWT signed HS256 with VAULTIC_JWT_SECRET, valid 15 minutes. Only HS256 is accepted, and
  exp, iss and aud are checked. The user is re-read on every check, so a deactivated user or a changed
  role takes effect at once.
- Refresh token: 32 random bytes, valid 8 hours, stored only as its sha256.
  - It rotates on every refresh.
  - Presenting an already rotated token revokes its whole family (reuse detection).
  - Logout revokes the family.
- Passwords: Argon2id (argon2-cffi defaults: time cost 3, 64 MiB, parallelism 4), rehashed on login when
  the parameters change, never logged or returned.
- Unknown usernames are checked against a dummy hash, and every failure says only "invalid username or
  password".
- Rate limit, in memory: 5 failures per username or 20 per client IP in 15 minutes.

There is no TLS on localhost. This is login with role checks for a local demo, not a security product.
"""

from __future__ import annotations

import hashlib
import os
import secrets
import threading
import time
import uuid
from collections import defaultdict, deque
from collections.abc import Mapping
from dataclasses import dataclass, field

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError

ACCESS_TTL = 15 * 60
REFRESH_TTL = 8 * 3600
ISSUER, AUDIENCE, ALGORITHM = "vaultic", "vaultic-api", "HS256"
MIN_SECRET = 32
ROLES = ("analyst", "admin")
RANK = {"analyst": 1, "admin": 2}  # admin includes analyst
FAIL_WINDOW, FAIL_PER_USER, FAIL_PER_IP = 15 * 60, 5, 20
GENERIC = "invalid username or password"

_hasher = PasswordHasher()  # Argon2id, RFC 9106 low-memory defaults
_dummy_hash: str | None = None


class AuthError(Exception):
    """A refusal with its HTTP status; `retry_after` (seconds) for rate-limited logins."""

    def __init__(self, status: int, message: str, retry_after: int | None = None):
        super().__init__(message)
        self.status, self.message, self.retry_after = status, message, retry_after


# ---- passwords -----------------------------------------------------------------------------------
def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(stored_hash: str, password: str) -> bool:
    try:
        return _hasher.verify(stored_hash, password)
    except (VerificationError, InvalidHashError):
        return False


def _dummy() -> str:
    global _dummy_hash
    if _dummy_hash is None:
        _dummy_hash = hash_password(secrets.token_urlsafe(16))
    return _dummy_hash


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


# ---- records -------------------------------------------------------------------------------------
@dataclass
class User:
    id: int
    username: str
    role: str
    active: bool
    password_hash: str


@dataclass
class RefreshRecord:
    id: str
    user_id: int
    token_hash: str
    family_id: str
    issued_at: float
    expires_at: float
    revoked_at: float | None = None
    replaced_by: str | None = None


class MemoryStore:
    """Users and refresh tokens in memory (tests and the store contract)."""

    def __init__(self):
        self.users: dict[int, User] = {}
        self.tokens: dict[str, RefreshRecord] = {}
        self._lock = threading.Lock()

    def add_user(self, username: str, role: str, password_hash: str, active: bool = True) -> User:
        with self._lock:
            u = User(len(self.users) + 1, username, role, active, password_hash)
            self.users[u.id] = u
            return u

    def user_by_name(self, username: str) -> User | None:
        return next((u for u in self.users.values() if u.username == username), None)

    def user(self, user_id: int) -> User | None:
        return self.users.get(user_id)

    def set_password_hash(self, user_id: int, password_hash: str) -> None:
        self.users[user_id].password_hash = password_hash

    def add_refresh(self, rec: RefreshRecord) -> None:
        with self._lock:
            self.tokens[rec.id] = rec

    def refresh_by_hash(self, h: str) -> RefreshRecord | None:
        return next((r for r in self.tokens.values() if r.token_hash == h), None)

    def rotate(self, old_id: str, new: RefreshRecord, now: float) -> bool:
        """Revoke `old_id` in favour of `new`; False if it was already revoked (a race lost)."""
        with self._lock:
            old = self.tokens[old_id]
            if old.revoked_at is not None:
                return False
            old.revoked_at, old.replaced_by = now, new.id
            self.tokens[new.id] = new
            return True

    def revoke_family(self, family_id: str, now: float) -> None:
        with self._lock:
            for r in self.tokens.values():
                if r.family_id == family_id and r.revoked_at is None:
                    r.revoked_at = now


class PgStore:
    """Users and refresh tokens in PostgreSQL (tables users and refresh_tokens, migration 0002)."""

    def __init__(self, url: str):
        self.url = url
        self._conn = None
        self._lock = threading.Lock()

    def _run(self, sql: str, params=(), fetch: str | None = None):
        import psycopg

        with self._lock:
            for attempt in (1, 2):  # one reconnect if the connection dropped
                try:
                    if self._conn is None or self._conn.closed:
                        self._conn = psycopg.connect(self.url, autocommit=True, connect_timeout=5)
                    cur = self._conn.execute(sql, params)
                    if fetch == "one":
                        return cur.fetchone()
                    return cur.fetchall() if fetch == "all" else None
                except psycopg.OperationalError:
                    self._conn = None
                    if attempt == 2:
                        raise

    _USER = "SELECT id, username, role, active, password_hash FROM users"

    def user_by_name(self, username: str) -> User | None:
        row = self._run(f"{self._USER} WHERE username = %s", (username,), "one")
        return User(*row) if row else None

    def user(self, user_id: int) -> User | None:
        row = self._run(f"{self._USER} WHERE id = %s", (user_id,), "one")
        return User(*row) if row else None

    def set_password_hash(self, user_id: int, password_hash: str) -> None:
        self._run("UPDATE users SET password_hash = %s WHERE id = %s", (password_hash, user_id))

    def add_refresh(self, r: RefreshRecord) -> None:
        self._run(
            """INSERT INTO refresh_tokens (id, user_id, token_hash, family_id, issued_at, expires_at)
               VALUES (%s, %s, %s, %s, to_timestamp(%s), to_timestamp(%s))""",
            (r.id, r.user_id, r.token_hash, r.family_id, r.issued_at, r.expires_at),
        )

    def refresh_by_hash(self, h: str) -> RefreshRecord | None:
        row = self._run(
            """SELECT id, user_id, token_hash, family_id, extract(epoch FROM issued_at),
                      extract(epoch FROM expires_at), extract(epoch FROM revoked_at), replaced_by
               FROM refresh_tokens WHERE token_hash = %s""",
            (h,),
            "one",
        )
        if not row:
            return None
        rid, uid, th, fam, iss, exp, rev, rep = row
        return RefreshRecord(rid, uid, th, fam, float(iss), float(exp),
                             float(rev) if rev is not None else None, rep)  # fmt: skip

    def rotate(self, old_id: str, new: RefreshRecord, now: float) -> bool:
        self.add_refresh(new)
        row = self._run(
            """UPDATE refresh_tokens SET revoked_at = to_timestamp(%s), replaced_by = %s
               WHERE id = %s AND revoked_at IS NULL RETURNING id""",
            (now, new.id, old_id),
            "one",
        )
        if row is None:  # lost a race: the old token was revoked meanwhile; retire the new one
            self._run("UPDATE refresh_tokens SET revoked_at = to_timestamp(%s) WHERE id = %s",
                      (now, new.id))  # fmt: skip
            return False
        return True

    def revoke_family(self, family_id: str, now: float) -> None:
        self._run(
            """UPDATE refresh_tokens SET revoked_at = to_timestamp(%s)
               WHERE family_id = %s AND revoked_at IS NULL""",
            (now, family_id),
        )


# ---- rate limit ----------------------------------------------------------------------------------
@dataclass
class RateLimiter:
    """Failed logins per username and per client IP in a sliding window (in memory, D235)."""

    clock: callable = time.time
    window: int = FAIL_WINDOW
    per_user: int = FAIL_PER_USER
    per_ip: int = FAIL_PER_IP
    _fails: dict = field(default_factory=lambda: defaultdict(deque))

    def _recent(self, key) -> deque:
        q, now = self._fails[key], self.clock()
        while q and q[0] <= now - self.window:
            q.popleft()
        return q

    def retry_after(self, username: str, ip: str) -> int | None:
        waits = []
        for key, limit in ((("user", username.lower()), self.per_user), (("ip", ip), self.per_ip)):
            q = self._recent(key)
            if len(q) >= limit:
                waits.append(int(q[0] + self.window - self.clock()) + 1)
        return max(waits) if waits else None

    def failed(self, username: str, ip: str) -> None:
        now = self.clock()
        self._fails[("user", username.lower())].append(now)
        self._fails[("ip", ip)].append(now)

    def succeeded(self, username: str) -> None:
        self._fails.pop(("user", username.lower()), None)


# ---- the service ---------------------------------------------------------------------------------
@dataclass
class Tokens:
    access: str
    refresh: str
    user: User
    access_expires_in: int = ACCESS_TTL


class AuthService:
    """Issues and checks tokens for the users in `store` (MemoryStore or PgStore)."""

    def __init__(self, store, secret: str, clock=time.time, limiter: RateLimiter | None = None):
        if not secret or len(secret) < MIN_SECRET:
            raise ValueError(f"VAULTIC_JWT_SECRET must be at least {MIN_SECRET} characters")
        self.store, self._secret, self.clock = store, secret, clock
        self.limiter = limiter or RateLimiter(clock=clock)

    # access tokens
    def issue_access(self, user: User) -> str:
        now = int(self.clock())
        claims = {"sub": str(user.id), "name": user.username, "role": user.role, "iat": now,
                  "exp": now + ACCESS_TTL, "jti": uuid.uuid4().hex, "iss": ISSUER, "aud": AUDIENCE}  # fmt: skip
        return jwt.encode(claims, self._secret, algorithm=ALGORITHM)

    def verify_access(self, token: str) -> User:
        try:
            # signature, algorithm (HS256 only), issuer and audience by PyJWT; the time checks below
            # use the service's clock, so they are testable without waiting 15 minutes
            claims = jwt.decode(
                token, self._secret, algorithms=[ALGORITHM], audience=AUDIENCE, issuer=ISSUER,
                options={"require": ["exp", "iat", "sub", "iss", "aud"], "verify_exp": False,
                         "verify_iat": False},
            )  # fmt: skip
        except jwt.PyJWTError:
            raise AuthError(401, "invalid token") from None
        if not isinstance(claims["exp"], int) or claims["exp"] <= self.clock():
            raise AuthError(401, "the session has expired")
        user = self.store.user(int(claims["sub"]))
        if user is None or not user.active:
            raise AuthError(401, "the user is not active")
        return user

    # login / refresh / logout
    def login(self, username: str, password: str, ip: str = "unknown") -> Tokens:
        wait = self.limiter.retry_after(username, ip)
        if wait is not None:
            raise AuthError(429, "too many failed logins; try again later", retry_after=wait)
        user = self.store.user_by_name(username)
        ok = verify_password(user.password_hash if user else _dummy(), password)
        if not ok or user is None or not user.active:
            self.limiter.failed(username, ip)
            raise AuthError(401, GENERIC)
        self.limiter.succeeded(username)
        if _hasher.check_needs_rehash(user.password_hash):
            self.store.set_password_hash(user.id, hash_password(password))
        return self._issue(user, family=uuid.uuid4().hex)

    def _issue(self, user: User, family: str, rotate_from: str | None = None) -> Tokens:
        now = self.clock()
        raw = secrets.token_urlsafe(32)
        rec = RefreshRecord(
            uuid.uuid4().hex, user.id, token_hash(raw), family, now, now + REFRESH_TTL
        )
        if rotate_from is None:
            self.store.add_refresh(rec)
        elif not self.store.rotate(rotate_from, rec, now):
            self.store.revoke_family(family, now)
            raise AuthError(401, "the session was ended; log in again")
        return Tokens(self.issue_access(user), raw, user)

    def refresh(self, raw: str) -> Tokens:
        """New tokens for a valid refresh token, the old one revoked (rotation). On reuse of an
        already rotated token the whole family is revoked; the AuthError carries reuse_user_id."""
        rec = self.store.refresh_by_hash(token_hash(raw or ""))
        now = self.clock()
        if rec is None:
            raise AuthError(401, "invalid refresh token")
        if rec.revoked_at is not None:
            self.store.revoke_family(rec.family_id, now)
            err = AuthError(401, "this refresh token was already used; the session is ended")
            err.reuse_user_id = rec.user_id  # for the audit entry
            raise err
        if rec.expires_at <= now:
            raise AuthError(401, "the session has expired")
        user = self.store.user(rec.user_id)
        if user is None or not user.active:
            self.store.revoke_family(rec.family_id, now)
            raise AuthError(401, "the user is not active")
        return self._issue(user, rec.family_id, rotate_from=rec.id)

    def logout(self, raw: str) -> User | None:
        rec = self.store.refresh_by_hash(token_hash(raw or ""))
        if rec is None:
            return None
        self.store.revoke_family(rec.family_id, self.clock())
        return self.store.user(rec.user_id)


def allowed(role: str, needed: str) -> bool:
    return RANK.get(role, 0) >= RANK[needed]


# ---- configuration -------------------------------------------------------------------------------
def auth_mode(env: Mapping[str, str] | None = None) -> str:
    """'jwt' when VAULTIC_AUTH=jwt, else 'demo' (the no-Docker demo, D235 Q8)."""
    env = os.environ if env is None else env
    return "jwt" if (env.get("VAULTIC_AUTH") or "").strip().lower() == "jwt" else "demo"


def service_from_env(env: Mapping[str, str] | None = None) -> AuthService:
    """The jwt-mode service; refuses to start without a long enough secret or a database."""
    from vaultic.infra import db

    if env is None:
        db._load_dotenv()
        env = os.environ
    secret = env.get("VAULTIC_JWT_SECRET") or ""
    if len(secret) < MIN_SECRET:
        raise RuntimeError(
            f"VAULTIC_AUTH=jwt needs VAULTIC_JWT_SECRET (at least {MIN_SECRET} characters) in .env"
        )
    return AuthService(PgStore(db.db_url(env)), secret)
