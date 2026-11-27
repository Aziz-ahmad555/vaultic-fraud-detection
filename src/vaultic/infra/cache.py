"""Redis as a response cache and a session cache, degrading to slower without it (Sprint 3 step 3.4, D244).

VAULTIC_CACHE (read from the environment and .env, like VAULTIC_STORE) chooses, and fails closed on a bad
value:
- empty: no Redis (today's behaviour);
- `redis`: Redis on 127.0.0.1:VAULTIC_REDIS_PORT (6390), with REDIS_PASSWORD if set.
Redis being down, at the start or later, never refuses anything: every call has a 50 ms timeout and a circuit
breaker, and on any error the code goes straight to the source.

- ResponseCache: an ASGI middleware for the user-independent GET routes in CACHED_PATHS. Their output is a
  function of the read-only bundle and the query only, so the key is the bundle's sha256, the path and the
  sorted query string (a what-if setting or a label delay never shares a key, D128). It is installed
  innermost, inside the role check and the CORS layers, so a cached response passes both like any other.
- CachedSessionStore: PostgreSQL stays the source of truth for refresh tokens; Redis holds a copy of each
  newly issued record, read first on refresh. A rotate or revoke deletes the family's copies after the
  PostgreSQL write. Reads never fill the cache, so a stale copy can only come from a failed delete: the
  session cache is then marked stale and bypassed until its keys are flushed. Users are never cached.
- No online features live in Redis (Q11).
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections.abc import Mapping
from dataclasses import asdict
from urllib.parse import parse_qsl, urlencode

MODES = ("", "redis")
TIMEOUT = 0.05  # seconds, connect and read
FAILS_TO_OPEN = 3  # consecutive failures that open the breaker
OPEN_FOR = 10.0  # seconds the breaker stays open before one trial call
RESPONSE_TTL = 300  # seconds
MAX_BODY = 2 * 1024 * 1024  # bytes; larger responses are not cached
SESSION_FAMILY_TTL = 12 * 3600  # the session limit (D238); a family set never outlives its tokens

# GET routes whose output depends only on the bundle and the query (labels are written, never read, by
# them), and that are the same for every user. A test checks this list against demo_server.py.
CACHED_PATHS = frozenset(
    {
        "/api/day-summary",
        "/api/alerts",
        "/api/network",
        "/api/queues",
        "/api/drift",
        "/api/whatif",
        "/api/lab",
        "/api/meta",
        "/api/range",
    }
)

RESP_PREFIX = "vaultic:resp:"
RT_PREFIX = "vaultic:rt:"  # one refresh-token record, by token hash
RF_PREFIX = "vaultic:rf:"  # the token hashes of one family


class CacheDown(Exception):
    """Redis is unreachable, too slow, or the breaker is open: use the source."""


def cache_mode(env: Mapping[str, str] | None = None) -> str:
    """'' (no Redis) or 'redis'; anything else refuses the start (D244)."""
    if env is None:
        from vaultic.infra import db

        db._load_dotenv()
        env = os.environ
    value = (env.get("VAULTIC_CACHE") or "").strip().lower()
    if value not in MODES:
        raise RuntimeError(f"VAULTIC_CACHE must be empty (no Redis) or 'redis', not {value!r}")
    return value


class RedisLink:
    """A Redis client behind a circuit breaker: run(op) returns op(client) or raises CacheDown."""

    def __init__(self, client, clock=time.monotonic, fails_to_open: int = FAILS_TO_OPEN,
                 open_for: float = OPEN_FOR):  # fmt: skip
        self.client, self.clock = client, clock
        self.fails_to_open, self.open_for = fails_to_open, open_for
        self._lock = threading.Lock()
        self._fails = 0
        self._open_until: float | None = None
        self._trial = False

    @property
    def is_open(self) -> bool:
        with self._lock:
            return self._open_until is not None and self.clock() < self._open_until

    def run(self, op):
        with self._lock:
            if self._open_until is not None:
                if self.clock() < self._open_until or self._trial:
                    raise CacheDown("circuit open")
                self._trial = True  # half-open: this call is the one trial
        try:
            out = op(self.client)
        except Exception as e:  # redis errors, timeouts, OSError: all mean "use the source"
            with self._lock:
                self._trial = False
                self._fails += 1
                if self._open_until is not None or self._fails >= self.fails_to_open:
                    self._open_until = self.clock() + self.open_for
            raise CacheDown(type(e).__name__) from e
        with self._lock:
            self._fails, self._open_until, self._trial = 0, None, False
        return out

    def status(self) -> str:
        """'up' or 'down', from one PING through the breaker."""
        try:
            self.run(lambda c: c.ping())
            return "up"
        except CacheDown:
            return "down"


def redis_client(env: Mapping[str, str]):
    import redis
    from redis.backoff import NoBackoff
    from redis.retry import Retry

    return redis.Redis(
        host="127.0.0.1",
        port=int(env.get("VAULTIC_REDIS_PORT") or 6390),
        password=env.get("REDIS_PASSWORD") or None,
        socket_timeout=TIMEOUT,
        socket_connect_timeout=TIMEOUT,
        retry=Retry(NoBackoff(), 0),  # no client-side retries: the breaker decides
    )


def link_from_env(env: Mapping[str, str] | None = None) -> RedisLink | None:
    if env is None:
        from vaultic.infra import db

        db._load_dotenv()
        env = os.environ
    return RedisLink(redis_client(env)) if cache_mode(env) == "redis" else None


def cache_banner(link: RedisLink | None) -> str:
    if link is None:
        return "Vaultic cache: off (VAULTIC_CACHE empty)"
    return f"Vaultic cache: Redis ({link.status()})"


# ---- response cache -----------------------------------------------------------------------------
def response_key(bundle_id: str, path: str, query: str) -> str:
    canonical = urlencode(sorted(parse_qsl(query, keep_blank_values=True)))
    return RESP_PREFIX + hashlib.sha256(f"{bundle_id}\n{path}\n{canonical}".encode()).hexdigest()


def _pack(content_type: bytes, body: bytes) -> bytes:
    return content_type + b"\n" + body


def _unpack(value: bytes) -> tuple[bytes, bytes]:
    content_type, _, body = value.partition(b"\n")
    return content_type, body


class ResponseCache:
    """ASGI middleware: GET responses of CACHED_PATHS from Redis when possible (D244)."""

    def __init__(self, app, link: RedisLink, bundle_id: str, ttl: int = RESPONSE_TTL,
                 max_body: int = MAX_BODY):  # fmt: skip
        self.app, self.link, self.bundle_id = app, link, bundle_id
        self.ttl, self.max_body = ttl, max_body

    async def __call__(self, scope, receive, send):
        if (scope["type"] != "http" or scope["method"] != "GET"
                or scope["path"] not in CACHED_PATHS):  # fmt: skip
            return await self.app(scope, receive, send)
        start = time.perf_counter()
        key = response_key(
            self.bundle_id, scope["path"], scope.get("query_string", b"").decode("latin-1")
        )

        def timing() -> tuple[bytes, bytes]:
            return b"server-timing", f"app;dur={(time.perf_counter() - start) * 1000:.1f}".encode()

        try:
            hit = self.link.run(lambda c: c.get(key))
        except CacheDown:
            hit, state = None, b"bypass"
        else:
            state = b"miss"
        if hit is not None:
            content_type, body = _unpack(hit)
            await send({"type": "http.response.start", "status": 200,
                        "headers": [(b"content-type", content_type),
                                    (b"content-length", str(len(body)).encode()),
                                    (b"x-vaultic-cache", b"hit"), timing()]})  # fmt: skip
            await send({"type": "http.response.body", "body": body})
            return
        captured = {"status": None, "type": b"", "chunks": [], "size": 0}

        async def capture(message):
            if message["type"] == "http.response.start":
                captured["status"] = message["status"]
                for k, v in message.get("headers", []):
                    if k.lower() == b"content-type":
                        captured["type"] = v
                message = {**message, "headers": [*message.get("headers", []),
                                                  (b"x-vaultic-cache", state), timing()]}  # fmt: skip
            elif message["type"] == "http.response.body" and captured["size"] <= self.max_body:
                chunk = message.get("body", b"")
                captured["size"] += len(chunk)
                captured["chunks"].append(chunk)
            await send(message)

        await self.app(scope, receive, capture)
        if (state == b"miss" and captured["status"] == 200 and captured["size"] <= self.max_body
                and captured["type"].startswith(b"application/json")):  # fmt: skip
            value = _pack(captured["type"], b"".join(captured["chunks"]))
            try:
                self.link.run(lambda c: c.set(key, value, ex=self.ttl))
            except CacheDown:
                pass  # the response was already sent; the next request is a miss again


def install_response_cache(app, link: RedisLink, bundle_id: str, **kw) -> None:
    """Add the cache as the INNERMOST middleware (just above the routes): the role check, the wildcard
    strip and the CORS layers all wrap it, so a hit gets exactly the headers a miss would get."""
    from starlette.middleware import Middleware

    if app.middleware_stack is not None:
        raise RuntimeError("the response cache must be installed before the app starts")
    app.user_middleware.append(Middleware(ResponseCache, link=link, bundle_id=bundle_id, **kw))


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ---- session cache ------------------------------------------------------------------------------
class CachedSessionStore:
    """PostgreSQL (the wrapped store) is the source of truth; Redis holds copies of new refresh-token
    records only (D244). Everything not about refresh tokens goes straight to the wrapped store."""

    def __init__(self, store, link: RedisLink, clock=time.time):
        self.store, self.link, self.clock = store, link, clock
        self.stale = False  # a delete may have failed: bypass Redis until the keys are flushed
        self._lock = threading.Lock()

    @property
    def source(self):
        """The source of truth (PostgreSQL), for reads that must not see a cached copy."""
        return self.store

    def __getattr__(self, name):  # user lookups etc.: never cached (a deactivation acts at once)
        return getattr(self.store, name)

    def _usable(self) -> bool:
        with self._lock:
            if not self.stale:
                return not self.link.is_open
            try:
                self.link.run(_flush_sessions)
            except CacheDown:
                return False
            self.stale = False
            return True

    def _put(self, r) -> None:
        if not self._usable():
            return
        ttl = int((r.expires_at - self.clock()) * 1000)
        if ttl <= 0:
            return
        value = json.dumps(asdict(r)).encode()

        def put(c):
            pipe = c.pipeline(transaction=True)  # the copy and its family entry, or neither
            pipe.set(RT_PREFIX + r.token_hash, value, px=ttl)
            pipe.sadd(RF_PREFIX + r.family_id, r.token_hash)
            pipe.expire(RF_PREFIX + r.family_id, SESSION_FAMILY_TTL)
            pipe.execute()

        try:
            self.link.run(put)
        except CacheDown:
            with self._lock:
                self.stale = (
                    True  # it may have been written: unknown, so flush before trusting Redis
                )

    def _drop_family(self, family_id: str) -> None:
        def drop(c):
            members = c.smembers(RF_PREFIX + family_id)
            c.delete(RF_PREFIX + family_id, *[RT_PREFIX + m.decode() for m in members])

        try:
            self.link.run(drop)
        except CacheDown:
            with self._lock:
                self.stale = True

    def add_refresh(self, r) -> None:
        self.store.add_refresh(r)
        self._put(r)

    def refresh_by_hash(self, h: str):
        if self._usable():
            try:
                value = self.link.run(lambda c: c.get(RT_PREFIX + h))
            except CacheDown:
                value = None
            if value is not None:
                from vaultic.infra.auth import RefreshRecord

                return RefreshRecord(**json.loads(value))
        return self.store.refresh_by_hash(h)  # reads never fill the cache

    def rotate(self, old_id: str, new, now: float) -> bool:
        ok = self.store.rotate(old_id, new, now)
        self._drop_family(new.family_id)
        if ok:
            self._put(new)
        return ok

    def revoke_family(self, family_id: str, now: float) -> None:
        self.store.revoke_family(family_id, now)
        self._drop_family(family_id)


def _flush_sessions(c) -> None:
    for prefix in (RT_PREFIX, RF_PREFIX):
        keys = list(c.scan_iter(match=prefix + "*", count=500))
        if keys:
            c.delete(*keys)
