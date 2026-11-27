"""Redis response cache and session cache, degrading to slower without it (Sprint 3 step 3.4, D244).

- VAULTIC_CACHE: empty or 'redis', anything else refuses the start.
- The circuit breaker opens after 3 failures, lets one trial through after 10 s, and closes on success.
- Response cache: a hit is byte-identical to the uncached response; what-if settings and label delays
  never share a key; only the listed user-independent GET routes are cached; the role check and CORS still
  apply to a hit; Redis down gives the same bodies with X-Vaultic-Cache: bypass.
- Session cache: PostgreSQL (here the memory store) stays the truth; Redis copies are read first, dropped
  on rotate and revoke; a failed drop marks the cache stale until it is flushed; Redis down never stops
  login, refresh or logout; a lost rotate race inside the grace is 409, not a logout.
- On a real Redis (marker 'integration'): the same flows, a closed port, and with PostgreSQL the login flow.
"""

from __future__ import annotations

import fnmatch
import inspect
import time

import pytest
from fastapi.testclient import TestClient
from test_auth import PASSWORDS, SECRET, _login, _platform, _service
from test_auth import pg as pg_conn  # noqa: F401 - the PostgreSQL fixture
from test_demo_server import _load_module

from vaultic.infra import auth as A
from vaultic.infra import cache as C

WHATIF = "c_fp=10&c_rev=5&c_step=1&step_up_success_rate=0.5&legit_step_up_friction=1"


class FakeRedis:
    """The few Redis commands the cache uses, in memory; `down` makes every call fail like a dead
    server. TTLs follow the injected clock."""

    def __init__(self, clock=time.time):
        self.data: dict[str, tuple[object, float | None]] = {}
        self.clock, self.down, self.calls = clock, False, 0

    def _check(self):
        self.calls += 1
        if self.down:
            raise ConnectionError("redis is down")

    def _live(self, key):
        v = self.data.get(key)
        if v is None or (v[1] is not None and self.clock() >= v[1]):
            self.data.pop(key, None)
            return None
        return v[0]

    def ping(self):
        self._check()
        return True

    def get(self, key):
        self._check()
        v = self._live(key)
        return v if isinstance(v, bytes) else None

    def set(self, key, value, ex=None, px=None):
        self._check()
        ttl = ex if ex is not None else (px / 1000 if px is not None else None)
        self.data[key] = (value, None if ttl is None else self.clock() + ttl)
        return True

    def sadd(self, key, member):
        self._check()
        s = self._live(key) or set()
        s.add(member.encode() if isinstance(member, str) else member)
        self.data[key] = (s, self.data.get(key, (None, None))[1])

    def smembers(self, key):
        self._check()
        return set(self._live(key) or set())

    def expire(self, key, seconds):
        self._check()
        if key in self.data:
            self.data[key] = (self.data[key][0], self.clock() + seconds)

    def delete(self, *keys):
        self._check()
        return sum(self.data.pop(k, None) is not None for k in keys)

    def scan_iter(self, match="*", count=None):
        self._check()
        return [k for k in list(self.data) if fnmatch.fnmatch(k, match)]

    def pipeline(self, transaction=True):
        return _Pipe(self)


class _Pipe:
    def __init__(self, r):
        self.r, self.ops = r, []

    def __getattr__(self, name):
        return lambda *a, **k: self.ops.append((name, a, k))

    def execute(self):
        self.r._check()
        return [getattr(self.r, n)(*a, **k) for n, a, k in self.ops]


# ---- mode and breaker --------------------------------------------------------------------------------
def test_the_mode_is_empty_or_redis_and_anything_else_refuses_the_start():
    assert C.cache_mode({}) == "" and C.cache_mode({"VAULTIC_CACHE": " Redis "}) == "redis"
    for bad in ("on", "true", "redis1", "memcached"):
        with pytest.raises(RuntimeError, match="VAULTIC_CACHE must be empty"):
            C.cache_mode({"VAULTIC_CACHE": bad})
    assert C.link_from_env({"VAULTIC_CACHE": ""}) is None


def test_the_breaker_opens_after_three_failures_lets_one_trial_through_and_closes():
    now = [0.0]
    r = FakeRedis()
    link = C.RedisLink(r, clock=lambda: now[0])
    r.down = True
    for _ in range(3):
        with pytest.raises(C.CacheDown):
            link.run(lambda c: c.get("k"))
    assert link.is_open and link.status() == "down"
    calls = r.calls
    with pytest.raises(C.CacheDown, match="circuit open"):
        link.run(lambda c: c.get("k"))
    assert r.calls == calls  # while open, Redis is not even called
    now[0] += C.OPEN_FOR + 0.1
    with pytest.raises(C.CacheDown):  # the trial fails: open again for another 10 s
        link.run(lambda c: c.get("k"))
    assert link.is_open
    r.down = False
    now[0] += C.OPEN_FOR + 0.1
    assert link.run(lambda c: c.ping()) is True and not link.is_open and link.status() == "up"


# ---- response cache ----------------------------------------------------------------------------------
def _demo_app(tmp_path, link, auth=None):
    from test_demo_server import _app, _bundle

    from vaultic.audit.store import SQLiteAuditStore

    tmp_path.mkdir(parents=True, exist_ok=True)
    mod = _load_module()
    bundle = _bundle(tmp_path)
    _app(mod, bundle)  # marker and final run (D144, D154)
    demo = mod.create_app(bundle, runs_dir=bundle.parent / "runs",
                          audit_store=SQLiteAuditStore(tmp_path / "a.sqlite"))  # fmt: skip
    return _platform().with_login(demo, auth, announce=lambda m: None, cache=link,
                                  bundle_id="bundle-sha")  # fmt: skip


@pytest.fixture
def cached(tmp_path):
    r = FakeRedis()
    link = C.RedisLink(r)
    return TestClient(_demo_app(tmp_path, link)), r, link


def _range(c):
    return c.get("/api/range").json()["last"]


def test_a_hit_is_byte_identical_to_the_uncached_response(cached, tmp_path):
    c, r, _ = cached
    plain = TestClient(_demo_app(tmp_path / "plain", None))
    t = _range(plain)
    for url in (f"/api/day-summary?time={t}", f"/api/alerts?time={t}&L=30",
                f"/api/network?time={t}", f"/api/queues?time={t}&k=50", f"/api/drift?time={t}",
                "/api/whatif", "/api/lab", "/api/meta", "/api/range",
                f"/api/day-summary?time={t}&{WHATIF}"):  # fmt: skip
        base = plain.get(url)
        miss, hit = c.get(url), c.get(url)
        assert base.status_code == 200, (url, base.text)
        assert "x-vaultic-cache" not in base.headers  # no Redis configured: no cache layer at all
        assert miss.headers["x-vaultic-cache"] == "miss" and hit.headers["x-vaultic-cache"] == "hit"
        assert miss.content == base.content == hit.content, url
        assert hit.headers["content-type"] == base.headers["content-type"]
        assert "app;dur=" in hit.headers["server-timing"]


def test_what_if_settings_and_label_delays_never_share_a_key(cached):
    c, r, _ = cached
    t = _range(c)
    a = c.get(f"/api/day-summary?time={t}")
    b = c.get(f"/api/day-summary?time={t}&{WHATIF}")
    assert b.headers["x-vaultic-cache"] == "miss" and a.json()["whatIf"] is False
    assert b.json()["whatIf"] is True and a.content != b.content
    # the same parameters in another order are the same request, so the same key
    same = "&".join(reversed(WHATIF.split("&")))
    assert c.get(f"/api/day-summary?{same}&time={t}").headers["x-vaultic-cache"] == "hit"
    assert c.get(f"/api/alerts?time={t}&L=30").headers["x-vaultic-cache"] == "miss"
    assert c.get(f"/api/alerts?time={t}&L=0").headers["x-vaultic-cache"] == "miss"
    assert C.response_key("b1", "/api/meta", "") != C.response_key("b2", "/api/meta", "")


def test_only_the_listed_get_routes_are_cached_and_they_read_no_per_user_state(cached):
    c, r, _ = cached
    t = _range(c)
    for url in (f"/api/stream?time={t}&start={t - 3600}&end={t}", "/api/audit", "/api/thresholds",
                "/api/auth/mode", "/api/health", "/api/audit/verify"):  # fmt: skip
        assert "x-vaultic-cache" not in c.get(url).headers, url
    c.post("/api/feedback/1000", json={"label": "fraud", "role": "analyst"})
    assert not [k for k in r.data if not k.startswith(C.RESP_PREFIX)]
    # every cached route exists as a GET route of the demo app, and its handler takes no request and
    # never reads the per-session state (labels): its output is the same for every user (D244)
    app = c.app
    routes = {r_.path: r_ for r_ in app.routes if "GET" in getattr(r_, "methods", ())}
    for path in C.CACHED_PATHS:
        src = inspect.getsource(routes[path].endpoint)
        assert "state" not in src and "request" not in src.lower(), path


def test_a_hit_still_passes_the_role_check_and_gets_cors_headers(tmp_path):
    now = [1_000_000.0]
    r = FakeRedis()
    from test_demo_server import _app, _bundle

    from vaultic.audit.store import SQLiteAuditStore

    mod = _load_module()
    bundle = _bundle(tmp_path)
    _app(mod, bundle)
    demo = mod.create_app(bundle, runs_dir=bundle.parent / "runs",
                          audit_store=SQLiteAuditStore(tmp_path / "a.sqlite"))  # fmt: skip
    svc = A.AuthService(_service(now).store, SECRET, clock=lambda: now[0])
    app = _platform().with_login(demo, svc, announce=lambda m: None, cache=C.RedisLink(r),
                                 bundle_id="b")  # fmt: skip
    c = TestClient(app)
    token = _login(c, "analyst1")
    origin = {"Origin": "http://127.0.0.1:5173"}
    first = c.get("/api/meta", headers={"Authorization": f"Bearer {token}", **origin})
    again = c.get("/api/meta", headers={"Authorization": f"Bearer {token}", **origin})
    assert first.headers["x-vaultic-cache"] == "miss" and again.headers["x-vaultic-cache"] == "hit"
    assert again.headers["access-control-allow-origin"] == "http://127.0.0.1:5173"
    anon = c.get(
        "/api/meta", headers=origin
    )  # cached, but without a token: refused before the cache
    assert anon.status_code == 401 and "x-vaultic-cache" not in anon.headers


def test_redis_down_gives_the_same_bodies_marked_bypass(cached):
    c, r, link = cached
    t = _range(c)
    urls = (f"/api/day-summary?time={t}", "/api/meta", f"/api/network?time={t}")
    up = [c.get(u).content for u in urls]
    r.down = True
    for u, body in zip(urls, up, strict=True):
        resp = c.get(u)
        assert resp.status_code == 200 and resp.content == body
        assert resp.headers["x-vaultic-cache"] == "bypass"
    assert link.is_open
    assert c.get("/api/health").json() == {"api": "up", "redis": "down", "postgres": "unused"}
    r.down = False
    link._open_until = 0.0  # the 10 s have passed
    assert c.get("/api/meta").headers["x-vaultic-cache"] == "hit"
    assert c.get("/api/health").json()["redis"] == "up"


def test_entries_expire_and_large_or_failed_responses_are_not_stored(tmp_path):
    now = [0.0]
    r = FakeRedis(clock=lambda: now[0])
    c = TestClient(_demo_app(tmp_path, C.RedisLink(r)))
    assert c.get("/api/meta").headers["x-vaultic-cache"] == "miss"
    assert c.get("/api/meta").headers["x-vaultic-cache"] == "hit"
    now[0] += C.RESPONSE_TTL
    assert c.get("/api/meta").headers["x-vaultic-cache"] == "miss"
    before = len(r.data)
    assert c.get("/api/day-summary?time=1&c_fp=1").status_code == 422  # an error is never stored
    assert len(r.data) == before
    small = _demo_app(tmp_path / "s", C.RedisLink(FakeRedis()))
    # a response over the size limit is passed through but not stored
    for m in small.user_middleware:
        if m.cls is C.ResponseCache:
            m.kwargs["max_body"] = 10
    c2 = TestClient(small)
    assert c2.get("/api/meta").headers["x-vaultic-cache"] == "miss"
    assert c2.get("/api/meta").headers["x-vaultic-cache"] == "miss"


def test_without_redis_the_platform_is_unchanged_and_health_says_off(tmp_path):
    c = TestClient(_demo_app(tmp_path, None))
    assert "x-vaultic-cache" not in c.get("/api/meta").headers
    assert c.get("/api/health").json() == {"api": "up", "redis": "off", "postgres": "unused"}
    assert C.cache_banner(None).startswith("Vaultic cache: off")


# ---- session cache -----------------------------------------------------------------------------------
class CountingStore:
    """The memory store, counting refresh-token reads (the PostgreSQL stand-in)."""

    def __init__(self, store):
        self.store, self.reads = store, 0

    def __getattr__(self, name):
        return getattr(self.store, name)

    def refresh_by_hash(self, h):
        self.reads += 1
        return self.store.refresh_by_hash(h)


def _cached_service(now, r=None):
    base = CountingStore(_service(now).store)
    r = r or FakeRedis(clock=lambda: now[0])
    store = C.CachedSessionStore(base, C.RedisLink(r, clock=lambda: now[0]), clock=lambda: now[0])
    return A.AuthService(store, SECRET, clock=lambda: now[0]), base, r, store


def test_refresh_reads_the_redis_copy_and_rotation_and_logout_drop_it():
    now = [1_000_000.0]
    svc, base, r, store = _cached_service(now)
    t = svc.login("analyst1", PASSWORDS["analyst1"])
    assert (C.RT_PREFIX + A.token_hash(t.refresh)) in r.data
    t2 = svc.refresh(t.refresh)
    assert base.reads == 0  # read from the Redis copy
    assert (C.RT_PREFIX + A.token_hash(t.refresh)) not in r.data  # dropped on rotate
    with pytest.raises(A.AuthError) as tab:  # the old token, from PostgreSQL: another tab
        svc.refresh(t.refresh)
    assert tab.value.status == 409 and base.reads == 1
    now[0] += A.REFRESH_GRACE + 1
    with pytest.raises(A.AuthError, match="already used"):
        svc.refresh(t.refresh)  # real reuse: the family is revoked in PostgreSQL and in Redis
    assert not [k for k in r.data if k.startswith(C.RT_PREFIX)]
    with pytest.raises(A.AuthError):
        svc.refresh(t2.refresh)
    t3 = svc.login("admin1", PASSWORDS["admin1"])
    assert svc.logout(t3.refresh).username == "admin1"
    assert not [k for k in r.data if k.startswith(C.RT_PREFIX)]
    with pytest.raises(A.AuthError, match="session was ended"):
        svc.refresh(t3.refresh)


def test_redis_down_never_stops_login_refresh_or_logout():
    now = [1_000_000.0]
    svc, base, r, store = _cached_service(now)
    r.down = True
    t = svc.login("analyst1", PASSWORDS["analyst1"])
    t2 = svc.refresh(t.refresh)
    assert svc.verify_access(t2.access).username == "analyst1"
    assert svc.logout(t2.refresh).username == "analyst1"
    with pytest.raises(A.AuthError):
        svc.refresh(t2.refresh)
    assert store.stale  # drops failed while down: Redis is not trusted until flushed


def test_a_failed_drop_marks_the_cache_stale_and_it_is_flushed_before_use():
    now = [1_000_000.0]
    svc, base, r, store = _cached_service(now)
    t = svc.login("analyst1", PASSWORDS["analyst1"])  # a copy is in Redis
    r.down = True
    svc.logout(t.refresh)  # PostgreSQL revokes; the Redis copy cannot be dropped
    r.down = False
    store.link._open_until = 0.0
    assert (C.RT_PREFIX + A.token_hash(t.refresh)) in r.data and store.stale
    with pytest.raises(A.AuthError, match="session was ended"):
        svc.refresh(t.refresh)  # the stale copy is flushed first, never used
    assert not store.stale and not [k for k in r.data if k.startswith(C.RT_PREFIX)]


def test_a_lost_rotate_race_inside_the_grace_is_409_not_a_logout():
    now = [1_000_000.0]
    svc = _service(now)
    t = svc.login("analyst1", PASSWORDS["analyst1"])
    old = svc.store.refresh_by_hash(A.token_hash(t.refresh))
    snapshot = A.RefreshRecord(**{**old.__dict__})  # read before the other tab's rotation
    svc.refresh(t.refresh)  # the other tab wins
    user = svc.store.user(old.user_id)
    with pytest.raises(A.AuthError) as tab:
        svc._issue(user, old.family_id, old.session_start, rotate_from=snapshot)
    assert tab.value.status == 409
    now[0] += A.REFRESH_GRACE + 1
    with pytest.raises(A.AuthError, match="session was ended"):
        svc._issue(user, old.family_id, old.session_start, rotate_from=snapshot)


# ---- on a real Redis ---------------------------------------------------------------------------------
@pytest.fixture
def real_redis():
    from vaultic.infra import db

    db._load_dotenv()
    import os

    client = C.redis_client({**os.environ, "VAULTIC_CACHE": "redis"})
    try:
        client.ping()
    except Exception:  # noqa: BLE001 - no Redis here
        pytest.skip("no Redis reachable (start it with `docker compose --profile cache up -d`)")
    yield client


@pytest.mark.integration
def test_on_a_real_redis_the_response_cache_and_a_closed_port(tmp_path, real_redis):
    link = C.RedisLink(real_redis)
    bundle_id = f"test-{time.time_ns()}"
    c = TestClient(_demo_app(tmp_path, link))
    for m in c.app.user_middleware:
        if m.cls is C.ResponseCache:
            m.kwargs.update(bundle_id=bundle_id, ttl=30)
    a, b = c.get("/api/meta"), c.get("/api/meta")
    assert a.headers["x-vaultic-cache"] == "miss" and b.headers["x-vaultic-cache"] == "hit"
    assert a.content == b.content
    assert c.get("/api/health").json()["redis"] == "up"
    real_redis.delete(C.response_key(bundle_id, "/api/meta", ""))
    closed = C.RedisLink(C.redis_client({"VAULTIC_REDIS_PORT": "1"}))  # nothing listens on port 1
    d = TestClient(_demo_app(tmp_path / "closed", closed))
    t0 = time.perf_counter()
    resp = d.get("/api/meta")
    assert resp.status_code == 200 and resp.headers["x-vaultic-cache"] == "bypass"
    assert resp.content == a.content and time.perf_counter() - t0 < 2


@pytest.mark.integration
def test_real_redis_and_postgresql_login_flow(real_redis, pg_conn):  # noqa: F811
    conn, url = pg_conn
    for name, role in (("analyst1", "analyst"), ("admin1", "admin")):
        conn.execute("INSERT INTO users (username, password_hash, role) VALUES (%s, %s, %s)",
                     (name, A.hash_password(PASSWORDS[name]), role))  # fmt: skip
    now = [time.time()]
    store = C.CachedSessionStore(A.PgStore(url), C.RedisLink(real_redis), clock=lambda: now[0])
    svc = A.AuthService(store, SECRET, clock=lambda: now[0])
    t = svc.login("analyst1", PASSWORDS["analyst1"])
    assert real_redis.get(C.RT_PREFIX + A.token_hash(t.refresh)) is not None
    t2 = svc.refresh(t.refresh)
    with pytest.raises(A.AuthError) as tab:
        svc.refresh(t.refresh)
    assert tab.value.status == 409
    now[0] += A.REFRESH_GRACE + 1
    with pytest.raises(A.AuthError, match="already used"):
        svc.refresh(t.refresh)
    with pytest.raises(A.AuthError):
        svc.refresh(t2.refresh)
    assert real_redis.get(C.RT_PREFIX + A.token_hash(t2.refresh)) is None
    t3 = svc.login("admin1", PASSWORDS["admin1"])
    assert svc.logout(t3.refresh).username == "admin1"
    assert real_redis.get(C.RT_PREFIX + A.token_hash(t3.refresh)) is None
