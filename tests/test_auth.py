"""Login with role checks (Sprint 3 step 3.2, D235).

- Every /api route has a role rule: the test fails if one is missing or stale.
- The middleware fails closed.
- Tokens: 15-minute access token, rotated refresh token with reuse detection, logout.
- Passwords: Argon2id, a generic error, and a rate limit.
- Login on: the D208 headers are ignored and the audit actor comes from the token.
- The no-login demo mode is unchanged and labelled.
- The seed command creates the test users.
- On a real PostgreSQL (marker 'integration'): the same flows with the refresh_tokens table, and the seed
  command."""

from __future__ import annotations

import time

import jwt
import pytest
from fastapi.testclient import TestClient
from test_demo_server import ROOT, _app, _bundle, _load_module

from vaultic.infra import auth as A

SECRET = "s" * 40
PASSWORDS = {"analyst1": "analyst-pass-123", "admin1": "admin-pass-12345"}


def _service(now):
    store = A.MemoryStore()
    store.add_user("analyst1", "analyst", A.hash_password(PASSWORDS["analyst1"]))
    store.add_user("admin1", "admin", A.hash_password(PASSWORDS["admin1"]))
    return A.AuthService(store, SECRET, clock=lambda: now[0])


@pytest.fixture
def jwt_app(tmp_path):
    now = [1_000_000.0]
    mod = _load_module()
    svc = _service(now)
    app = _app_with(mod, tmp_path, svc)
    return app, svc, now


def _platform():
    """services/api/platform_server.py, loaded from its file (D236)."""
    import importlib.util
    import sys

    if "platform_server" in sys.modules:
        return sys.modules["platform_server"]
    spec = importlib.util.spec_from_file_location(
        "platform_server", ROOT / "services/api/platform_server.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _app_with(mod, tmp_path, auth):
    from vaultic.audit.store import SQLiteAuditStore

    bundle = _bundle(tmp_path)
    runs = bundle.parent / "runs"
    _app(mod, bundle)  # writes the test-data marker and the final run (D144, D154)
    demo = mod.create_app(
        bundle, runs_dir=runs, audit_store=SQLiteAuditStore(tmp_path / "a.sqlite")
    )
    return _platform().with_login(demo, auth)  # demo_server.py is E20-frozen: wrapped (D236)


def _login(c, user):
    r = c.post("/api/auth/login", json={"username": user, "password": PASSWORDS[user]})
    assert r.status_code == 200, r.text
    return r.json()["accessToken"]


def _bearer(token):
    return {"Authorization": f"Bearer {token}"}


# ---- every route has a rule ------------------------------------------------------------------------
def _all_api_routes(app) -> set[tuple[str, str]]:
    from fastapi.routing import APIRoute

    def walk(routes):
        for r in routes:
            if isinstance(r, APIRoute):
                yield from ((m, r.path) for m in r.methods if m != "HEAD")
            sub = getattr(r, "original_router", None) or getattr(r, "router", None)
            if sub is not None and hasattr(sub, "routes"):  # nested routers (FastAPI 0.142)
                yield from walk(sub.routes)

    return {(m, p) for m, p in walk(app.routes) if p.startswith("/api")}


def test_every_api_route_has_exactly_one_role_rule_and_no_rule_is_stale(tmp_path):
    from types import SimpleNamespace

    from vaultic.live.service import TestLabService

    mod = _load_module()
    app = _app_with(mod, tmp_path, None)
    mod.mount_testlab(app, service=TestLabService(), start=False)  # Test Lab + Attack Lab routers
    mod.mount_copilot(app, SimpleNamespace())
    routes = _all_api_routes(app)
    rules = set(_platform().auth_api.ROUTE_ROLES)
    assert len(routes) == 40  # 5 auth + 20 demo backend + 8 Test Lab + 3 Attack Lab + 4 copilot
    assert routes - rules == set(), "routes without a role rule"
    assert rules - routes == set(), "role rules for routes that do not exist"
    roles = _platform().auth_api.ROUTE_ROLES
    assert {k for k, v in roles.items() if v == "public"} == {
        ("GET", "/api/auth/mode"), ("POST", "/api/auth/login"),
        ("POST", "/api/auth/refresh"), ("POST", "/api/auth/logout")}  # fmt: skip
    assert (
        roles[("GET", "/api/audit/export.csv")] == "admin"
        and roles[("POST", "/api/attack/run")] == "admin"
    )


def test_a_route_without_a_rule_is_refused_in_both_modes(tmp_path, jwt_app):
    (tmp_path / "demo").mkdir()
    for app in (_app_with(_load_module(), tmp_path / "demo", None), jwt_app[0]):
        app.get("/api/not-in-the-table")(lambda: {"leak": True})
        r = TestClient(app).get("/api/not-in-the-table")
        assert r.status_code == 500 and "no role rule" in r.text


# ---- login on (jwt) --------------------------------------------------------------------------------
def test_login_on_requires_a_token_and_ignores_the_demo_headers(jwt_app):
    app, _, _ = jwt_app
    c = TestClient(app)
    assert c.get("/api/auth/mode").json()["mode"] == "jwt"
    assert c.get("/api/meta").status_code == 401
    r = c.get("/api/meta", headers={"X-Vaultic-Analyst": "mallory", "X-Vaultic-Role": "admin"})
    assert r.status_code == 401 and r.headers["WWW-Authenticate"] == "Bearer"
    assert c.get("/api/meta", headers=_bearer(_login(c, "analyst1"))).status_code == 200


def test_roles_are_enforced_admin_includes_analyst(jwt_app):
    app, _, _ = jwt_app
    c = TestClient(app)
    analyst, admin = _bearer(_login(c, "analyst1")), _bearer(_login(c, "admin1"))
    assert c.get("/api/audit/export.csv", headers=analyst).status_code == 403
    assert c.get("/api/audit/verify", headers=analyst).status_code == 403
    assert c.get("/api/audit/export.csv", headers=admin).status_code == 200
    assert c.get("/api/meta", headers=admin).status_code == 200


def test_the_audit_actor_comes_from_the_token_not_the_headers(jwt_app):
    app, _, _ = jwt_app
    c = TestClient(app)
    h = {
        **_bearer(_login(c, "analyst1")),
        "X-Vaultic-Analyst": "mallory",
        "X-Vaultic-Role": "admin",
    }
    me = c.get("/api/auth/me", headers=h).json()
    assert me == {"mode": "jwt", "username": "analyst1", "role": "analyst"}
    # an audited action (the refused threshold change, D129) sent with someone else's headers
    assert c.put("/api/thresholds", headers=h, json={}).status_code >= 400
    entries = c.get("/api/audit", headers=h).json()
    refused = [e for e in entries if e["target"] == "thresholds"]
    assert refused and {(e["analyst"], e["role"]) for e in refused} == {("analyst1", "analyst")}
    assert "mallory" not in {e["analyst"] for e in entries}


def test_bad_tokens_are_refused(jwt_app):
    app, svc, now = jwt_app
    c = TestClient(app)
    good = _login(c, "analyst1")
    claims = jwt.decode(good, options={"verify_signature": False})
    forged = [
        jwt.encode(claims, "x" * 40, algorithm="HS256"),  # wrong secret
        jwt.encode(claims, SECRET, algorithm="HS512"),  # another algorithm
        jwt.encode(claims, None, algorithm="none"),  # unsigned
        jwt.encode({**claims, "aud": "other"}, SECRET, algorithm="HS256"),
        jwt.encode({**claims, "iss": "other"}, SECRET, algorithm="HS256"),
        good[:-3] + ("AAA" if not good.endswith("AAA") else "BBB"),  # tampered signature
        "not-a-token",
    ]
    for t in forged:
        assert c.get("/api/meta", headers=_bearer(t)).status_code == 401, t
    now[0] += A.ACCESS_TTL + 1  # 15 minutes later
    r = c.get("/api/meta", headers=_bearer(good))
    assert r.status_code == 401 and "expired" in r.text


def test_a_deactivated_user_or_changed_role_takes_effect_at_once(jwt_app):
    app, svc, _ = jwt_app
    c = TestClient(app)
    t = _login(c, "admin1")
    assert c.get("/api/audit/export.csv", headers=_bearer(t)).status_code == 200
    svc.store.user_by_name("admin1").role = "analyst"
    assert c.get("/api/audit/export.csv", headers=_bearer(t)).status_code == 403
    svc.store.user_by_name("admin1").active = False
    assert c.get("/api/meta", headers=_bearer(t)).status_code == 401


def test_refresh_rotates_and_a_reused_token_ends_the_whole_session(jwt_app):
    app, svc, now = jwt_app
    c = TestClient(app)
    _login(c, "analyst1")
    first = c.cookies.get(A_COOKIE := "vaultic_refresh")
    assert first
    assert c.post("/api/auth/refresh").status_code == 403  # the CSRF header is required
    r = c.post("/api/auth/refresh", headers={"X-Vaultic-CSRF": "1"})
    assert r.status_code == 200 and r.json()["accessToken"]
    second = c.cookies.get(A_COOKIE)
    assert second and second != first
    # someone replays the first (rotated) token after the grace window: refused, and the whole
    # family is revoked (within the window it would be a concurrent refresh, D238)
    now[0] += A.REFRESH_GRACE + 1
    thief = TestClient(app)
    thief.cookies.set(A_COOKIE, first, path="/api/auth")
    assert thief.post("/api/auth/refresh", headers={"X-Vaultic-CSRF": "1"}).status_code == 401
    assert c.post("/api/auth/refresh", headers={"X-Vaultic-CSRF": "1"}).status_code == 401
    admin = _bearer(_login(TestClient(app), "admin1"))
    actions = [e["action"] for e in TestClient(app).get("/api/audit", headers=admin).json()]
    assert "refresh.reuse" in actions


def test_logout_revokes_and_an_expired_refresh_token_is_refused(jwt_app):
    app, _, now = jwt_app
    c = TestClient(app)
    _login(c, "analyst1")
    token = c.cookies.get("vaultic_refresh")
    assert c.post("/api/auth/logout", headers={"X-Vaultic-CSRF": "1"}).json() == {"loggedOut": True}
    again = TestClient(app)
    again.cookies.set("vaultic_refresh", token, path="/api/auth")
    assert again.post("/api/auth/refresh", headers={"X-Vaultic-CSRF": "1"}).status_code == 401
    c2 = TestClient(app)
    _login(c2, "analyst1")
    now[0] += A.REFRESH_TTL + 1
    r = c2.post("/api/auth/refresh", headers={"X-Vaultic-CSRF": "1"})
    assert r.status_code == 401 and "expired" in r.text


def test_failures_are_generic_rate_limited_and_never_reveal_the_password(jwt_app):
    app, svc, now = jwt_app
    c = TestClient(app)
    wrong = c.post("/api/auth/login", json={"username": "analyst1", "password": "nope-nope-nope"})
    unknown = c.post("/api/auth/login", json={"username": "nobody", "password": "nope-nope-nope"})
    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json() == unknown.json() == {"detail": "invalid username or password"}
    for _ in range(A.FAIL_PER_USER - 1):
        c.post("/api/auth/login", json={"username": "analyst1", "password": "nope-nope-nope"})
    r = c.post("/api/auth/login", json={"username": "analyst1", "password": PASSWORDS["analyst1"]})
    assert (
        r.status_code == 429 and int(r.headers["Retry-After"]) > 0
    )  # even the right password waits
    now[0] += A.FAIL_WINDOW + 1
    assert c.post("/api/auth/login", json={"username": "analyst1",
                                           "password": PASSWORDS["analyst1"]}).status_code == 200  # fmt: skip
    admin = _bearer(_login(c, "admin1"))
    csv = c.get("/api/audit/export.csv", headers=admin).text
    assert "login.failed" in csv and "nope-nope-nope" not in csv
    assert all(p not in csv for p in PASSWORDS.values())
    assert svc.store.user_by_name("analyst1").password_hash.startswith("$argon2id$")


# ---- the no-login demo mode ------------------------------------------------------------------------
def test_the_demo_mode_is_unchanged_labelled_and_has_no_login(tmp_path):
    app = _app_with(_load_module(), tmp_path, None)
    c = TestClient(app)
    mode = c.get("/api/auth/mode").json()
    assert mode["mode"] == "demo" and "no login" in mode["demoLabel"]
    assert c.get("/api/meta").status_code == 200  # no token needed
    h = {"X-Vaultic-Analyst": "Zoe", "X-Vaultic-Role": "analyst"}
    assert c.get("/api/auth/me", headers=h).json() == {
        "mode": "demo",
        "username": "Zoe",
        "role": "analyst",
    }
    assert c.get("/api/audit/export.csv", headers=h).status_code == 200  # roles are labels here
    assert c.post("/api/auth/login", json={"username": "a", "password": "b"}).status_code == 409


def test_login_on_needs_a_long_secret_and_a_database():
    assert A.auth_mode({}) == "demo" and A.auth_mode({"VAULTIC_AUTH": "JWT"}) == "jwt"
    with pytest.raises(RuntimeError, match="VAULTIC_JWT_SECRET"):
        A.service_from_env({"VAULTIC_AUTH": "jwt", "VAULTIC_JWT_SECRET": "short"})
    with pytest.raises(ValueError):
        A.AuthService(A.MemoryStore(), "short")


# ---- seed (Q9) -------------------------------------------------------------------------------------
def test_the_seed_file_has_no_passwords_and_passwords_come_from_env_or_are_generated(tmp_path):
    from vaultic.infra import users

    assert [(u["username"], u["role"]) for u in users.seed_users()] == [("analyst1", "analyst"),
                                                                        ("admin1", "admin")]  # fmt: skip
    bad = tmp_path / "u.json"
    bad.write_text('{"users": [{"username": "x", "role": "analyst", "password": "p"}]}')
    with pytest.raises(users.SeedError, match="never stored"):
        users.seed_users(bad)
    assert users.env_key("analyst1") == "VAULTIC_SEED_PASSWORD_ANALYST1"
    assert users.password_for("analyst1", {"VAULTIC_SEED_PASSWORD_ANALYST1": "x" * 12}) == (
        "x" * 12,
        False,
    )
    pw, generated = users.password_for("analyst1", {})
    assert generated and len(pw) >= 20
    with pytest.raises(users.SeedError, match="at least 12"):
        users.password_for("analyst1", {"VAULTIC_SEED_PASSWORD_ANALYST1": "short"})


# ---- on a real PostgreSQL --------------------------------------------------------------------------
@pytest.fixture
def pg(tmp_path):
    import uuid

    from vaultic.infra import db

    try:
        url = db.db_url()
        conn = db.connect(url, timeout=2)
    except Exception:  # noqa: BLE001 - no database here
        pytest.skip("no PostgreSQL reachable (start it with `docker compose up -d`)")
    schema = f"test_{uuid.uuid4().hex[:12]}"
    conn.execute(f"CREATE SCHEMA {schema}")
    conn.execute(f"SET search_path TO {schema}")
    db.migrate(conn, log=lambda *_: None)
    sep = "&" if "?" in url else "?"
    scoped = f"{url}{sep}options=-csearch_path%3D{schema}"
    try:
        yield conn, scoped
    finally:
        conn.execute(f"DROP SCHEMA {schema} CASCADE")
        conn.close()


@pytest.mark.integration
def test_the_seed_command_creates_users_once_and_prints_a_generated_password_once(pg):
    from vaultic.infra import users

    conn, _ = pg
    out = []
    env = {"VAULTIC_SEED_PASSWORD_ADMIN1": "admin-from-env-123"}
    assert users.seed(conn, users.seed_users(), env, out=out.append) == {
        "created": 2,
        "reset": 0,
        "unchanged": 0,
    }
    shown = [x for x in out if "shown once" in x]
    assert len(shown) == 1 and shown[0].startswith("analyst1") and "from VAULTIC_SEED" in out[1]
    stored = dict(conn.execute("SELECT username, password_hash FROM users").fetchall())
    assert all(h.startswith("$argon2id$") for h in stored.values())
    assert "admin-from-env-123" not in str(stored)
    out.clear()
    assert users.seed(conn, users.seed_users(), env, out=out.append)["unchanged"] == 2
    assert not any("password" in x and "shown" in x for x in out)  # nothing printed again


@pytest.mark.integration
def test_login_refresh_reuse_and_logout_on_postgresql(pg):
    conn, url = pg
    for name, role in (("analyst1", "analyst"), ("admin1", "admin")):
        conn.execute("INSERT INTO users (username, password_hash, role) VALUES (%s, %s, %s)",
                     (name, A.hash_password(PASSWORDS[name]), role))  # fmt: skip
    now = [time.time()]
    svc = A.AuthService(A.PgStore(url), SECRET, clock=lambda: now[0])
    t = svc.login("analyst1", PASSWORDS["analyst1"])
    assert svc.verify_access(t.access).username == "analyst1"
    t2 = svc.refresh(t.refresh)
    assert t2.refresh != t.refresh
    with pytest.raises(A.AuthError) as tab:  # A5: within the grace, another tab's refresh
        svc.refresh(t.refresh)
    assert tab.value.status == 409
    now[0] += A.REFRESH_GRACE + 1
    with pytest.raises(A.AuthError, match="already used"):
        svc.refresh(t.refresh)  # reuse: the family is revoked
    with pytest.raises(A.AuthError):
        svc.refresh(t2.refresh)
    revoked = conn.execute(
        "SELECT count(*) FROM refresh_tokens WHERE revoked_at IS NULL"
    ).fetchone()[0]
    assert revoked == 0
    t3 = svc.login("admin1", PASSWORDS["admin1"])
    assert svc.logout(t3.refresh).username == "admin1"
    with pytest.raises(A.AuthError):
        svc.refresh(t3.refresh)
    # B1 on PostgreSQL (migration 0003): refreshing keeps the session start; 12 h after login it ends
    t4 = svc.login("analyst1", PASSWORDS["analyst1"])
    start = now[0]
    for _ in range(3):  # refresh every 4 hours
        now[0] += 4 * 3600 - 60
        t4 = svc.refresh(t4.refresh)
    rec = svc.store.refresh_by_hash(A.token_hash(t4.refresh))
    assert abs(rec.session_start - start) < 1 and rec.expires_at <= start + A.SESSION_TTL + 1
    now[0] = start + A.SESSION_TTL + 1
    with pytest.raises(A.AuthError, match="12-hour limit"):
        svc.refresh(t4.refresh)
    hashes = [r[0] for r in conn.execute("SELECT token_hash FROM refresh_tokens").fetchall()]
    assert (
        t3.refresh not in hashes and A.token_hash(t3.refresh) in hashes
    )  # only the hash is stored


def test_with_login_on_only_the_allowed_dashboard_origins_get_cors_headers(jwt_app):
    app, _, _ = jwt_app
    c = TestClient(app)
    ok = c.get("/api/auth/mode", headers={"Origin": "http://127.0.0.1:5173"})
    assert ok.headers.get("access-control-allow-origin") == "http://127.0.0.1:5173"
    assert ok.headers.get("access-control-allow-credentials") == "true"
    other = c.get("/api/auth/mode", headers={"Origin": "http://evil.example"})
    assert (
        "access-control-allow-origin" not in other.headers
    )  # the demo app's '*' is removed (D236)
    pre = c.options("/api/auth/login", headers={"Origin": "http://evil.example",
                                                "Access-Control-Request-Method": "POST"})  # fmt: skip
    assert pre.status_code == 400
    denied = c.get("/api/meta", headers={"Origin": "http://127.0.0.1:5173"})
    assert denied.status_code == 401 and denied.headers.get(
        "access-control-allow-origin"
    )  # readable 401


def test_the_frozen_demo_server_is_unchanged_and_has_no_login_routes(tmp_path):
    """D236: demo_server.py is E20-frozen; the plain no-Docker demo has no /api/auth routes, and the
    dashboard treats that as the no-login demo mode."""
    import yaml

    from vaultic.eval.e20 import lf_sha256

    frozen = yaml.safe_load((ROOT / "experiments/configs/EXP-320.yaml").read_text("utf-8"))
    assert (
        lf_sha256(ROOT / "services/api/demo_server.py")
        == frozen["frozen"]["files"]["services/api/demo_server.py"]
    )
    from vaultic.audit.store import SQLiteAuditStore

    mod = _load_module()
    bundle = _bundle(tmp_path)
    _app(mod, bundle)
    plain = mod.create_app(bundle, runs_dir=bundle.parent / "runs",
                           audit_store=SQLiteAuditStore(tmp_path / "p.sqlite"))  # fmt: skip
    c = TestClient(plain)
    assert c.get("/api/auth/mode").status_code == 404
    assert c.get("/api/meta").status_code == 200


# ---- 3.2a: the fixes of the post-merge review of PR #18 (D238) ---------------------------------------
def test_a1_the_mode_is_read_from_dot_env_and_only_empty_or_jwt_is_accepted(monkeypatch, tmp_path):
    import os

    env_file = tmp_path / "main.env"
    env_file.write_text("VAULTIC_AUTH=jwt\n", "utf-8")
    monkeypatch.delenv("VAULTIC_AUTH", raising=False)
    monkeypatch.setenv("VAULTIC_ENV_FILE", str(env_file))
    try:
        assert A.auth_mode() == "jwt"  # set only in .env: login is on, not silently off
    finally:
        os.environ.pop("VAULTIC_AUTH", None)
    assert A.auth_mode({}) == A.auth_mode({"VAULTIC_AUTH": "  "}) == "demo"
    assert A.auth_mode({"VAULTIC_AUTH": " JWT "}) == "jwt"
    for bad in ("on", "true", "jwt1", "1", "yes"):
        with pytest.raises(RuntimeError, match="must be empty .* or 'jwt'"):
            A.auth_mode({"VAULTIC_AUTH": bad})


def test_a1_the_active_mode_is_announced_at_start_up(tmp_path):
    from vaultic.audit.store import SQLiteAuditStore

    mod = _load_module()
    bundle = _bundle(tmp_path)
    _app(mod, bundle)
    said = []
    for auth in (None, _service([0.0])):
        demo = mod.create_app(bundle, runs_dir=bundle.parent / "runs",
                              audit_store=SQLiteAuditStore(tmp_path / f"{len(said)}.sqlite"))  # fmt: skip
        _platform().with_login(demo, auth, announce=said.append)
    assert said[0].startswith("Vaultic login: OFF") and said[1] == "Vaultic login: ON (jwt)"


def test_a2_parallel_wrong_guesses_check_no_more_passwords_than_the_limit(monkeypatch):
    import threading

    svc = _service([1_000_000.0])
    checked, lock = [], threading.Lock()
    real = A.verify_password

    def counting(stored, password):
        with lock:
            checked.append(password)
        return real(stored, password)

    monkeypatch.setattr(A, "verify_password", counting)
    results = []

    def guess(i):
        try:
            svc.login("analyst1", f"wrong-{i}", ip=f"10.0.0.{i}")  # spread over IPs: the user limit
            results.append(200)
        except A.AuthError as e:
            results.append(e.status)

    threads = [threading.Thread(target=guess, args=(i,)) for i in range(30)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(checked) == A.FAIL_PER_USER  # 5, not 30
    assert sorted(results) == [401] * A.FAIL_PER_USER + [429] * (30 - A.FAIL_PER_USER)


def test_a2_a_success_gives_its_attempt_back_and_resets_the_user():
    lim = A.RateLimiter(clock=lambda: 0.0, per_user=2, per_ip=3)
    wait, t1 = lim.attempt("u", "ip")
    assert wait is None
    lim.succeeded(t1)
    for _ in range(2):
        assert lim.attempt("u", "ip")[0] is None
    assert lim.attempt("u", "ip")[0] is not None  # 2 per user (the refused one is not counted)
    assert (
        lim.attempt("other", "ip")[0] is None
    )  # the IP's 3rd attempt: the success gave its own back
    assert lim.attempt("third", "ip")[0] is not None  # 3 per IP


def test_a3_x_forwarded_for_cannot_change_the_client_ip():
    import socket
    import threading
    import time
    import urllib.request

    import uvicorn

    from vaultic.infra import serve

    async def echo(scope, receive, send):
        if scope["type"] != "http":
            return
        body = (scope["client"][0] if scope.get("client") else "none").encode()
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"text/plain")],
            }
        )
        await send({"type": "http.response.body", "body": body})

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    cfg = serve.config(echo, port=port)
    assert cfg.proxy_headers is False and cfg.forwarded_allow_ips == "" and cfg.root_path == ""
    server = uvicorn.Server(cfg)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        for _ in range(100):
            if server.started:
                break
            time.sleep(0.05)
        req = urllib.request.Request(f"http://127.0.0.1:{port}/",
                                     headers={"X-Forwarded-For": "203.0.113.9", "X-Real-IP": "203.0.113.9"})  # fmt: skip
        assert urllib.request.urlopen(req, timeout=5).read().decode() == "127.0.0.1"
    finally:
        server.should_exit = True
        thread.join(timeout=10)


def test_a3_every_documented_uvicorn_command_turns_proxy_headers_off():
    import re

    for doc in ("docs/PLATFORM.md", "services/api/platform_server.py"):
        text = (ROOT / doc).read_text("utf-8")
        for cmd in re.findall(r"uvicorn services\.api\.platform_server:app[^\n`]*", text):
            assert "--no-proxy-headers" in cmd, f"{doc}: {cmd}"


@pytest.mark.parametrize("bad", ["*", "http://127.0.0.1:5173,*", "http://x.example/", "http://x.example/path",
                                 "127.0.0.1:5173", "ftp://x.example", "https://*.example.com", " , "])  # fmt: skip
def test_a4_a_wildcard_or_malformed_cors_origin_refuses_the_start(bad, tmp_path, monkeypatch):
    auth_api = _platform().auth_api
    with pytest.raises(ValueError, match="VAULTIC_CORS_ORIGINS"):
        auth_api.cors_origins({"VAULTIC_CORS_ORIGINS": bad})
    monkeypatch.setenv("VAULTIC_CORS_ORIGINS", bad)
    with pytest.raises(ValueError):
        _app_with(_load_module(), tmp_path, _service([0.0]))


def test_a4_good_origins_are_accepted():
    auth_api = _platform().auth_api
    assert auth_api.cors_origins({}) == ["http://127.0.0.1:5173", "http://localhost:5173"]
    assert auth_api.cors_origins({"VAULTIC_CORS_ORIGINS": "https://demo.example:8443, http://[::1]:5173"}) == [
        "https://demo.example:8443", "http://[::1]:5173"]  # fmt: skip


def test_a5_concurrent_refresh_logout_and_real_reuse_are_told_apart(jwt_app):
    app, svc, now = jwt_app
    c = TestClient(app)
    _login(c, "analyst1")
    old = c.cookies.get("vaultic_refresh")
    assert c.post("/api/auth/refresh", headers={"X-Vaultic-CSRF": "1"}).status_code == 200
    new = c.cookies.get("vaultic_refresh")
    # a second tab refreshes with the old token a moment later: 409, nothing revoked, cookie kept
    tab2 = TestClient(app)
    tab2.cookies.set("vaultic_refresh", old, path="/api/auth")
    r = tab2.post("/api/auth/refresh", headers={"X-Vaultic-CSRF": "1"})
    assert r.status_code == 409 and "set-cookie" not in r.headers
    assert (
        c.post("/api/auth/refresh", headers={"X-Vaultic-CSRF": "1"}).status_code == 200
    )  # still on
    # a refresh after logout (the current token, revoked without a successor): refused, not reuse
    current = c.cookies.get("vaultic_refresh")
    assert current and current != new
    c.post("/api/auth/logout", headers={"X-Vaultic-CSRF": "1"})
    late = TestClient(app)
    late.cookies.set("vaultic_refresh", current, path="/api/auth")
    assert late.post("/api/auth/refresh", headers={"X-Vaultic-CSRF": "1"}).status_code == 401
    admin = _bearer(_login(TestClient(app), "admin1"))
    actions = [e["action"] for e in TestClient(app).get("/api/audit", headers=admin).json()]
    assert "refresh.reuse" not in actions  # no false reuse entries
    # real reuse: an old rotated token presented after the grace window
    c3 = TestClient(app)
    _login(c3, "analyst1")
    first = c3.cookies.get("vaultic_refresh")
    c3.post("/api/auth/refresh", headers={"X-Vaultic-CSRF": "1"})
    now[0] += A.REFRESH_GRACE + 1
    thief = TestClient(app)
    thief.cookies.set("vaultic_refresh", first, path="/api/auth")
    assert thief.post("/api/auth/refresh", headers={"X-Vaultic-CSRF": "1"}).status_code == 401
    assert (
        c3.post("/api/auth/refresh", headers={"X-Vaultic-CSRF": "1"}).status_code == 401
    )  # family ended
    actions = [e["action"] for e in TestClient(app).get("/api/audit", headers=admin).json()]
    assert actions.count("refresh.reuse") == 1


def test_b1_a_session_ends_12_hours_after_login_however_often_it_is_refreshed():
    now = [1_000_000.0]
    svc = _service(now)
    t = svc.login("analyst1", PASSWORDS["analyst1"])
    start = now[0]
    for _ in range(3):  # refresh every 4 hours
        now[0] += 4 * 3600 - 60
        t = svc.refresh(t.refresh)
    rec = svc.store.refresh_by_hash(A.token_hash(t.refresh))
    assert rec.expires_at <= start + A.SESSION_TTL and rec.session_start == start
    now[0] = start + A.SESSION_TTL + 1
    with pytest.raises(A.AuthError, match="12-hour limit"):
        svc.refresh(t.refresh)


def test_b4_a_failed_login_keeps_the_typed_name_only_if_it_is_a_user(jwt_app):
    app, _, _ = jwt_app
    c = TestClient(app)
    c.post("/api/auth/login", json={"username": "MyS3cretPassw0rd", "password": "x"})
    c.post("/api/auth/login", json={"username": "admin1", "password": "wrong-password"})
    admin = _bearer(_login(c, "admin1"))
    failed = [e for e in c.get("/api/audit", headers=admin).json() if e["action"] == "login.failed"]
    assert [e["target"] for e in failed] and {e["target"] for e in failed} == {None, "admin1"}
    assert "MyS3cretPassw0rd" not in c.get("/api/audit/export.csv", headers=admin).text


def test_b7_a_root_path_can_never_skip_the_role_check(jwt_app, tmp_path):
    for app in (jwt_app[0], _app_with(_load_module(), tmp_path, None)):
        c = TestClient(app, root_path="/x")
        for path in ("/x/api/meta", "/api/meta", "/x/api/audit/export.csv"):
            r = c.get(path)
            assert r.status_code == 500 and "root_path" in r.text, path
