"""Login and role checks for the demo backend (Sprint 3 step 3.2, D235).

ROUTE_ROLES maps every (method, path template) of every /api route to the role it needs, or to PUBLIC.
AuthMiddleware enforces it for each request and fails closed: an /api request that matches no rule
answers 500 'no role rule'. tests/test_auth.py fails if a route is missing from the table, or if the
table names a route that does not exist.

Two modes:
- jwt (VAULTIC_AUTH=jwt):
  - a bearer access token is required and the role is checked (admin includes analyst);
  - the actor recorded in the audit log comes from the token;
  - the X-Vaultic-Analyst / X-Vaultic-Role headers are ignored.
- demo (the no-Docker demo, D235 Q8):
  - no login;
  - the D208 headers name the actor, and the roles are self-declared labels that are not enforced;
  - the dashboard labels this on screen;
  - the table still applies, so a route without a rule is refused in both modes.
"""

from __future__ import annotations

import logging
import re

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from vaultic.infra.auth import REFRESH_TTL, AuthError, allowed

log = logging.getLogger("vaultic.auth")
PUBLIC = "public"
COOKIE = "vaultic_refresh"
CSRF_HEADER = "X-Vaultic-CSRF"

ROUTE_ROLES: dict[tuple[str, str], str] = {
    # login itself (D235)
    ("GET", "/api/auth/mode"): PUBLIC,
    ("POST", "/api/auth/login"): PUBLIC,
    ("POST", "/api/auth/refresh"): PUBLIC,
    ("POST", "/api/auth/logout"): PUBLIC,
    ("GET", "/api/auth/me"): "analyst",
    # demo backend
    ("GET", "/api/meta"): "analyst",
    ("GET", "/api/range"): "analyst",
    ("GET", "/api/day-summary"): "analyst",
    ("GET", "/api/stream"): "analyst",
    ("GET", "/api/transactions/{tid}"): "analyst",
    ("GET", "/api/alerts"): "analyst",
    ("GET", "/api/customers/{uid}/history"): "analyst",
    ("GET", "/api/network"): "analyst",
    ("GET", "/api/queues"): "analyst",
    ("GET", "/api/drift"): "analyst",
    ("GET", "/api/lab"): "analyst",
    ("POST", "/api/score"): "analyst",
    ("GET", "/api/thresholds"): "analyst",
    (
        "PUT",
        "/api/thresholds",
    ): "analyst",  # refused for everyone (D129), and the refusal is audited
    ("POST", "/api/feedback/{tid}"): "analyst",
    ("GET", "/api/whatif"): "analyst",
    ("GET", "/api/audit"): "analyst",
    ("GET", "/api/audit/verify"): "admin",
    ("GET", "/api/audit/export.csv"): "admin",
    ("POST", "/api/audit/event"): "analyst",
    # Test Lab (D182)
    ("GET", "/api/live/status"): "analyst",
    ("GET", "/api/live/sample"): "analyst",
    ("POST", "/api/score-live"): "analyst",
    ("GET", "/api/live/reveal/{tid}"): "analyst",
    ("GET", "/api/live/latency"): "analyst",
    ("GET", "/api/live/options"): "analyst",
    ("GET", "/api/live/template.csv"): "analyst",
    ("POST", "/api/live/batch"): "analyst",
    # Attack Lab
    ("GET", "/api/attack/scenarios"): "analyst",
    ("POST", "/api/attack/run"): "admin",
    ("GET", "/api/attack/suite"): "admin",
    # copilot (D224)
    ("GET", "/api/copilot/status"): "analyst",
    ("GET", "/api/copilot/facts"): "analyst",
    ("POST", "/api/copilot/ask"): "analyst",
    ("POST", "/api/copilot/summary"): "analyst",
}


def _pattern(template: str) -> re.Pattern:
    parts = re.split(r"(\{[^}]+\})", template)
    return re.compile(
        "^" + "".join("[^/]+" if p.startswith("{") else re.escape(p) for p in parts) + "$"
    )


_RULES = [(m, _pattern(t), t, role) for (m, t), role in ROUTE_ROLES.items()]


def rule_for(method: str, path: str) -> tuple[str, str] | None:
    """(template, role) of the rule for this request, or None (fail closed)."""
    method = "GET" if method == "HEAD" else method
    for m, pat, template, role in _RULES:
        if m == method and pat.match(path):
            return template, role
    return None


def _json(status: int, detail: str, headers: dict | None = None) -> JSONResponse:
    return JSONResponse({"detail": detail}, status_code=status, headers=headers)


class AuthMiddleware:
    """Enforces ROUTE_ROLES on every /api request (pure ASGI, so nested routers are covered too)."""

    def __init__(self, app, service=None):
        self.app, self.service = app, service

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not scope["path"].startswith("/api"):
            return await self.app(scope, receive, send)
        state = scope.setdefault("state", {})
        state["auth_mode"] = "jwt" if self.service is not None else "demo"
        if scope["method"] == "OPTIONS":  # CORS preflight, answered by the CORS middleware
            return await self.app(scope, receive, send)
        rule = rule_for(scope["method"], scope["path"])
        if rule is None:
            log.error("no role rule for %s %s", scope["method"], scope["path"])
            return await _json(500, "no role rule for this route")(scope, receive, send)
        _template, needed = rule
        if needed == PUBLIC or self.service is None:  # demo mode: roles are labels (D235 Q8)
            return await self.app(scope, receive, send)
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope["headers"]}
        auth = headers.get("authorization", "")
        if not auth.lower().startswith("bearer "):
            return await _json(401, "log in first", {"WWW-Authenticate": "Bearer"})(
                scope, receive, send
            )
        try:
            user = self.service.verify_access(auth[7:].strip())
        except AuthError as e:
            return await _json(e.status, e.message, {"WWW-Authenticate": "Bearer"})(
                scope, receive, send
            )
        if not allowed(user.role, needed):
            return await _json(403, f"this needs the {needed} role")(scope, receive, send)
        state["identity"] = (user.username, user.role)
        return await self.app(scope, receive, send)


class LoginIn(BaseModel):
    username: str
    password: str


def _audit(store, analyst, role, action, target=None, detail="") -> None:
    """Fail-open audit write (D209) for the auth events; never the password."""
    if store is None:
        return
    try:
        store.append(analyst, role, action, target=target, detail=detail)
    except Exception:  # noqa: BLE001 - logins never fail because the audit log did
        log.exception("audit write failed for %r", action)


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def make_router(service=None, audit=None) -> APIRouter:
    """The /api/auth routes. With service None (demo mode) login is off and says so."""
    router = APIRouter()
    mode = "jwt" if service is not None else "demo"

    def need_jwt():
        if service is None:
            raise HTTPException(
                409, "login is off in demo mode (start the API with VAULTIC_AUTH=jwt)"
            )

    def need_csrf(request: Request):
        if request.headers.get(CSRF_HEADER) != "1":
            raise HTTPException(403, f"missing the {CSRF_HEADER} header")

    def with_cookie(body: dict, refresh: str | None) -> JSONResponse:
        res = JSONResponse(body)
        if refresh is None:
            res.delete_cookie(COOKIE, path="/api/auth", httponly=True, samesite="strict")
        else:  # no TLS on localhost, so not Secure (stated in D235)
            res.set_cookie(COOKIE, refresh, max_age=REFRESH_TTL, path="/api/auth", httponly=True,
                           samesite="strict", secure=False)  # fmt: skip
        return res

    def token_body(t) -> dict:
        return {"accessToken": t.access, "expiresIn": t.access_expires_in,
                "user": {"username": t.user.username, "role": t.user.role}}  # fmt: skip

    @router.get("/api/auth/mode")
    def get_mode():
        return {"mode": mode, "demoLabel": None if service is not None else
                "Demo mode: no login; names and roles are self-declared"}  # fmt: skip

    @router.post("/api/auth/login")
    def login(body: LoginIn, request: Request):
        need_jwt()
        try:
            t = service.login(body.username, body.password, _client_ip(request))
        except AuthError as e:
            _audit(audit, "unknown", "unknown", "login.failed", target=body.username[:60],
                   detail=e.message)  # fmt: skip
            headers = {"Retry-After": str(e.retry_after)} if e.retry_after else None
            raise HTTPException(e.status, e.message, headers=headers) from None
        _audit(audit, t.user.username, t.user.role, "login", target=t.user.username)
        return with_cookie(token_body(t), t.refresh)

    @router.post("/api/auth/refresh")
    def refresh(request: Request):
        need_jwt()
        need_csrf(request)
        try:
            t = service.refresh(request.cookies.get(COOKIE, ""))
        except AuthError as e:
            uid = getattr(e, "reuse_user_id", None)
            if uid is not None:
                u = service.store.user(uid)
                _audit(audit, u.username if u else "unknown", u.role if u else "unknown",
                       "refresh.reuse", target=u.username if u else None,
                       detail="an already rotated refresh token was presented; the session family "
                       "was revoked")  # fmt: skip
            res = _json(e.status, e.message)
            res.delete_cookie(COOKIE, path="/api/auth", httponly=True, samesite="strict")
            return res
        return with_cookie(token_body(t), t.refresh)

    @router.post("/api/auth/logout")
    def logout(request: Request):
        need_jwt()
        need_csrf(request)
        user = service.logout(request.cookies.get(COOKIE, ""))
        if user is not None:
            _audit(audit, user.username, user.role, "logout", target=user.username)
        return with_cookie({"loggedOut": True}, None)

    @router.get("/api/auth/me")
    def me(request: Request):
        from vaultic.audit.http import actor_of

        name, role = actor_of(request)
        return {"mode": mode, "username": name, "role": role}

    return router


def cors_origins(env) -> list[str]:
    raw = env.get("VAULTIC_CORS_ORIGINS") or "http://127.0.0.1:5173,http://localhost:5173"
    return [o.strip() for o in raw.split(",") if o.strip()]
