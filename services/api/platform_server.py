"""The demo backend with login (Sprint 3 step 3.2, D235, D236).

    VAULTIC_AUTH=jwt python -m vaultic.infra.serve        login on
    python -m vaultic.infra.serve                         no-login demo mode
    (or uvicorn services.api.platform_server:app --port 8000 --no-proxy-headers --forwarded-allow-ips "")

Client IPs are never taken from proxy headers (D238, A3); VAULTIC_AUTH must be empty or 'jwt', read from
the environment and .env (D238, A1); the active mode is printed at start-up.

services/api/demo_server.py is part of E20's frozen code, so it stays byte-identical and is wrapped here,
not changed (D236). with_login(app, auth) adds:
- the role table of services/api/auth.py (every /api route needs its role or is PUBLIC; a route without a
  rule is refused), in both modes;
- the /api/auth routes;
- with login on: CORS for the allowed dashboard origins only, with credentials, and the demo app's own
  wildcard CORS header removed.

The no-Docker demo can still run `uvicorn services.api.demo_server:app` exactly as before.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

from fastapi import HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRoute
from pydantic import BaseModel

HERE = Path(__file__).resolve().parent


def _load(name: str, file: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # pydantic resolves the request models' annotations through it
    spec.loader.exec_module(mod)
    return mod


auth_api = _load("auth_api", "auth.py")


class _NoWildcardCORS:
    """With login on, drop the demo app's `Access-Control-Allow-Origin: *`; the outer CORS then
    sets the header only for the allowed origins (D236)."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        async def strip(message):
            if message["type"] == "http.response.start":
                message["headers"] = [
                    (k, v)
                    for k, v in message.get("headers", [])
                    if not (k.lower() == b"access-control-allow-origin" and v == b"*")
                ]
            await send(message)

        return await self.app(scope, receive, strip)


def resolve_auth(auth="env"):
    """An AuthService (login on), None (no-login demo mode), or "env": login on only with
    VAULTIC_AUTH=jwt, which then needs VAULTIC_JWT_SECRET and PostgreSQL (D235)."""
    if isinstance(auth, str) and auth == "env":
        from vaultic.infra.auth import auth_mode, service_from_env

        return service_from_env() if auth_mode() == "jwt" else None
    return auth


def mode_banner(auth) -> str:
    return ("Vaultic login: ON (jwt)" if auth is not None
            else "Vaultic login: OFF - no-login demo mode (names and roles are self-declared)")  # fmt: skip


def _to_stderr(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def with_login(app, auth="env", announce=_to_stderr):
    """Add login and role checks to a demo app (from demo_server.create_app), before it starts.
    Refuses (raises) on a bad VAULTIC_AUTH or VAULTIC_CORS_ORIGINS, before anything is added (D238).
    """
    auth = resolve_auth(auth)
    origins = auth_api.cors_origins(os.environ) if auth is not None else None  # A4: checked first
    if auth is not None:
        app.add_middleware(_NoWildcardCORS)
    app.add_middleware(auth_api.AuthMiddleware, service=auth)
    if auth is not None:  # outermost: answers preflights and covers 401 / 403 responses too
        app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"],
                           allow_headers=["*"], allow_credentials=True)  # fmt: skip
    app.include_router(auth_api.make_router(auth, audit=getattr(app.state, "audit", None)))
    app.state.auth = auth
    with_records(app)
    announce(mode_banner(auth))  # the active mode, clearly, at start-up (D238, A1)
    from vaultic.infra.stores import store_banner

    announce(store_banner(getattr(app.state, "audit", None)))  # and the store (D240)
    return app


class FeedbackIn(BaseModel):
    label: str
    role: str | None = None  # older clients; the role of the request wins (D208)


def with_records(app):
    """With the PostgreSQL audit store (VAULTIC_STORE=postgres, D240), labels persist: the demo's
    in-memory label route is replaced by one that writes the audit entry and the label row in one
    transaction (fail-closed, D209), and GET /api/feedback/{tid} reads the current label and its
    history. With the SQLite log only that GET is added, answering 501 (no persistent labels)."""
    from vaultic.audit.pg_store import LABELS, PostgresAuditStore

    store = getattr(app.state, "audit", None)
    path = "/api/feedback/{tid}"
    if not isinstance(store, PostgresAuditStore):

        @app.get(path)
        def no_persistent_labels(tid: int):
            raise HTTPException(501, "Labels are kept in memory for this session only; persistent "
                                     "labels need VAULTIC_STORE=postgres (D240).")  # fmt: skip

        return app
    from vaultic.audit.http import actor_of
    from vaultic.infra.stores import LabelStore

    labels = LabelStore(store)
    app.router.routes[:] = [r for r in app.router.routes
                            if not (isinstance(r, APIRoute) and r.path == path and "POST" in r.methods)]  # fmt: skip

    def post_feedback(tid: int, body: FeedbackIn, request: Request):
        if body.label not in LABELS:
            raise HTTPException(422, f"label must be one of {', '.join(LABELS)}")
        analyst, role = actor_of(request, body.role)
        try:
            labels.label(analyst, role, tid, body.label)
        except Exception as e:
            raise HTTPException(503, "The label was not saved: the audit log could not be "
                                     f"written ({type(e).__name__}). Try again.") from e  # fmt: skip
        return {"ok": True}

    def get_feedback(tid: int):
        return {"transaction": tid, "current": labels.current(tid), "history": labels.history(tid)}

    app.add_api_route(path, post_feedback, methods=["POST"])
    app.add_api_route(path, get_feedback, methods=["GET"])
    app.state.labels = labels
    return app


demo = _load("demo_server", "demo_server.py")
create_demo_app, mount_testlab, mount_copilot = (
    demo.create_app,
    demo.mount_testlab,
    demo.mount_copilot,
)
# the demo module already built its app from the bundle; wrap that one (the bundle is loaded once)
app = with_login(demo.app) if demo.app is not None else None
