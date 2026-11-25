"""The acting analyst of an HTTP request (D208, D209, D235). With login on (VAULTIC_AUTH=jwt) it is
the logged-in user. In the no-login demo mode the dashboard sends a named analyst and a role with every
request: self-declared by the browser and not authenticated, so attribution, not proof. A request
without them is recorded as 'unknown'."""

from __future__ import annotations

import logging
from urllib.parse import unquote

ANALYST_HEADER = "X-Vaultic-Analyst"
ROLE_HEADER = "X-Vaultic-Role"
AUDIT_FAILED = "failed"  # the response flag of a fail-open action whose audit write failed (D209)
# client-side actions the server cannot see; the dashboard reports them explicitly
CLIENT_EVENTS = {
    "report_printed": "Printed or saved the daily report",
    "whatif_applied": "Applied a what-if cost setting",
    "whatif_cleared": "Cleared the what-if cost setting",
}
log = logging.getLogger("vaultic.audit")


def actor_of(request, fallback_role: str | None = None) -> tuple[str | None, str | None]:
    """(analyst, role) of a request. With login on (D235) it is the logged-in user, set by the auth
    middleware, and the headers are ignored. In the no-login demo mode it comes from the request
    headers; header values are ASCII, so the dashboard percent-encodes a non-ASCII name and it is
    decoded here ("Zo%C3%AB" -> "Zoë")."""
    state = getattr(request, "state", None)
    identity = getattr(state, "identity", None) if state is not None else None
    if identity:
        return identity
    if state is not None and getattr(state, "auth_mode", None) == "jwt":
        return None, None  # login on but no user (a public route): never trust the headers
    headers = getattr(request, "headers", None) or {}
    name = headers.get(ANALYST_HEADER)
    if name and "%" in name:
        name = unquote(name)
    return name, headers.get(ROLE_HEADER) or fallback_role


def record_open(store, request, action: str, **kw) -> bool:
    """Fail-open audit write (D209): True when recorded; on failure the error is logged and False
    is returned so the caller can mark its response audit = 'failed'. No store = nothing to do."""
    if store is None:
        return True
    try:
        analyst, role = actor_of(request, kw.pop("fallback_role", None))
        store.append(analyst, role, action, **kw)
        return True
    except Exception:  # never block a fail-open action on the audit log
        log.exception("audit write failed for %r", action)
        return False


def public(entry: dict) -> dict:
    """An entry as the API returns it (before / after decoded from their JSON text)."""
    import json

    out = dict(entry)
    for k in ("before", "after"):
        out[k] = None if out.get(k) is None else json.loads(out[k])
    return out
