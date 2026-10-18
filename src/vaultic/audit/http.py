"""The acting analyst of an HTTP request (D208): the dashboard sends a named analyst and a role
with every request. A request without them is recorded as 'unknown' (no sign-in yet; JWT later)."""

from __future__ import annotations

ANALYST_HEADER = "X-Vaultic-Analyst"
ROLE_HEADER = "X-Vaultic-Role"
# client-side actions the server cannot see; the dashboard reports them explicitly
CLIENT_EVENTS = {
    "report_printed": "Printed or saved the daily report",
    "whatif_applied": "Applied a what-if cost setting",
    "whatif_cleared": "Cleared the what-if cost setting",
}


def actor_of(request, fallback_role: str | None = None) -> tuple[str | None, str | None]:
    """(analyst, role) from the request headers; cleaned by the store (clean_actor)."""
    headers = getattr(request, "headers", None) or {}
    return headers.get(ANALYST_HEADER), headers.get(ROLE_HEADER) or fallback_role


def public(entry: dict) -> dict:
    """An entry as the API returns it (before / after decoded from their JSON text)."""
    import json

    out = dict(entry)
    for k in ("before", "after"):
        out[k] = None if out.get(k) is None else json.loads(out[k])
    return out
