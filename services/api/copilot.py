"""Copilot endpoints (D224): mounted into the demo backend (services/api/demo_server.create_app) only
with VAULTIC_COPILOT=1. Read-only: the copilot answers questions about one case from its evidence
and never changes a decision, label, threshold or setting; one audit entry per request.

  GET  /api/copilot/status    model, whether it is running and loaded, Test Lab state, cached summaries
  GET  /api/copilot/facts     ?case_id=&time=&L=: the case's key facts by code (no model, D226)
  POST /api/copilot/ask       {"case_id", "question", "time", "L" (default 30), "stream" (default
                              true)}: NDJSON events (start, token = draft not yet checked, final =
                              the checked answer, or refused / error); stream=false: the final JSON
  POST /api/copilot/summary   {"case_id", "time", "L", "stream"}: the case summary; a cached demo
                              summary is labelled "cached, generated at <time> by <model>"

A model that is not running answers 503 with how to start it (Ollama on 127.0.0.1 only). The body is
parsed here, not by FastAPI, so an invalid body or label delay also leaves its one audit entry
(D224d (6)).
"""

from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ValidationError

LABEL_DELAYS = (0, 7, 30, 60)  # the dashboard's label delays (D125)


class AskIn(BaseModel):
    case_id: int
    question: str = ""
    time: float
    L: int = 30
    stream: bool = True


def make_router(copilot) -> APIRouter:
    from vaultic.audit.http import actor_of
    from vaultic.copilot.service import CopilotError

    router = APIRouter()

    def respond(kind: str, raw: bytes, request: Request):
        actor, data = actor_of(request), None
        try:
            data = json.loads(raw or b"null")
            body = AskIn.model_validate(data)
        except (ValueError, ValidationError) as e:
            d = data if isinstance(data, dict) else {}
            copilot.reject(kind, d.get("case_id"), d.get("question"), actor, "body")
            raise HTTPException(422, f"Invalid request body: {type(e).__name__}.") from e
        if body.L not in LABEL_DELAYS:
            copilot.reject(kind, body.case_id, body.question, actor, f"label delay {body.L}")
            raise HTTPException(
                422, f"Label delay must be one of {', '.join(map(str, LABEL_DELAYS))} days."
            )
        try:
            prep = copilot.start(kind, body.case_id, body.question, body.time, body.L, actor)
        except CopilotError as e:
            raise HTTPException(e.status, e.message) from e
        if not body.stream:
            return copilot.answer(prep)
        lines = (json.dumps(ev) + "\n" for ev in copilot.run(prep))
        return StreamingResponse(lines, media_type="application/x-ndjson")

    @router.get("/api/copilot/status")
    def status():
        return copilot.status()

    @router.get("/api/copilot/facts")
    def facts(case_id: int, time: float, L: int = 30):
        """The case's key facts, by code from the copilot's tools; no model (D226)."""
        if L not in LABEL_DELAYS:
            raise HTTPException(
                422, f"Label delay must be one of {', '.join(map(str, LABEL_DELAYS))} days."
            )
        try:
            return copilot.facts(case_id, time, L)
        except CopilotError as e:
            raise HTTPException(e.status, e.message) from e

    @router.post("/api/copilot/ask")
    async def ask(request: Request):
        return await run_in_threadpool(respond, "question", await request.body(), request)

    @router.post("/api/copilot/summary")
    async def summary(request: Request):
        return await run_in_threadpool(respond, "summary", await request.body(), request)

    return router
