"""Copilot endpoints (D224): mounted into the demo backend (services/api/demo_server.create_app) only
with VAULTIC_COPILOT=1. Read-only: the copilot answers questions about one case from its evidence
and never changes a decision, label, threshold or setting; one audit entry per request.

  GET  /api/copilot/status    model, whether it is running, Test Lab state, cached summaries
  POST /api/copilot/ask       {"case_id", "question", "time", "L" (default 30), "stream" (default
                              true)}: NDJSON events (start, token = draft not yet checked, final =
                              the checked answer, or refused / error); stream=false: the final JSON
  POST /api/copilot/summary   {"case_id", "time", "L", "stream"}: the case summary; a cached demo
                              summary is labelled "cached, generated at <time> by <model>"

A model that is not running answers 503 with how to start it (Ollama on 127.0.0.1 only).
"""

from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

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

    def respond(kind: str, body: AskIn, request: Request):
        if body.L not in LABEL_DELAYS:
            raise HTTPException(
                422, f"Label delay must be one of {', '.join(map(str, LABEL_DELAYS))} days."
            )
        try:
            prep = copilot.start(
                kind, body.case_id, body.question, body.time, body.L, actor_of(request)
            )
        except CopilotError as e:
            raise HTTPException(e.status, e.message) from e
        if not body.stream:
            return copilot.answer(prep)
        lines = (json.dumps(ev) + "\n" for ev in copilot.run(prep))
        return StreamingResponse(lines, media_type="application/x-ndjson")

    @router.get("/api/copilot/status")
    def status():
        return copilot.status()

    @router.post("/api/copilot/ask")
    def ask(body: AskIn, request: Request):
        return respond("question", body, request)

    @router.post("/api/copilot/summary")
    def summary(body: AskIn, request: Request):
        return respond("summary", body, request)

    return router
