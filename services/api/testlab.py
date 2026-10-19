"""Test Lab endpoints (D182-D186): live scoring of one transaction by the frozen stage-2 system.

Mounted into the demo backend (services/api/demo_server.create_app) when VAULTIC_TESTLAB=1; the
engine loads in the background at startup (status endpoint), and every scoring endpoint answers
503 with the reason until the parity gate has passed (refused if it failed).

  GET  /api/live/status              state, message, parity result, latency p50 / p95
  GET  /api/live/sample?filter=&seed=  a random held-out test transaction, label hidden
  POST /api/score-live               {"transaction_id": id} or {"custom": {...}}: one live score
  GET  /api/live/reveal/{id}         the label of a test transaction (after scoring)
  GET  /api/live/latency             p50 / p95 / max over the session
  GET  /api/live/options             allowed values for the custom form
  GET  /api/live/template.csv        the batch template (IEEE-CIS columns)
  POST /api/live/batch               a CSV body; results, metrics and the scored CSV
"""

from __future__ import annotations

from fastapi import APIRouter, Body, HTTPException, Request
from fastapi.responses import PlainTextResponse


def make_router(service, audit=None) -> APIRouter:
    """audit: the persistent audit store (D208); every score, reveal and batch is recorded."""
    from vaultic.audit.http import AUDIT_FAILED, record_open
    from vaultic.live.batch import BatchError
    from vaultic.live.custom import CustomInputError
    from vaultic.live.service import FILTERS, NotFound, NotReady

    router = APIRouter()

    def record(request, out, action, target=None, after=None, detail=""):
        """Fail-open (D209): the result is returned either way; a failed audit write is flagged."""
        if not record_open(audit, request, action, target=target, after=after, detail=detail):
            if isinstance(out, dict):
                out["audit"] = AUDIT_FAILED
        return out

    def guard(fn, *a, **k):
        try:
            return fn(*a, **k)
        except NotReady as e:
            raise HTTPException(503, f"Test Lab not ready: {e}") from e
        except (BatchError, CustomInputError) as e:
            raise HTTPException(400, str(e)) from e
        except NotFound as e:
            raise HTTPException(404, str(e)) from e
        except Exception as e:  # an internal error: reported as such, never as "not found"
            raise HTTPException(500, f"internal error: {type(e).__name__}: {e}") from e

    @router.get("/api/live/status")
    def status():
        return service.status()

    @router.get("/api/live/sample")
    def sample(filter: str = "any", seed: int | None = None):  # noqa: A002
        if filter not in FILTERS:
            raise HTTPException(400, f"filter must be one of {FILTERS}")
        return guard(service.sample, filter, seed)  # seed: reproducible draws (benchmark)

    @router.post("/api/score-live")
    def score_live(request: Request, body: dict = Body(...)):  # noqa: B008
        if "transaction_id" in body:
            tid = int(body["transaction_id"])
            out = guard(service.score_real, tid)
            return record(request, out, "Test Lab: scored a real test transaction",
                          target=f"tx:{tid}", detail=f"decision {out.get('decision')}")  # fmt: skip
        if "custom" in body and isinstance(body["custom"], dict):
            out = guard(service.score_custom, body["custom"])
            c = body["custom"]
            return record(request, out, "Test Lab: scored a custom transaction",
                          target=f"customer:{c.get('customer') or 'new'}",
                          detail=f"amount {c.get('amount')}, day {c.get('day')}; decision {out.get('decision')}")  # fmt: skip
        raise HTTPException(400, "send {'transaction_id': id} or {'custom': {...}}")

    @router.get("/api/live/reveal/{tid}")
    def reveal(tid: int, request: Request):
        out = guard(service.reveal, tid)
        return record(request, out, "Test Lab: revealed a label", target=f"tx:{tid}",
                      after=out.get("isFraud"))  # fmt: skip

    @router.get("/api/live/latency")
    def latency():
        e = guard(service._ready)
        return e.latency.summary()

    @router.get("/api/live/options")
    def options():
        return guard(service.options)

    @router.get("/api/live/template.csv", response_class=PlainTextResponse)
    def template():
        return PlainTextResponse(guard(service.template), media_type="text/csv",
                                 headers={"Content-Disposition": "attachment; filename=testlab_template.csv"})  # fmt: skip

    @router.post("/api/live/batch")
    async def batch(request: Request):
        raw = await request.body()
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError as e:
            raise HTTPException(400, "the file must be UTF-8 text (CSV)") from e
        out = guard(service.score_batch, text)
        rows = len(out.get("rows", [])) if isinstance(out, dict) else None
        return record(request, out, "Test Lab: scored and exported a batch file", target="batch",
                      detail=f"{rows} rows")  # fmt: skip

    return router
