"""Attack Lab endpoints (Phase B): synthetic attack scenarios run live through the Test Lab engine.
Synthetic scenario - not the evaluation. Mounted with the Test Lab (VAULTIC_TESTLAB=1); answers 503
until the Test Lab's parity gate has passed.

  GET  /api/attack/scenarios   scenarios, intensities, notes, threshold sets
  POST /api/attack/run         {"scenario", "intensity", "seed", "model": true|false}: one run
  GET  /api/attack/suite       the 5-seed x 3-intensity suite scorecard and post-hoc diagnostics
"""

from __future__ import annotations

from fastapi import APIRouter, Body, HTTPException, Request


def make_router(lab, audit=None) -> APIRouter:
    """audit: the persistent audit store (D208); every run is recorded."""
    from vaultic.audit.http import AUDIT_FAILED, record_open
    from vaultic.live.service import NotReady
    from vaultic.scenarios.service import AttackInputError

    router = APIRouter()

    @router.get("/api/attack/scenarios")
    def scenarios():
        return lab.scenarios()

    @router.post("/api/attack/run")
    def run(request: Request, body: dict = Body(...)):  # noqa: B008
        try:
            out = lab.run(body)
            detail = (f"intensity {out.get('intensity')}, seed {out.get('seed')}, generator "
                      f"{out.get('generator')}, model {out.get('model')}")  # fmt: skip
            # fail-open (D209): the run is returned either way; a failed audit write is flagged
            if not record_open(audit, request, "Attack Lab: ran a scenario",
                               target=out.get("scenario"), detail=detail):  # fmt: skip
                out["audit"] = AUDIT_FAILED
            return out
        except NotReady as e:
            raise HTTPException(503, f"Test Lab not ready: {e}") from e
        except AttackInputError as e:
            raise HTTPException(400, str(e)) from e
        except Exception as e:
            raise HTTPException(500, f"internal error: {type(e).__name__}: {e}") from e

    @router.get("/api/attack/suite")
    def suite():
        return lab.suite()

    return router
