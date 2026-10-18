"""Attack Lab endpoints (Phase B): synthetic attack scenarios run live through the Test Lab engine.
Synthetic scenario - not the evaluation. Mounted with the Test Lab (VAULTIC_TESTLAB=1); answers 503
until the Test Lab's parity gate has passed.

  GET  /api/attack/scenarios   scenarios, intensities, notes, threshold sets
  POST /api/attack/run         {"scenario", "intensity", "seed", "model": true|false}: one run
  GET  /api/attack/suite       the 5-seed x 3-intensity suite scorecard and post-hoc diagnostics
"""

from __future__ import annotations

from fastapi import APIRouter, Body, HTTPException


def make_router(lab) -> APIRouter:
    from vaultic.live.service import NotReady
    from vaultic.scenarios.service import AttackInputError

    router = APIRouter()

    @router.get("/api/attack/scenarios")
    def scenarios():
        return lab.scenarios()

    @router.post("/api/attack/run")
    def run(body: dict = Body(...)):  # noqa: B008
        try:
            return lab.run(body)
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
