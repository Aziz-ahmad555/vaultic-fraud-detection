"""Warm the Test Lab's caches on drive E once, so the next start is fast (D220).

Run:  python -m vaultic.live.warm      (repository root; same environment as the backend)

Checks the inputs, loads the frozen system, builds (or loads) the history store and saves it
under VAULTIC_CACHE_DIR/testlab/, then runs (or reads) the parity check (D183, D189) and caches
its result. Prints the time of each step. A start of the backend afterwards loads both from the
cache. Run it again after a merge that changes the live or feature code or the data.
"""

from __future__ import annotations

import json
import sys
import time


def warm(log=print, build=None, parity=None) -> dict:
    from vaultic.live.engine import STAGE2_DEV_SYSTEM, SYSTEM_CACHE, build_engine
    from vaultic.live.parity import cached_parity

    started = time.perf_counter()
    engine = (build or build_engine)(log)
    engine_s = time.perf_counter() - started
    path = SYSTEM_CACHE if SYSTEM_CACHE.exists() else STAGE2_DEV_SYSTEM
    result = (parity or cached_parity)(engine, path, log)
    total_s = time.perf_counter() - started
    out = {"engine_seconds": round(engine_s, 1), "parity_seconds": round(total_s - engine_s, 1),
           "total_seconds": round(total_s, 1), "parity_cached": bool(result.get("cached")),
           "parity_passed": bool(result.get("passed")),
           "store": {k: v for k, v in getattr(engine.store, "info", {}).items()
                     if k in ("cache", "load_seconds", "saved_seconds", "built_seconds")}}  # fmt: skip
    log(json.dumps(out, indent=1))
    return out


def main() -> int:
    out = warm(log=lambda m: print(m, flush=True))
    return 0 if out["parity_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
