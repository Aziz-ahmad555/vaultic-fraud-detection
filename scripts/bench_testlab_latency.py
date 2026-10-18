"""Reproduce the Test Lab latency figures (D182): 200 live scores of real test transactions.

Start the backend with the Test Lab first (dashboard/README.md, "Running the Test Lab"), wait until
/api/live/status says "ready", keep the machine otherwise idle, then:

    python scripts/bench_testlab_latency.py [--url http://127.0.0.1:8000] [--n 200]

Transactions are drawn with seeds 0..n-1 (/api/live/sample?seed=i), so every run scores the same
transactions. Reported: the server-side scoring latency (features + scoring, as the engine records
it), the explanation time, the end-to-end HTTP request, and how many live scores matched the stored
final-run scores. Standard library only. Results are also written as JSON next to the cache
(VAULTIC_CACHE_DIR, default E:/dev-cache, under testlab/).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
import urllib.request
from datetime import datetime
from pathlib import Path


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolation percentile (numpy's default), without numpy."""
    v = sorted(values)
    k = (len(v) - 1) * q / 100
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--n", type=int, default=200)
    args = parser.parse_args()
    base = args.url.rstrip("/")

    def get(path):
        return json.loads(urllib.request.urlopen(base + path).read())

    def post(path, body):
        req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})  # fmt: skip
        return json.loads(urllib.request.urlopen(req).read())

    status = get("/api/live/status")
    if status["state"] != "ready":
        raise SystemExit(f"Test Lab not ready: {status['state']} ({status['message']})")
    scoring, explain, wall, matched = [], [], [], 0
    for i in range(args.n):
        s = get(f"/api/live/sample?filter=any&seed={i}")
        t0 = time.perf_counter()
        r = post("/api/score-live", {"transaction_id": s["transaction_id"]})
        wall.append((time.perf_counter() - t0) * 1000)
        scoring.append(r["latency_ms"])
        explain.append(r.get("explain_ms", 0.0))
        matched += bool(r.get("matches_frozen_run"))
    out = {
        "measured": datetime.now().isoformat(timespec="seconds"), "n": args.n,
        "scoring_ms": {"p50": percentile(scoring, 50), "p95": percentile(scoring, 95),
                       "max": max(scoring), "mean": statistics.mean(scoring)},
        "explanation_ms": {"p50": percentile(explain, 50), "p95": percentile(explain, 95)},
        "request_ms": {"p50": percentile(wall, 50), "p95": percentile(wall, 95)},
        "matched_frozen_run": matched,
    }  # fmt: skip
    print(json.dumps(out, indent=1))
    root = Path(os.environ.get("VAULTIC_CACHE_DIR") or "E:/dev-cache") / "testlab"
    root.mkdir(parents=True, exist_ok=True)
    (root / f"latency_{datetime.now():%Y%m%d-%H%M%S}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
