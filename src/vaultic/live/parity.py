"""The Test Lab's parity gate (D183, tolerances amended by D189).

For a random sample of real test transactions (2,000, seed 0) the online features must equal the
offline frozen features within FEATURE_TOLERANCE (1e-6; the features are float32; every column of
the feature frame, NaN on the same columns), and the live MVAF and B5 scores must equal the stored
EXP-200-final scores within SCORE_TOLERANCE (1e-12; the observed difference is 3.3e-16, so 1e-6
would not mean anything for float64 scores). If not, the Test Lab refuses to score and reports
which columns failed.

The result is cached on drive E under a key made of the data-file hashes of the stage-2 freeze
record, the SHA-256 of the live and feature code, and the system cache, so it is recomputed only
when one of them changes.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from vaultic.live.store import CACHE_DIR

N = 2000
SEED = 0
FEATURE_TOLERANCE = 1e-6  # float32 features (D183)
SCORE_TOLERANCE = 1e-12  # float64 scores (D189)
CODE = ("live/features.py", "live/store.py", "live/engine.py", "features/pipeline.py",
        "features/behavioral.py", "features/graph.py", "features/sequences.py",
        "features/sets.py", "views/temporal_diag.py", "stage2/system.py")  # fmt: skip


def cache_key(system_path: Path | None) -> str:
    from vaultic.paths import REPO_ROOT, RESEARCH_DIR

    h = hashlib.sha256()
    src = REPO_ROOT / "src" / "vaultic"
    for rel in CODE:
        h.update((src / rel).read_bytes())
    h.update((RESEARCH_DIR / "frozen_stage2.md").read_bytes())  # data hashes of the freeze
    if system_path is not None and system_path.exists():
        st = system_path.stat()
        h.update(f"{system_path}:{st.st_size}:{st.st_mtime_ns}".encode())
    h.update(f"{N}:{SEED}:{FEATURE_TOLERANCE}:{SCORE_TOLERANCE}".encode())
    return h.hexdigest()


def _diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per element |a - b|; 0 where both NaN, inf where only one is NaN."""
    both = np.isnan(a) & np.isnan(b)
    d = np.abs(a - b)
    d[both] = 0.0
    d[np.isnan(d)] = np.inf
    return d


def run_parity(engine, n: int = N, seed: int = SEED, log=print, pool=None) -> dict:
    """pool: candidate row positions (default: every test-period row)."""
    from vaultic.live.features import online_features

    store, system = engine.store, engine.system
    day = store.df["day"].to_numpy()
    test = np.flatnonzero(day >= store.splits.test.first) if pool is None else np.asarray(pool)
    pick = np.sort(np.random.default_rng(seed).choice(test, min(n, len(test)), replace=False))
    worst: dict[str, float] = {}
    score_diff = {"MVAF": 0.0, "B5": 0.0}
    started = time.perf_counter()
    for i, p in enumerate(pick):
        q = store.df.iloc[[p]]
        f = online_features(store, system, q, int(store.uid[p]))
        cols = [c for c in f.columns if c != "uid"]
        a = f[cols].to_numpy(dtype=float)[0]
        b = store.features.iloc[p][cols].to_numpy(dtype=float)
        for c, v in zip(cols, _diff(a, b), strict=True):
            if v > worst.get(c, 0.0):
                worst[c] = float(v)
        ctx = system.context(q.reset_index(drop=True), f)
        risk = float(system.score(f, ctx)["risk"][0])
        b5 = float(system.b5(f)[0][0])
        tid = int(q["TransactionID"].iloc[0])
        if engine.stored is not None:
            score_diff["MVAF"] = max(score_diff["MVAF"],
                                     abs(risk - float(engine.stored.at[tid, "score_MVAF"])))  # fmt: skip
            score_diff["B5"] = max(score_diff["B5"],
                                   abs(b5 - float(engine.stored.at[tid, "score_B5"])))  # fmt: skip
        if log and (i + 1) % 250 == 0:
            log(f"parity {i + 1}/{len(pick)} ({time.perf_counter() - started:.0f} s)")
    failing = {c: v for c, v in worst.items() if v > FEATURE_TOLERANCE}
    passed = not failing and all(v <= SCORE_TOLERANCE for v in score_diff.values())
    return {"n": int(len(pick)), "seed": seed, "feature_tolerance": FEATURE_TOLERANCE,
            "score_tolerance": SCORE_TOLERANCE,
            "max_feature_diff": max(worst.values()) if worst else 0.0,
            "failing_features": dict(sorted(failing.items(), key=lambda x: -x[1])[:25]),
            "n_failing_features": len(failing), "max_score_diff": score_diff,
            "passed": bool(passed), "seconds": round(time.perf_counter() - started)}  # fmt: skip


def cached_parity(engine, system_path: Path | None, log=print) -> dict:
    """The parity result for the current data, code and system, computed once per key."""
    key = cache_key(system_path)
    path = CACHE_DIR / f"parity_{key[:16]}.json"
    if path.exists():
        out = json.loads(path.read_text("utf-8"))
        out["cached"] = True
        return out
    out = run_parity(engine, log=log)
    out["key"] = key
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1), "utf-8")
    out["cached"] = False
    return out
