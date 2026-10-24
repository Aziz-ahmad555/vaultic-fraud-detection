"""The Test Lab's parity gate (D183, tolerances amended by D189).

For a random sample of real test transactions (2,000, seed 0) the online features must equal the
offline frozen features within FEATURE_TOLERANCE (1e-6; the features are float32; every column of
the feature frame, NaN on the same columns), and the live MVAF and B5 scores must equal the stored
EXP-200-final scores within SCORE_TOLERANCE (1e-12; the observed difference is 3.3e-16, so 1e-6
would not mean anything for float64 scores). If not, the Test Lab refuses to score and reports
which columns failed.

The result is cached under a key made of the SHA-256 of the live and feature code, the stage-2
freeze record (data hashes), the stored final-run predictions, final_results.json and the system
cache, so it is recomputed only when one of them changes. Each file is hashed with its line
endings normalised (CRLF -> LF, D220), so the same content in checkouts with different line
endings gives the same key; the repository's own line endings are left alone (freeze hashes).
A missing input refuses the start with a clear message instead of entering the key (D220).
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


def key_files() -> list[Path]:
    """Files whose content keys the cached parity result besides the code: the stage-2 freeze
    record (data hashes), the stored final-run predictions the scores are compared with, and
    final_results.json (the thresholds the engine reads)."""
    from vaultic.paths import REPO_ROOT, RESEARCH_DIR
    from vaultic.stage2.final import E10_RUN

    return [RESEARCH_DIR / "frozen_stage2.md", REPO_ROOT / E10_RUN / "predictions.parquet",
            RESEARCH_DIR / "tables" / "final_results.json"]  # fmt: skip


TEXT_SUFFIXES = (".py", ".md", ".json", ".yaml", ".yml", ".txt")  # D221


def content_digest(path: Path) -> bytes:
    """SHA-256 of a file's content: text files with line endings normalised (CRLF -> LF, D220),
    every other file (parquet, joblib, ...) as its raw bytes (D221)."""
    data = Path(path).read_bytes()
    if Path(path).suffix.lower() in TEXT_SUFFIXES:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).digest()


def code_digest(root: Path, dirs) -> bytes:
    """SHA-256 over every .py file under root/<dir> for each dir (relative path + content),
    in sorted order (D221)."""
    h = hashlib.sha256()
    for d in dirs:
        for f in sorted((Path(root) / d).rglob("*.py")):
            h.update(f.relative_to(root).as_posix().encode())
            h.update(content_digest(f))
    return h.digest()


def missing_inputs(files: list[Path] | None = None) -> list[Path]:
    """Inputs the Test Lab needs that do not exist (D220)."""
    return [f for f in (key_files() if files is None else files) if not Path(f).exists()]


def require_inputs(files: list[Path] | None = None) -> None:
    """Refuse to start, naming every missing input, rather than run a parity that must fail."""
    missing = missing_inputs(files)
    if missing:
        raise FileNotFoundError(
            "the Test Lab cannot start, missing input(s): "
            + "; ".join(str(f) for f in missing)
            + ". In a worktree, link experiments/runs to the main checkout's run folders; "
            "the stored final-run predictions and the freeze record must be present."
        )


def cache_key(system_path: Path | None, files: list[Path] | None = None,
              store_key: str | None = None) -> str:  # fmt: skip
    """Key of a cached parity result; `store_key` (the history store's key: its code, the data
    files' size and modification time, D221) makes a data change force a full parity run."""
    from vaultic.paths import REPO_ROOT

    files = key_files() if files is None else files
    require_inputs(files)
    h = hashlib.sha256()
    src = REPO_ROOT / "src" / "vaultic"
    for rel in CODE:
        h.update(content_digest(src / rel))
    for f in files:
        h.update(f.name.encode())
        h.update(content_digest(f))
    if system_path is not None and system_path.exists():
        st = system_path.stat()
        h.update(f"{system_path}:{st.st_size}:{st.st_mtime_ns}".encode())
    if store_key is not None:
        h.update(f"store:{store_key}".encode())
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
    """pool: candidate row positions (default: every test-period row). Without the stored
    final-run scores the scores cannot be compared, so parity FAILS (it is never skipped)."""
    from vaultic.live.features import online_features

    if engine.stored is None:
        return {"n": 0, "seed": seed, "feature_tolerance": FEATURE_TOLERANCE,
                "score_tolerance": SCORE_TOLERANCE, "max_feature_diff": None,
                "failing_features": {}, "n_failing_features": 0,
                "max_score_diff": {"MVAF": None, "B5": None}, "passed": False,
                "reason": "the stored EXP-200-final scores are not available, so the live scores "
                          "cannot be compared", "seconds": 0}  # fmt: skip
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
        score_diff["MVAF"] = max(score_diff["MVAF"],
                                 abs(risk - float(engine.stored.at[tid, "score_MVAF"])))  # fmt: skip
        score_diff["B5"] = max(score_diff["B5"], abs(b5 - float(engine.stored.at[tid, "score_B5"])))
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
    store = getattr(engine, "store", None)
    key = cache_key(system_path, store_key=getattr(store, "info", {}).get("key"))
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
