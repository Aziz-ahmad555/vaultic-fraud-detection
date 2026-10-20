"""Inputs of the feedback simulation (EXP-310; D210 section 6, D211).

The B5 inputs come from the frozen point-in-time feature pipeline (stage2.system.load_parts). The
three label-derived B5 columns (uid_fraud_known, uid_fraud_rate_known, uid_n_labels_known, D89)
are stored with the system's 30-day label delay; for a simulation at delay L != 30 they are
recomputed with the same code (features.pipeline.build_features) at delay L, so the model's inputs
never use a label the loop is not allowed to know (for L = 60 the stored values would). Every other
input is label-free. Recomputed columns are cached under the dev cache on drive E.

The design run (EXP-310-dev) reads days 1-120 only: rows after day 120 are dropped right after
loading and `require_last_day` refuses any that get through.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from vaultic.views.definitions import LABEL_DERIVED

STORED_DELAY = 30
DEV_LAST_DAY = 120
EXP_FINAL = "EXP-310-final"


def require_last_day(day, last: int) -> None:
    if len(day) and int(np.max(day)) > last:
        raise RuntimeError(f"a row after day {last} reached the design run")


def label_derived_at(df: pd.DataFrame, uid: pd.Series, delay: int, train_mask) -> pd.DataFrame:
    """The three label-derived columns recomputed at label delay `delay` (days), with the
    pipeline's own point-in-time code; df sorted by TransactionDT."""
    from vaultic.features.pipeline import INPUT_COLUMNS, build_features, fit_encoders

    train_mask = np.asarray(train_mask, bool)
    enc = fit_encoders(df.loc[train_mask], uid[train_mask])
    out = build_features(df[INPUT_COLUMNS].reset_index(drop=True), uid.reset_index(drop=True),
                         enc, delay)  # fmt: skip
    return out[list(LABEL_DERIVED)].reset_index(drop=True)


def require_final_started(run_dir, token=None, experiment_log=None) -> None:
    """D215: proof that the final-run guards passed in THIS process (the token start() / resume()
    issued for this run folder), that the latest FINAL-STARTED / FINAL-RESUMED row naming it is in
    the experiment log AND on origin, and that the run has status 'started'."""
    import json
    from pathlib import Path

    from vaultic.learning import feedback_final as ff

    if run_dir is None:
        raise RuntimeError("EXP-310-final needs Aziz's go-ahead and the final-run guards")
    run_dir = Path(run_dir)
    if not ff.guards_passed(run_dir, token):
        raise RuntimeError(f"the final-run guards did not pass in this process for {run_dir.name}: "
                           "test data is read only through start() / resume()")  # fmt: skip
    ff.latest_row_on_origin(run_dir, experiment_log)
    meta = run_dir / "metrics.json"
    status = json.loads(meta.read_text("utf-8")).get("status") if meta.exists() else None
    if status != "started":
        raise RuntimeError(f"{run_dir.name} has status {status!r}, not 'started'")


def load(mode: str, run_dir=None, token=None) -> dict:
    """Data and B5 inputs for 'dev' (days 1-120) or 'final' (every day). 'final' needs the run
    folder and the token from start() / resume() in this process (require_final_started)."""
    from vaultic.stage2.system import load_parts

    if mode not in ("dev", "final"):
        raise ValueError("mode must be dev or final")
    if mode == "final":
        require_final_started(run_dir, token)
        parts = load_parts(dev=False)  # every day, test period included (after FINAL-STARTED)
        return {"df": parts["df"], "features": parts["features"], "b5_cols": list(parts["b5_cols"]),
                "splits": parts["splits"], "mode": mode}  # fmt: skip
    parts = load_parts(dev=True)  # test-period rows dropped at load (D180)
    keep = parts["df"]["day"].to_numpy() <= DEV_LAST_DAY
    df = parts["df"][keep].reset_index(drop=True)
    features = parts["features"][keep].reset_index(drop=True)
    require_last_day(df["day"].to_numpy(), DEV_LAST_DAY)
    return {"df": df, "features": features, "b5_cols": list(parts["b5_cols"]),
            "splits": parts["splits"], "mode": mode}  # fmt: skip


def input_key(loaded: dict, delay: int) -> str:
    """SHA-256 of everything the recomputed columns depend on: row ids, times, labels, the uid,
    the delay and the code that computes them (D214)."""
    import hashlib
    from pathlib import Path

    import vaultic.features.pipeline as pipeline

    h = hashlib.sha256()
    df = loaded["df"]
    for col in ("TransactionID", "TransactionDT", "isFraud"):
        h.update(np.ascontiguousarray(df[col].to_numpy(np.int64)).tobytes())
    h.update(
        pd.util.hash_pandas_object(loaded["features"]["uid"].astype(str), index=False)
        .to_numpy()
        .tobytes()
    )
    h.update(str(delay).encode())
    for src in (Path(__file__), Path(pipeline.__file__)):
        h.update(src.read_bytes())
    return h.hexdigest()


def _file_sha256(path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def inputs_for_delay(loaded: dict, delay: int, cache_dir=None) -> pd.DataFrame:
    """B5 inputs with the label-derived columns at delay L (stored values when L = 30). A cached
    recomputation is used only if its recorded input key and file hash both match (D214)."""
    import json

    X = loaded["features"][loaded["b5_cols"]].copy()
    if delay == STORED_DELAY:
        return X
    path = side = key = None
    if cache_dir is not None:
        key = input_key(loaded, delay)
        path = cache_dir / f"label_derived_{loaded['mode']}_L{delay}.parquet"
        side = path.with_suffix(".json")
        if path.exists() and side.exists():
            meta = json.loads(side.read_text("utf-8"))
            if meta.get("input_key") == key and meta.get("file_sha256") == _file_sha256(path):
                cols = pd.read_parquet(path)
                for c in LABEL_DERIVED:
                    X[c] = cols[c].to_numpy(dtype=np.float32)
                return X
    df = loaded["df"]
    train = loaded["splits"].train.contains(df["day"].to_numpy())
    cols = label_derived_at(df, loaded["features"]["uid"], delay, train)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        cols.to_parquet(path, index=False)
        side.write_text(json.dumps({"input_key": key, "file_sha256": _file_sha256(path),
                                    "delay": delay, "rows": len(cols)}), "utf-8")  # fmt: skip
    for c in LABEL_DERIVED:
        X[c] = cols[c].to_numpy(dtype=np.float32)
    return X
