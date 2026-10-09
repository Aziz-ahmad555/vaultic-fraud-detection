"""Phase 6: precompute the anomaly view's scores for every transaction, point-in-time.

Run:  python -m vaultic.views.anomaly_build [--max-train-rows N] [--timing]

Writes data/features/anomaly_<uid variant>.parquet with anomaly_if, anomaly_ae and
anomaly_uid_if (views/anomaly.py), used by the b5_anomaly feature set (D68).

Forward chaining, so every score is out-of-sample and uses only earlier data:
  - the training period is cut into BLOCK_DAYS-day blocks; rows of block k are scored by models
    fitted on the legit rows of blocks < k; the first block has no earlier data and stays NaN
    (masked, rule 11);
  - rows after the training period (gap, validation, test) are scored by models fitted on all
    training-period legit rows.
Labels are used only to keep legit rows for fitting, under the same convention as every
baseline: the labels of the data a model is fitted on count as known when it is built (the
harness default; E25 / EXP-125 studies label maturity separately).
Scoring training rows with models fitted on those same rows would make them look less anomalous
than validation rows (in-sample), the same problem as review finding M5.

Inputs: the B5 design matrix without its label-derived features (anomaly scores stay
label-free; the labels are used only to keep legit rows for fitting, as in views/anomaly.py).
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from vaultic.views.anomaly import AnomalyView

BLOCK_DAYS = 30
LABEL_DERIVED = ("uid_fraud_known", "uid_fraud_rate_known", "uid_n_labels_known")
SCORES = ("anomaly_if", "anomaly_ae", "anomaly_uid_if")


def forward_chained_scores(
    X: np.ndarray,
    y: np.ndarray,
    uid: np.ndarray,
    n_past: np.ndarray,
    day: np.ndarray,
    train_last_day: int,
    first_day: int = 1,
    block_days: int = BLOCK_DAYS,
    max_fit_rows: int | None = None,
    seed: int = 0,
    **view_kwargs,
) -> pd.DataFrame:
    """Scores for every row; each from models fitted only on legit rows of earlier days."""
    out = pd.DataFrame(np.nan, index=np.arange(len(X)), columns=list(SCORES))
    starts = list(range(first_day, train_last_day + 1, block_days))
    # (fit rows: day < start, score rows: start <= day < next start); the last cut scores
    # everything after the training period with a model fitted on the whole training period
    cuts = [(s, min(s + block_days, train_last_day + 1)) for s in starts[1:]]
    cuts.append((train_last_day + 1, int(day.max()) + 1))
    rng = np.random.default_rng(seed)
    for start, stop in cuts:
        fit = np.flatnonzero((day < start) & (day >= first_day))
        score = (day >= start) & (day < stop)
        if not score.any() or len(fit) == 0:
            continue
        if max_fit_rows is not None and len(fit) > max_fit_rows:
            fit = np.sort(rng.choice(fit, max_fit_rows, replace=False))
        view = AnomalyView(seed=seed, **view_kwargs).fit(X[fit], y[fit], uid[fit])
        scored = view.transform(X[score], uid[score], n_past[score])
        out.loc[score, list(SCORES)] = scored[list(SCORES)].to_numpy()
    return out


def main() -> None:
    from vaultic.data.load import load_merged
    from vaultic.data.splits import load_splits
    from vaultic.data.uid import UID_PATH
    from vaultic.features.pipeline import features_path
    from vaultic.features.sets import design_matrix
    from vaultic.paths import FEATURES_DIR, MERGED_PATH

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-fit-rows", type=int, default=150_000,
                        help="legit rows sampled per fit (autoencoder cost)")  # fmt: skip
    parser.add_argument("--ae-max-iter", type=int, default=50)
    parser.add_argument("--timing", action="store_true", help="every 10th row only; no output")
    args = parser.parse_args()
    splits = load_splits()
    variant = splits.uid_variant
    df = load_merged(MERGED_PATH)
    base = pd.read_parquet(features_path(variant))
    uid = pd.read_parquet(UID_PATH, columns=["TransactionID", variant])[variant].to_numpy()
    if args.timing:
        keep = df["TransactionID"].to_numpy() % 10 == 0
        df, base, uid = (
            df[keep].reset_index(drop=True),
            base[keep].reset_index(drop=True),
            uid[keep],
        )
    X = design_matrix(df, base, "b5").drop(columns=list(LABEL_DERIVED))
    started = time.perf_counter()
    scores = forward_chained_scores(
        X.to_numpy(dtype=float),
        df["isFraud"].to_numpy(),
        uid,
        base["uid_n_past"].to_numpy(),
        df["day"].to_numpy(),
        splits.train.last,
        first_day=splits.train.first,
        max_fit_rows=args.max_fit_rows,
        max_iter=args.ae_max_iter,
    )
    info = {
        "seconds": round(time.perf_counter() - started, 1),
        "rows": len(scores),
        "inputs": X.shape[1],
        "max_fit_rows": args.max_fit_rows,
        "ae_max_iter": args.ae_max_iter,
        "block_days": BLOCK_DAYS,
        "coverage": {c: float(scores[c].notna().mean()) for c in SCORES},
    }
    print(json.dumps(info, indent=1))
    if not args.timing:
        scores.insert(0, "TransactionID", df["TransactionID"].to_numpy())
        out = FEATURES_DIR / f"anomaly_{variant}.parquet"
        scores.to_parquet(out, index=False)
        out.with_suffix(".json").write_text(json.dumps(info, indent=1), encoding="utf-8")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
