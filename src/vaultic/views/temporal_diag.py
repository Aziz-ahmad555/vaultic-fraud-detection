"""GRU diagnosis, step 1 (validation only, D74): XGBoost on the GRU's exact inputs.

Run:  python -m vaultic.views.temporal_diag <inputs dir>     (main venv; no PyTorch needed)

Reads the folder written by `temporal_dev prepare` (inputs, plan, encoder, settings), builds the
same sequences the GRU sees (features/sequences.py), flattens them (last N steps x step
features, padded steps as NaN so padding is "missing", not a zero-valued transaction) and adds
the number of real steps and the scored transaction's own step features. XGBoost is trained on
the same fit rows as the GRU (training rows with history, minus the latest 20% by time) with
early stopping on the same latest-20% rows (D45), and scored on the validation rows with
history. Its PR-AUC is reported next to the GRU's on exactly the same rows; if a GRU training
curve exists (<dir>/p_temporal.history.json), its best epoch is shown too.

If XGBoost is also weak, the inputs carry little signal (enrich them); if it is much stronger,
the GRU's training is at fault. Writes <dir>/diag.json and <dir>/p_temporal_xgb.parquet (the
XGBoost temporal view's predictions for every predicted row, NaN without history; under D86/D88
this is the temporal view MVAF uses) and prints a one-line summary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from vaultic.eval.metrics import pr_auc

XGB_PARAMS = {"n_estimators": 2000, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.8,
              "colsample_bytree": 0.8, "min_child_weight": 3, "tree_method": "hist"}  # fmt: skip
EARLY_STOP = 100


def flatten(seq) -> np.ndarray:
    """(n, N*F + 1 + F): history steps (NaN where padded), number of real steps, current step."""
    values = seq.values.astype(np.float32).copy()
    values[~seq.mask] = np.nan
    n = len(values)
    return np.column_stack(
        [values.reshape(n, -1), seq.mask.sum(axis=1).astype(np.float32), seq.current]
    )


def build_sequence_features() -> Path:
    """data/features/sequence_<uid>.parquet: the flattened sequence inputs (as `flatten`) for
    EVERY row, so the temporal view and F0 can use them through the harness (D88, D90). Label-
    free and point-in-time (each row sees only earlier rows of its uid); the ProductCD encoder
    is fitted on the training period; same steps and channels as the GRU development runs."""
    from vaultic.data.load import load_merged
    from vaultic.data.splits import load_splits
    from vaultic.data.uid import UID_PATH
    from vaultic.features.categories import fit_on_training_period
    from vaultic.features.sequences import build_sequences
    from vaultic.paths import FEATURES_DIR, MERGED_PATH
    from vaultic.views.temporal_dev import EXTRA, N_STEPS
    from vaultic.views.temporal_step import BASE_COLUMNS

    splits = load_splits()
    df = load_merged(MERGED_PATH, columns=[c for c in BASE_COLUMNS if c != "isFraud"] + list(EXTRA))
    uid = pd.read_parquet(UID_PATH, columns=["TransactionID", splits.uid_variant])
    if not (uid["TransactionID"].to_numpy() == df["TransactionID"].to_numpy()).all():
        raise ValueError("uids.parquet is out of date")
    encoder = fit_on_training_period(df, splits, columns=("ProductCD",))
    seq = build_sequences(df, uid[splits.uid_variant].astype(str), encoder, n_steps=N_STEPS,
                          extra_columns=EXTRA)  # fmt: skip
    X = flatten(seq)
    names = [f"seq{t:02d}_{f}" for t in range(N_STEPS) for f in seq.feature_names]
    names += ["seq_n_steps", *[f"seq_cur_{f}" for f in seq.feature_names]]
    out = pd.DataFrame(X, columns=names)
    out.loc[~seq.has_history, names] = np.nan  # no history: the view is missing (rule 11)
    out.insert(0, "TransactionID", df["TransactionID"].to_numpy())
    path = FEATURES_DIR / f"sequence_{splits.uid_variant}.parquet"
    out.to_parquet(path, index=False)
    return path


def run(in_dir: Path) -> dict:
    from xgboost import XGBClassifier

    from vaultic.features.categories import CategoryEncoder
    from vaultic.features.sequences import build_sequences
    from vaultic.views.plan import plan_from_json

    in_dir = Path(in_dir)
    df = pd.read_parquet(in_dir / "inputs.parquet")
    (fold,) = plan_from_json((in_dir / "plan.json").read_text(encoding="utf-8"))
    encoder = CategoryEncoder.load(in_dir / "encoder.json")
    settings = json.loads((in_dir / "settings.json").read_text(encoding="utf-8"))
    seq = build_sequences(df, df["uid"], encoder, n_steps=settings["n_steps"],
                          extra_columns=tuple(settings["extra_columns"]))  # fmt: skip
    X, y = flatten(seq), df["isFraud"].to_numpy()
    day, time = df["day"].to_numpy(), df["TransactionDT"].to_numpy()
    hist = seq.has_history

    train = np.flatnonzero(fold.train_rows(day, time) & hist)
    cut = np.quantile(time[train], 1 - settings["early_stop_fraction"])
    fit_rows, stop_rows = train[time[train] < cut], train[time[train] >= cut]
    pred = np.flatnonzero(fold.predict_rows(day) & hist)

    neg, pos = (y[fit_rows] == 0).sum(), (y[fit_rows] == 1).sum()
    model = XGBClassifier(**XGB_PARAMS, eval_metric="aucpr", early_stopping_rounds=EARLY_STOP,
                          scale_pos_weight=float(np.sqrt(neg / pos)), random_state=0)  # fmt: skip
    model.fit(X[fit_rows], y[fit_rows], eval_set=[(X[stop_rows], y[stop_rows])], verbose=False)
    p_xgb = model.predict_proba(X[pred])[:, 1]

    out = {
        "inputs_dir": str(in_dir),
        "n_steps": settings["n_steps"],
        "step_features": seq.feature_names,
        "n_inputs": int(X.shape[1]),
        "fit_rows": int(len(fit_rows)),
        "stop_rows": int(len(stop_rows)),
        "validation_rows_with_history": int(len(pred)),
        "validation_frauds": int(y[pred].sum()),
        "xgb_best_iteration": int(model.best_iteration),
        "xgb_stop_pr_auc": float(model.best_score),
        "xgb_val_pr_auc": pr_auc(y[pred], p_xgb),
    }
    gru_path = in_dir / "p_temporal.parquet"
    if gru_path.exists():
        gru = pd.read_parquet(gru_path).set_index("TransactionID")["p_temporal"]
        g = gru.reindex(df["TransactionID"].to_numpy()[pred]).to_numpy(dtype=float)
        if np.isnan(g).any():
            raise ValueError("GRU predictions are missing for some validation rows with history")
        out["gru_val_pr_auc"] = pr_auc(y[pred], g)
    hist_path = in_dir / "p_temporal.history.json"
    if hist_path.exists():
        h = next(iter(json.loads(hist_path.read_text(encoding="utf-8")).values()))
        out["gru_best_epoch"] = h["best_epoch"]
        out["gru_stop_pr_auc"] = h["best_stop_pr_auc"]
        out["gru_curve"] = [round(e["val_pr_auc"], 4) for e in h["epochs"]]
    (in_dir / "diag.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    # the XGBoost temporal view's predictions in the external-view format of
    # views/orchestrate.view_table (D88): every predicted row, NaN where the uid has no history
    rows = np.flatnonzero(fold.predict_rows(day))
    p = np.full(len(rows), np.nan)
    with_hist = hist[rows]
    p[with_hist] = model.predict_proba(X[rows[with_hist]])[:, 1]
    pd.DataFrame({"TransactionID": df["TransactionID"].to_numpy()[rows], "fold": fold.name,
                  "p_temporal": p}).to_parquet(in_dir / "p_temporal_xgb.parquet", index=False)  # fmt: skip
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("in_dir", type=Path, nargs="?")
    parser.add_argument("--build-features", action="store_true",
                        help="write data/features/sequence_<uid>.parquet for every row (D90)")  # fmt: skip
    args = parser.parse_args()
    if args.build_features:
        print(f"wrote {build_sequence_features()}")
        return
    out = run(args.in_dir)
    print(json.dumps({k: v for k, v in out.items() if k != "step_features"}, indent=1))


if __name__ == "__main__":
    main()
