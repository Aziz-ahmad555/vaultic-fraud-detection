"""The temporal view as a separate step: it needs PyTorch, which lives only in .venv-torch
(D40, D41), while everything else runs in the main .venv. The two exchange files:

  1. main venv:   write_inputs(dir, df, uid, plan, encoder)    -> inputs.parquet, plan.json,
                                                                   encoder.json, settings.json
  2. .venv-torch: python -m vaultic.views.temporal_step DIR OUT -> OUT (parquet:
                                                                   TransactionID, fold, p_temporal)
  3. main venv:   view_table(..., external={"temporal": pd.read_parquet(OUT)})

Only rows up to the last predicted day of the plan are written, so a development plan never
puts test-period rows into the inputs. Per fold the GRU is trained on the fold's training
rows; early stopping uses the latest `early_stop_fraction` of those rows (by time), never the
predicted block, so the predictions stay out-of-sample. Uids without history get NaN.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from vaultic.features.categories import CategoryEncoder
from vaultic.views.plan import Fold, plan_from_json, plan_to_json

BASE_COLUMNS = ["TransactionID", "TransactionDT", "day", "TransactionAmt", "ProductCD", "isFraud"]


def write_inputs(
    out_dir: Path,
    df: pd.DataFrame,
    uid: pd.Series,
    plan: list[Fold],
    encoder: CategoryEncoder,
    extra_columns: tuple[str, ...] = (),
    n_steps: int = 10,
    early_stop_fraction: float = 0.2,
    gru: dict | None = None,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    last_day = max(f.predict[1] for f in plan)
    keep = df["day"].to_numpy() <= last_day
    frame = df.loc[keep, [*BASE_COLUMNS, *extra_columns]].copy()
    frame["ProductCD"] = frame["ProductCD"].astype(object)
    frame["uid"] = uid.to_numpy()[keep].astype(str)
    frame.reset_index(drop=True).to_parquet(out_dir / "inputs.parquet", index=False)
    (out_dir / "plan.json").write_text(plan_to_json(plan), encoding="utf-8")
    encoder.save(out_dir / "encoder.json")
    settings = {"extra_columns": list(extra_columns), "n_steps": n_steps,
                "early_stop_fraction": early_stop_fraction, "gru": gru or {}}  # fmt: skip
    (out_dir / "settings.json").write_text(json.dumps(settings, indent=1), encoding="utf-8")
    return out_dir


def run(in_dir: Path, out_path: Path) -> pd.DataFrame:
    from vaultic.features.sequences import build_sequences
    from vaultic.views.temporal import GRUTemporalView, TemporalData

    in_dir = Path(in_dir)
    df = pd.read_parquet(in_dir / "inputs.parquet")
    plan = plan_from_json((in_dir / "plan.json").read_text(encoding="utf-8"))
    encoder = CategoryEncoder.load(in_dir / "encoder.json")
    settings = json.loads((in_dir / "settings.json").read_text(encoding="utf-8"))

    seq = build_sequences(df, df["uid"], encoder, n_steps=settings["n_steps"],
                          extra_columns=tuple(settings["extra_columns"]))  # fmt: skip
    data = TemporalData.from_sequences(seq, df["isFraud"].to_numpy())
    day, time = df["day"].to_numpy(), df["TransactionDT"].to_numpy()

    parts, histories = [], {}
    for fold in plan:
        train = np.flatnonzero(fold.train_rows(day, time))
        cut = np.quantile(time[train], 1 - settings["early_stop_fraction"])
        fit_rows, stop_rows = train[time[train] < cut], train[time[train] >= cut]
        view = GRUTemporalView(encoder.n_codes("ProductCD"), **settings["gru"])
        view.fit(data.subset(fit_rows), data.subset(stop_rows))
        histories[fold.name] = {"best_epoch": view.best_epoch_, "best_stop_pr_auc": view.best_val_pr_auc_,
                                "fit_rows": int(len(fit_rows)), "stop_rows": int(len(stop_rows)),
                                "pos_weight": view.pos_weight_, "epochs": view.history_}  # fmt: skip
        pred = np.flatnonzero(fold.predict_rows(day))
        parts.append(pd.DataFrame({"TransactionID": df["TransactionID"].to_numpy()[pred],
                                   "fold": fold.name,
                                   "p_temporal": view.predict_proba(data.subset(pred))}))  # fmt: skip
    out = pd.concat(parts, ignore_index=True)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    # training curve per fold (early-stopping PR-AUC per epoch), for diagnosis
    Path(out_path).with_suffix(".history.json").write_text(json.dumps(histories, indent=1),
                                                           encoding="utf-8")  # fmt: skip
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Temporal view predictions (run in .venv-torch)")
    parser.add_argument("in_dir", type=Path, help="folder written by write_inputs")
    parser.add_argument("out", type=Path, help="parquet file for TransactionID, fold, p_temporal")
    args = parser.parse_args()
    out = run(args.in_dir, args.out)
    print(f"{len(out)} rows, {int(out['p_temporal'].notna().sum())} scored -> {args.out}")


if __name__ == "__main__":
    main()
