"""POST-HOC / EXPLORATORY E5 graph-drift diagnostic, validation only (D120; for Paper 1).

Run:  python -m vaultic.reports.e5_graph_drift

E5 (B5 + the setting-C graph features, EXP-105-tuned) scored below B5 on validation (D99/D100),
also on rows without graph evidence. In setting C the per-entity counts (g_deg_tx_*,
g_deg_uid_*, g_shared_uids_*, g_shared_nonhub, g_twohop_fraud) count every earlier transaction
since the start of the data, so they grow with calendar time; components and communities use a
30-day window. This diagnostic, on the training period (days 1-120) and validation (128-150)
only, never the test period:

  1. drift   per graph feature: train vs validation PSI (10 train-quantile bins) and KS on the
             rows where it is available, share of validation rows above the training maximum,
             and the Spearman correlation with the day inside the training period
  2. variants  B5+graph refitted with EXP-105-tuned's frozen hyperparameters on days 1-120 and
             scored on validation, with the cumulative counts (a) as built (reproduces EXP-105
             seed 0), (b) per elapsed day (count / days since the data start), (c) dropped, and
             (d) log1p (a monotone transform, which tree models should ignore: a check). Seeds
             0-2 (exploratory screening, 3 of the 5 seeds to keep the job short); compared with
             the stored B5 (EXP-009) and E5 (EXP-105-tuned) validation predictions of the same
             seeds, overall and with / without graph evidence. Caps at the training maximum are
             not refitted: for trees a value above the training maximum already falls in the
             last split, so a cap changes no prediction.

No frozen view, config or feature file changes. Writes research/e5_graph_drift.md and
research/tables/e5_graph_drift.json.
"""

from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd
import yaml

from vaultic.paths import CONFIG_DIR, RESEARCH_DIR, RUNS_DIR

OUT_MD = RESEARCH_DIR / "e5_graph_drift.md"
OUT_JSON = RESEARCH_DIR / "tables" / "e5_graph_drift.json"
E5_RUN = RUNS_DIR / "EXP-105-tuned" / "20261010-164859-235447"
B5_RUN = RUNS_DIR / "EXP-009" / "20261008-211052-888539"
SEEDS = (0, 1, 2)
CUMULATIVE_PREFIXES = ("g_deg_tx_", "g_deg_uid_", "g_shared_uids_")
CUMULATIVE_EXTRA = ("g_shared_nonhub", "g_twohop_fraud")


def cumulative_columns(columns) -> list[str]:
    return [c for c in columns if c.startswith(CUMULATIVE_PREFIXES) or c in CUMULATIVE_EXTRA]


def psi(ref, new, bins: int = 10) -> float:
    from vaultic.fusion.f3_diagnosis import psi as _psi

    return _psi(np.asarray(ref, float), np.asarray(new, float), bins)


def drift_table(X: pd.DataFrame, day: np.ndarray, train: np.ndarray, val: np.ndarray) -> list[dict]:
    from scipy.stats import ks_2samp, spearmanr

    rows = []
    for c in [c for c in X.columns if c.startswith("g_")]:
        a, b = X.loc[train, c].to_numpy(float), X.loc[val, c].to_numpy(float)
        fa, fb = np.isfinite(a), np.isfinite(b)
        tmax = float(np.nanmax(a)) if fa.any() else float("nan")
        rho = spearmanr(day[train][fa], a[fa]).statistic if fa.sum() > 2 else float("nan")
        rows.append({"feature": c, "cumulative": c in cumulative_columns([c]),
                     "available_train": float(fa.mean()), "available_val": float(fb.mean()),
                     "median_train": float(np.nanmedian(a)) if fa.any() else float("nan"),
                     "median_val": float(np.nanmedian(b)) if fb.any() else float("nan"),
                     "max_train": tmax, "max_val": float(np.nanmax(b)) if fb.any() else float("nan"),
                     "psi": psi(a[fa], b[fb]), "ks": float(ks_2samp(a[fa], b[fb]).statistic)
                     if fa.any() and fb.any() else float("nan"),
                     "share_val_above_train_max": float(np.mean(b[fb] > tmax)) if fb.any() else 0.0,
                     "spearman_day_train": float(rho)})  # fmt: skip
    return rows


def variant(X: pd.DataFrame, day: np.ndarray, kind: str) -> pd.DataFrame:
    cols = cumulative_columns(X.columns)
    if kind == "as built":
        return X
    if kind == "dropped":
        return X.drop(columns=cols)
    out = X.copy()
    if kind == "per elapsed day":
        out[cols] = X[cols].to_numpy(float) / np.maximum(day, 1)[:, None]
    elif kind == "log1p":
        out[cols] = np.log1p(X[cols].to_numpy(float))
    else:
        raise ValueError(kind)
    return out


def main() -> None:
    from vaultic.data.splits import load_splits
    from vaultic.eval.metrics import pr_auc
    from vaultic.eval.run import load_inputs
    from vaultic.features.sets import design_matrix
    from vaultic.views.tabular import make_model

    cfg = yaml.safe_load((CONFIG_DIR / "EXP-105-tuned.yaml").read_text("utf-8"))
    splits = load_splits()
    df, base, _ = load_inputs(cfg, splits)
    X = design_matrix(df, base, cfg["features"])
    day = df["day"].to_numpy()
    train, val = splits.train.contains(day), splits.validation.contains(day)
    if splits.test.contains(day[val]).any():
        raise RuntimeError("a test-period row reached the validation set")
    y = df["isFraud"].to_numpy()
    ids = df["TransactionID"].to_numpy()
    drift = drift_table(X, day, train, val)
    print("drift table done", flush=True)

    graph_ok = X.loc[val, "g_shared_nonhub"].to_numpy(float) > 0  # graph-view availability, D52
    groups = {"all validation rows": np.ones(val.sum(), bool), "graph evidence": graph_ok,
              "no graph evidence": ~graph_ok}  # fmt: skip

    def scores_from_run(run, seeds):
        p = pd.read_parquet(run / "predictions.parquet")
        p = p[p["split"] == "validation"].set_index("TransactionID")
        return np.mean([p[f"score_seed{s}"].reindex(ids[val]).to_numpy() for s in seeds], axis=0)

    yv = y[val]
    results = {"B5 (stored EXP-009)": scores_from_run(B5_RUN, SEEDS),
               "E5 (stored EXP-105-tuned)": scores_from_run(E5_RUN, SEEDS)}  # fmt: skip
    params = cfg["model"]["params"]
    repro = None
    plan = [("as built", (0,)), ("log1p", (0,)), ("per elapsed day", SEEDS), ("dropped", SEEDS)]
    for kind, seeds in plan:
        Xk = variant(X, day, kind)
        preds = []
        for s in seeds:
            m = make_model("xgboost", params, s, "cpu").fit(Xk[train], y[train])
            preds.append(m.predict_proba(Xk[val])[:, 1])
            print(f"{kind} seed {s} done", flush=True)
        if kind == "as built":
            ref = scores_from_run(E5_RUN, (0,))
            repro = {"max_abs_diff": float(np.max(np.abs(ref - preds[0]))),
                     "corr": float(np.corrcoef(ref, preds[0])[0, 1])}  # fmt: skip
        results[f"B5+graph, cumulative counts {kind} (seeds {','.join(map(str, seeds))})"] = (
            np.mean(preds, axis=0)
        )
    table = []
    for name, s in results.items():
        table.append({"model": name, **{g: float(pr_auc(yv[m], s[m])) for g, m in groups.items()}})
    result = {"generated": date.today().isoformat(), "post_hoc": True, "validation_only": True,
              "seeds": list(SEEDS), "reproduction_seed0": repro,
              "group_sizes": {g: [int(m.sum()), int(yv[m].sum())] for g, m in groups.items()},
              "drift": drift, "models": table}  # fmt: skip
    OUT_JSON.write_text(json.dumps(result, indent=1), encoding="utf-8")
    from vaultic.reports.render_saved import keep_hand_section

    keep_hand_section(OUT_MD, lambda: write_md(result))  # never overwrite the hand-written reading
    print(f"wrote {OUT_MD} and {OUT_JSON}")


def _f(x, nd=3):
    return "" if x is None or not np.isfinite(x) else f"{x:,.{nd}f}"


def write_md(r: dict) -> None:
    rp = r["reproduction_seed0"]
    lines = ["# E5 graph-feature drift (POST-HOC / EXPLORATORY, validation only)", "",
             "**POST-HOC / EXPLORATORY, for Paper 1.** Not pre-registered; uses the training period "
             "(days 1-120) and validation (128-150) only, never the test period. No frozen view, "
             f"config or feature file changes. Generated {r['generated']} by "
             "`python -m vaultic.reports.e5_graph_drift` (D120).", "",
             "## 1. Drift of each graph feature (train -> validation)", "",
             "Cumulative = counts every earlier transaction since the data start (setting C entity "
             "counts); the others are rates or 30-day-window structure. PSI with 10 training-quantile "
             "bins and KS on rows where the feature is available. Spearman: correlation with the day "
             "inside the training period (growth with calendar time).", "",
             "| feature | cumulative | available train -> val | median train -> val | max train -> val | PSI | KS | val above train max | Spearman(day) |",
             "|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for d in sorted(r["drift"], key=lambda d: -d["psi"] if np.isfinite(d["psi"]) else 0):
        lines.append(f"| {d['feature']} | {'yes' if d['cumulative'] else 'no'} | "
                     f"{d['available_train']:.2f} -> {d['available_val']:.2f} | "
                     f"{_f(d['median_train'], 2)} -> {_f(d['median_val'], 2)} | "
                     f"{_f(d['max_train'], 0)} -> {_f(d['max_val'], 0)} | {_f(d['psi'])} | "
                     f"{_f(d['ks'])} | {d['share_val_above_train_max']:.3f} | "
                     f"{_f(d['spearman_day_train'])} |")  # fmt: skip
    g = r["group_sizes"]
    lines += ["", "## 2. Does time-normalising the cumulative counts remove E5's penalty?", "",
              f"Validation PR-AUC. Seeds {r['seeds']} (stored runs: the same seeds' scores). "
              f"Reproduction of EXP-105-tuned seed 0 by the 'as built' refit: max |diff| "
              f"{rp['max_abs_diff']:.2e}, correlation {rp['corr']:.6f}. Rows: "
              + "; ".join(f"{k} {v[0]:,} ({v[1]:,} frauds)" for k, v in g.items()) + ".", "",
              "| model | " + " | ".join(g) + " |", "|---|" + "---|" * len(g)]  # fmt: skip
    for m in r["models"]:
        lines.append(f"| {m['model']} | " + " | ".join(f"{m[k]:.4f}" for k in g) + " |")
    lines += ["", "## 3. Reading (added by hand, not generated)", "", "_To be written._", ""]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
