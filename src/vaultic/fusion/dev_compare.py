"""First development comparison of MVAF with F1-F7 (item 7, D71). Validation only, no test rows.

Run:  python -m vaultic.fusion.dev_compare --b5-run <B5 run dir> --f0-run <F0 run dir> [--smoke]

1. View-prediction table on the FIXED plan: ONE set of view models trained on days 1-120,
   predicting validation 128-150 (D77); roles gate_train 128-143, calibrate_views 144-146 and
   calibrate_fused 147-150 (D53, D76):
     tabular     XGBoost on the TABULAR view: B5 features without the label-derived ones (D89;
                 frozen B5 hyperparameters, seed 0)
     behavioral  XGBoost on the BEHAVIORAL view: behavioral features + the label-derived customer
                 features (D89), only for uids with history
     graph       XGBoost on the graph features (setting C), only with non-hub evidence (D52)
     anomaly     logistic link on the forward-chained anomaly scores (D68), only where scored;
                 the scores of training rows are out-of-sample, so the link is fitted where
                 they mean what they mean at prediction time (as in D57)
     temporal    XGBoost on the sequence inputs (views/temporal_diag.py, D88: the GRU was not
                 kept), joined as external; NaN for uids without history
2. calibrate_views: one calibrator per view on the calibrate_views rows (144-146) only (D56, D76;
   Platt below 100 frauds, D83).
3. fusion_split + fit_all: MVAF and F1-F7 with default settings fitted on the SAME gate rows
   (the tune rows are not used: no hyperparameter search in this first look).
4. Each method scored on the calibrate_fused rows (147-150, which no per-view calibrator saw):
   PR-AUC with a 1,000-resample bootstrap CI, and a paired bootstrap of MVAF minus each
   baseline. B5 and F0 (D90) are scored on the same rows from their development runs'
   validation predictions (mean over seeds); both were tuned on the whole validation period,
   which includes these days, so they are favoured here (stated in the report).

Writes research/tables/fusion_dev.md (+ .json) and the view table to
data/interim/view_table_fixed.parquet. --smoke: a fixed 1 in 10 subset of uids.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from vaultic.eval.bootstrap import paired_bootstrap
from vaultic.eval.metrics import pr_auc
from vaultic.paths import INTERIM_DIR, RESEARCH_DIR

METHODS = ("MVAF", "F1", "F2", "F3", "F4", "F5", "F6", "F7")
N_BOOT = 1000


def subgroup_masks(rows) -> dict[str, np.ndarray]:
    """Evaluation subgroups from the gate context and view masks: history, identity, graph."""
    from vaultic.views.orchestrate import CONTEXT, VIEWS

    hist = rows.context[:, CONTEXT.index("ctx_hist_n_past")]
    ident = rows.context[:, CONTEXT.index("ctx_has_identity")]
    graph = ~np.isnan(rows.views[:, VIEWS.index("graph")])
    return {
        "cold start": hist == 0, "with history": hist > 0,
        "has_identity yes": ident == 1, "has_identity no": ident == 0,
        "graph view available": graph, "graph view missing": ~graph,
    }  # fmt: skip


def subgroup_table(scores: dict[str, np.ndarray], rows, versus=("B5", "F0")) -> pd.DataFrame:
    """PR-AUC of every method per subgroup, and paired MVAF - B5 / MVAF - F0 with 95% CIs."""
    out = []
    for name, m in subgroup_masks(rows).items():
        y = rows.y[m]
        row = {"subgroup": name, "rows": int(m.sum()), "frauds": int(y.sum())}
        if 0 < row["frauds"] < row["rows"]:
            row.update({k: pr_auc(y, s[m]) for k, s in scores.items()})
            for other in versus:
                if other in scores and "MVAF" in scores:
                    r = paired_bootstrap(y, [scores["MVAF"][m]], [scores[other][m]], pr_auc,
                                         n_boot=N_BOOT, seed=0)  # fmt: skip
                    row[f"MVAF - {other}"] = (r["diff"], r["ci_low"], r["ci_high"], r["p_value"])
        out.append(row)
    return pd.DataFrame(out)


def evaluate(fitted: dict, rows, external: dict | None = None) -> pd.DataFrame:
    """PR-AUC of each fused score on `rows`, bootstrap CI, and paired MVAF - method.
    `external`: name -> scores already aligned with `rows` (e.g. B5, F0 run predictions)."""
    y = rows.y
    scores = {name: m.predict_proba(rows.views, rows.context) for name, m in fitted.items()}
    scores.update(external or {})
    rng = np.random.default_rng(0)
    idx = [rng.integers(0, len(y), len(y)) for _ in range(N_BOOT)]
    out = []
    for name, s in scores.items():
        boots = [pr_auc(y[i], s[i]) for i in idx if 0 < y[i].sum() < len(i)]
        row = {"method": name, "PR-AUC": pr_auc(y, s),
               "ci_low": float(np.quantile(boots, 0.025)), "ci_high": float(np.quantile(boots, 0.975))}  # fmt: skip
        if name != "MVAF" and "MVAF" in scores:
            r = paired_bootstrap(y, [scores["MVAF"]], [s], pr_auc, n_boot=N_BOOT, seed=0)
            row.update({"MVAF - method": r["diff"], "d_low": r["ci_low"], "d_high": r["ci_high"],
                        "p": r["p_value"]})  # fmt: skip
        out.append(row)
    return pd.DataFrame(out)


def external_scores(run_dir: Path, ids: np.ndarray) -> np.ndarray:
    """A development run's validation score (mean over seeds) for the given TransactionIDs."""
    preds = pd.read_parquet(run_dir / "predictions.parquet")
    preds = preds[preds["split"] == "validation"].set_index("TransactionID")["score"]
    out = preds.reindex(ids).to_numpy(dtype=float)
    if np.isnan(out).any():
        raise ValueError(f"{run_dir} lacks validation predictions for some evaluation rows")
    return out


def build_table(smoke: bool, d95: bool = False):
    """The view table. d95=True (D95): the cross-fitted plan, the views with their tuned
    hyperparameters (configs EXP-V-<view>.yaml from `vaultic.eval.tune --view`), XGBoost for the
    anomaly and temporal views (temporal on the sequence features, in-process)."""
    import yaml

    from vaultic.data.load import load_merged
    from vaultic.data.splits import load_splits
    from vaultic.data.uid import UID_PATH
    from vaultic.features.categories import fit_on_training_period
    from vaultic.features.pipeline import features_path
    from vaultic.features.sets import design_matrix
    from vaultic.paths import CONFIG_DIR, FEATURES_DIR, MERGED_PATH
    from vaultic.paths import INTERIM_DIR as _INTERIM
    from vaultic.views.definitions import available
    from vaultic.views.orchestrate import (
        SupervisedView,
        has_graph_evidence,
        has_history,
        view_table,
    )
    from vaultic.views.plan import crossfit_plan, fixed_plan

    splits = load_splits()
    variant = splits.uid_variant
    df = load_merged(MERGED_PATH)
    base = pd.read_parquet(features_path(variant))
    extra = {k: pd.read_parquet(FEATURES_DIR / f"{k}_{variant}.parquet")
             for k in ("behavioral", "graph", "anomaly", *(("sequence",) if d95 else ()))}  # fmt: skip
    uid = pd.read_parquet(UID_PATH, columns=["TransactionID", variant])[variant]
    for name, frame in [("base", base), *extra.items()]:
        if not np.array_equal(frame["TransactionID"].to_numpy(), df["TransactionID"].to_numpy()):
            raise ValueError(f"{name} features are not aligned with the data")
    if smoke:
        keep = (pd.util.hash_pandas_object(uid.astype(str), index=False) % 10 == 0).to_numpy()
        df, base, uid = (x[keep].reset_index(drop=True) for x in (df, base, uid))
        extra = {k: v[keep].reset_index(drop=True) for k, v in extra.items()}

    from vaultic.views.definitions import LABEL_DERIVED, behavioral_columns, tabular_columns

    tab = design_matrix(df, base, "b5")
    tab_cols = tabular_columns(tab.columns)  # D89: label-free
    cols = {k: [c for c in v.columns if c != "TransactionID"] for k, v in extra.items()}
    features = pd.concat([tab, *(extra[k][cols[k]] for k in extra)], axis=1)
    assert all(c in features for c in LABEL_DERIVED)  # the behavioral view needs them (D89)
    features["uid"] = uid.to_numpy()
    params = yaml.safe_load((CONFIG_DIR / "EXP-009.yaml").read_text("utf-8"))["model"]["params"]
    if d95:

        def tuned(view):
            cfg = yaml.safe_load((CONFIG_DIR / f"EXP-V-{view}.yaml").read_text("utf-8"))
            return cfg["model"]["params"]

        def avail(view):
            return lambda f: available(view, f)

        views = {
            "tabular": SupervisedView(tab_cols, "xgboost", tuned("tabular")),
            "behavioral": SupervisedView(behavioral_columns(cols["behavioral"]), "xgboost",
                                         tuned("behavioral"), available=avail("behavioral")),
            "graph": SupervisedView(cols["graph"], "xgboost", tuned("graph"),
                                    available=avail("graph")),
            "anomaly": SupervisedView(cols["anomaly"], "xgboost", tuned("anomaly"),
                                      available=avail("anomaly")),
            "temporal": SupervisedView(cols["sequence"], "xgboost", tuned("temporal"),
                                       available=avail("temporal")),
        }  # fmt: skip
        encoder = fit_on_training_period(df, splits, columns=("ProductCD",))
        return view_table(df, features, views, crossfit_plan(splits), encoder, splits)
    views = {
        "tabular": SupervisedView(tab_cols, "xgboost", params),
        "behavioral": SupervisedView(behavioral_columns(cols["behavioral"]), "xgboost", params,
                                     available=has_history),  # fmt: skip
        "graph": SupervisedView(cols["graph"], "xgboost", params, available=has_graph_evidence),
        "anomaly": SupervisedView(cols["anomaly"], "logistic_regression", {"max_iter": 2000},
                                  available=lambda f: f["anomaly_if"].notna().to_numpy()),  # fmt: skip
    }
    # D88: the temporal view is XGBoost on the sequence inputs of the fixed plan
    temporal = pd.read_parquet(_INTERIM / "temporal_dev_v4" / "p_temporal_xgb.parquet")
    encoder = fit_on_training_period(df, splits, columns=("ProductCD",))
    table = view_table(df, features, views, fixed_plan(splits), encoder, splits,
                       external={"temporal": temporal})  # fmt: skip
    return table


def main() -> None:
    from vaultic.fusion.baselines import make_fusion
    from vaultic.fusion.data import fit_all, fusion_split
    from vaultic.views.orchestrate import VIEWS, calibrate_views

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--b5-run", type=Path, required=True, help="B5 run (validation predictions)"
    )
    parser.add_argument("--f0-run", type=Path, default=None,
                        help="F0 run (validation predictions); added once F0 is tuned (D90)")  # fmt: skip
    parser.add_argument("--reuse-table", action="store_true",
                        help="reuse the saved calibrated view table instead of rebuilding it")  # fmt: skip
    args = parser.parse_args()
    started = time.perf_counter()
    suffix = "-smoke" if args.smoke else ""
    table_path = INTERIM_DIR / f"view_table_fixed{suffix}.parquet"
    out = RESEARCH_DIR / "tables" / f"fusion_dev{suffix}.md"
    if args.reuse_table:
        table = pd.read_parquet(table_path)
        table.attrs["calibrated"] = True  # saved after calibrate_views
        cal_info = json.loads(out.with_suffix(".json").read_text("utf-8"))["calibrators"]
    else:
        table = build_table(args.smoke)
        if (table["role"] == "test").any() or (table.get("split", pd.Series()) == "test").any():
            raise RuntimeError("a test-period row reached the development view table")
        table, info = calibrate_views(table)
        cal_info = {f"{f}/{v}": e for (f, v), e in info.items()}
        table.to_parquet(table_path, index=False)
    split = fusion_split(table)
    fitted = fit_all(split, {m: (lambda m=m: make_fusion(m, seed=0)) for m in METHODS})
    runs = [("B5", args.b5_run)] + ([("F0", args.f0_run)] if args.f0_run else [])
    external = {name: external_scores(run, split.calibrate.ids) for name, run in runs}
    result = evaluate(fitted, split.calibrate, external)
    rows = split.calibrate
    scores = {n: m.predict_proba(rows.views, rows.context) for n, m in fitted.items()}
    sub = subgroup_table({**scores, **external}, rows)

    cal = table[table["role"] == "calibrate_fused"]
    view_rows = []
    for v in VIEWS:
        m = cal[f"m_{v}"].to_numpy() == 1
        y = cal["label"].to_numpy()[m]
        auc = (
            pr_auc(y, cal[f"p_{v}"].to_numpy()[m]) if m.any() and 0 < y.sum() < m.sum() else np.nan
        )
        view_rows.append({"view": v, "available": float(m.mean()), "rows": int(m.sum()),
                          "PR-AUC (available rows)": auc})  # fmt: skip
    lines = [
        "# MVAF vs F1-F7, B5"
        + (" and F0" if args.f0_run else "")
        + ": development comparison"
        + (" — SMOKE (not a result)" if args.smoke else ""),
        "",
        f"Generated {date.today().isoformat()} by `python -m vaultic.fusion.dev_compare`. "
        "**Validation period only; no test rows.** Fixed plan: views trained on days 1-120; fusion "
        f"fitted on gate rows (days 128-143 minus the inner tune days; {len(split.fit)} rows), "
        f"evaluated on the calibrate_fused slice (days 147-150; {len(split.calibrate)} rows, "
        f"{int(split.calibrate.y.sum())} frauds), which the per-view calibrators (days 144-146) "
        "never saw. Default settings, no hyperparameter search, one seed; temporal view = "
        "XGBoost on the sequence inputs (D88). Bootstrap 1,000 resamples.",
        "",
        "## Views on the evaluation rows (calibrate_fused, days 147-150)",
        "",
        "| view | available | rows | PR-AUC (available rows) |",
        "|---|---|---|---|",
        *[f"| {r['view']} | {r['available']:.1%} | {r['rows']} | {r['PR-AUC (available rows)']:.4f} |"
          for r in view_rows],  # fmt: skip
        "",
        "## Fused scores on the evaluation rows (calibrate_fused, days 147-150)",
        "",
        "| method | PR-AUC | 95% CI | MVAF − method | 95% CI | p |",
        "|---|---|---|---|---|---|",
    ]
    for r in result.to_dict("records"):
        d = (
            ("", "", "")
            if pd.isna(r.get("MVAF - method", np.nan))
            else (
                f"{r['MVAF - method']:+.4f}",
                f"[{r['d_low']:+.4f}, {r['d_high']:+.4f}]",
                f"{r['p']:.3f}",
            )
        )
        lines.append(f"| {r['method']} | {r['PR-AUC']:.4f} | [{r['ci_low']:.4f}, {r['ci_high']:.4f}] | "
                     f"{d[0]} | {d[1]} | {d[2]} |")  # fmt: skip
    methods = [c for c in ("MVAF", "B5", "F0", *METHODS[1:]) if c in sub.columns]
    versus = [c for c in sub.columns if c.startswith("MVAF - ")]
    lines += [
        "",
        "## Subgroups (evaluation rows)",
        "",
        "PR-AUC per method; paired MVAF − B5 (and − F0) with 95% CI. History = the uid has an earlier "
        "transaction; graph view available = non-hub relational evidence (D52).",
        "",
        "| subgroup | rows | frauds | " + " | ".join(methods) + " | " + " | ".join(versus) + " |",
        "|---|---|---|" + "---|" * (len(methods) + len(versus)),
    ]
    for r in sub.to_dict("records"):
        vals = ["" if pd.isna(r.get(m, np.nan)) else f"{r[m]:.4f}" for m in methods]
        diffs = ["" if not isinstance(r.get(v), tuple) else
                 f"{r[v][0]:+.4f} [{r[v][1]:+.4f}, {r[v][2]:+.4f}]" for v in versus]  # fmt: skip
        lines.append(f"| {r['subgroup']} | {r['rows']} | {r['frauds']} | " + " | ".join(vals)
                     + " | " + " | ".join(diffs) + " |")  # fmt: skip
    lines += ["", f"Runtime {time.perf_counter() - started:.0f} s.", ""]
    out.write_text("\n".join(lines), encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps({"fusion": result.to_dict("records"),
        "views": view_rows, "calibrators": cal_info,
        "subgroups": sub.to_dict("records")}, indent=1, default=str), encoding="utf-8")  # fmt: skip
    print(f"wrote {out}")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
