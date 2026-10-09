"""Phase 5 development run of the GRU temporal view (validation only), in three steps:

  1. main venv:   python -m vaultic.views.temporal_dev prepare [--smoke]
  2. .venv-torch: python -m vaultic.views.temporal_step <dir> <dir>/p_temporal.parquet
  3. main venv:   python -m vaultic.views.temporal_dev report <b5_run_dir> [--smoke]

Step 1 writes the inputs for the FIXED development plan (train days 1-120 -> predict validation
128-150; early stopping on the latest 20% of the training rows by time, D45), so no test-period
row is written. --smoke keeps a fixed 1-in-10 subset of uids (whole histories) under
<dir>-smoke for a pipeline check.

Step 3 compares the GRU with tuned B5 on the validation rows WITH history (the temporal view is
masked for cold-start rows, rule 11), by history length (1-4, 5-19, 20+ earlier transactions):
GRU alone, and an unfitted fusion, the mean of the two scores' ranks, so nothing is fitted on
validation. Paired bootstrap (1,000 resamples) of fusion - B5 and GRU - B5 per group. Writes
research/tables/temporal_dev.md. D70.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from vaultic.eval.bootstrap import paired_bootstrap
from vaultic.eval.metrics import pr_auc
from vaultic.paths import INTERIM_DIR, RESEARCH_DIR

EXTRA = ("C1", "C13", "D1", "dist1")
N_STEPS = 20
GROUPS = [
    ("1-4 past", 1, 4),
    ("5-19 past", 5, 19),
    ("20+ past", 20, None),
    ("all with history", 1, None),
]
N_BOOT = 1000


def work_dir(smoke: bool, name: str = "temporal_dev") -> Path:
    return INTERIM_DIR / (f"{name}-smoke" if smoke else name)


def prepare(
    smoke: bool, extra: tuple[str, ...] = EXTRA, name: str = "temporal_dev", gru: dict | None = None
) -> Path:
    from vaultic.data.load import load_merged
    from vaultic.data.splits import load_splits
    from vaultic.data.uid import UID_PATH
    from vaultic.features.categories import fit_on_training_period
    from vaultic.paths import MERGED_PATH
    from vaultic.views.plan import fixed_plan
    from vaultic.views.temporal_step import BASE_COLUMNS, write_inputs

    splits = load_splits()
    df = load_merged(MERGED_PATH, columns=[*BASE_COLUMNS, *extra])
    uid = pd.read_parquet(UID_PATH, columns=["TransactionID", splits.uid_variant])
    if not (uid["TransactionID"].to_numpy() == df["TransactionID"].to_numpy()).all():
        raise ValueError("uids.parquet is out of date")
    uid = uid[splits.uid_variant]
    if smoke:
        keep = (pd.util.hash_pandas_object(uid.astype(str), index=False) % 10 == 0).to_numpy()
        df, uid = df[keep].reset_index(drop=True), uid[keep].reset_index(drop=True)
    encoder = fit_on_training_period(df, splits, columns=("ProductCD",))
    return write_inputs(work_dir(smoke, name), df, uid, fixed_plan(splits), encoder,
                        extra_columns=tuple(extra), n_steps=N_STEPS, gru=gru)  # fmt: skip


def _rank(s: np.ndarray) -> np.ndarray:
    return pd.Series(s).rank(method="average").to_numpy() / len(s)


def compare(y, b5_seeds: list[np.ndarray], gru: np.ndarray, n_past: np.ndarray) -> pd.DataFrame:
    """Per history group: B5, GRU and rank-mean fusion PR-AUC, with paired gains over B5."""
    rows = []
    has = ~np.isnan(gru)
    for name, lo, hi in GROUPS:
        m = has & (n_past >= lo) & ((n_past <= hi) if hi is not None else True)
        yy = y[m]
        row = {"group": name, "rows": int(m.sum()), "frauds": int(yy.sum())}
        if row["frauds"] == 0 or row["frauds"] == row["rows"]:
            rows.append(row)
            continue
        b5 = [s[m] for s in b5_seeds]
        g = gru[m]
        fused = [(_rank(s) + _rank(g)) / 2 for s in b5]
        row["B5"] = float(np.mean([pr_auc(yy, s) for s in b5]))
        row["GRU"] = pr_auc(yy, g)
        row["fusion"] = float(np.mean([pr_auc(yy, s) for s in fused]))
        for label, scores in (("GRU-B5", [g] * len(b5)), ("fusion-B5", fused)):
            r = paired_bootstrap(yy, scores, b5, pr_auc, n_boot=N_BOOT, seed=0)
            row[label] = (r["diff"], r["ci_low"], r["ci_high"], r["p_value"])
        rows.append(row)
    return pd.DataFrame(rows)


def report(b5_run: Path, smoke: bool, name: str = "temporal_dev") -> Path:
    from vaultic.data.splits import load_splits
    from vaultic.features.pipeline import features_path

    temporal = pd.read_parquet(work_dir(smoke, name) / "p_temporal.parquet")
    preds = pd.read_parquet(b5_run / "predictions.parquet")
    preds = preds[preds["split"] == "validation"]
    merged = preds.merge(temporal[["TransactionID", "p_temporal"]], on="TransactionID", how="inner")
    feats = pd.read_parquet(features_path(load_splits().uid_variant),
                            columns=["TransactionID", "uid_n_past"])  # fmt: skip
    merged = merged.merge(feats, on="TransactionID", how="left")
    seeds = [merged[c].to_numpy() for c in merged.columns if c.startswith("score_seed")]
    table = compare(merged["label"].to_numpy(), seeds, merged["p_temporal"].to_numpy(dtype=float),
                    merged["uid_n_past"].to_numpy())  # fmt: skip

    def gain(v):
        return (
            ""
            if not isinstance(v, tuple)
            else (f"{v[0]:+.4f} [{v[1]:+.4f}, {v[2]:+.4f}], p = {v[3]:.3f}")
        )

    lines = [
        "# Temporal view (GRU), development run"
        + (" — SMOKE (1 in 10 uids, not a result)" if smoke else ""),
        "",
        f"Generated {date.today().isoformat()} by `python -m vaultic.views.temporal_dev report`. "
        "**Validation period only.** GRU over the uid's last 20 transactions (amount, gap, "
        "ProductCD, C1, C13, D1, dist1), trained on days 1-120 with early stopping on the latest "
        "20% of the training rows (D45), one seed. Compared with tuned B5 (5 seeds) on the "
        "validation rows where the temporal view exists (uid has history); fusion = mean of the "
        "two scores' ranks (nothing fitted on validation). Paired bootstrap, 1,000 resamples.",
        "",
        f"- B5 run: `{b5_run.name}`; temporal rows scored: {int(temporal['p_temporal'].notna().sum())} "
        f"of {len(temporal)} validation rows (the rest have no history: masked).",
        "",
        "| history | rows | frauds | B5 | GRU | fusion | GRU − B5 | fusion − B5 |",
        "|---|---|---|---|---|---|---|---|",
    ]

    def num(r, k):
        return "" if k not in r or pd.isna(r.get(k)) else f"{r[k]:.4f}"

    for r in table.to_dict("records"):
        f = lambda k, r=r: num(r, k)  # noqa: E731
        lines.append(f"| {r['group']} | {r['rows']} | {r['frauds']} | {f('B5')} | {f('GRU')} | "
                     f"{f('fusion')} | {gain(r.get('GRU-B5'))} | {gain(r.get('fusion-B5'))} |")  # fmt: skip
    out = RESEARCH_DIR / "tables" / (f"{name}-smoke.md" if smoke else f"{name}.md")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out.with_suffix(".json").write_text(
        json.dumps(table.to_dict("records"), indent=1, default=str), encoding="utf-8"
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=["prepare", "report"])
    parser.add_argument("b5_run", nargs="?", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--name", default="temporal_dev", help="input folder / report name")
    parser.add_argument("--extra", nargs="*", default=list(EXTRA), help="extra step columns")
    parser.add_argument("--gru", default="{}", help="GRU settings as JSON (GRUTemporalView)")
    args = parser.parse_args()
    if args.step == "prepare":
        out = prepare(args.smoke, tuple(args.extra), args.name, json.loads(args.gru))
        print(f"wrote inputs to {out}")
    else:
        if args.b5_run is None:
            parser.error("report needs the B5 run directory")
        print(f"wrote {report(args.b5_run, args.smoke, args.name)}")


if __name__ == "__main__":
    main()
