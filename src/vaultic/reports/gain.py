"""Paired gain of one development run over another, overall and by customer-history group.

Run:  python -m vaultic.reports.gain <base_run_dir> <new_run_dir> --name b5_behavioral \
          [--groups history|gru] [--title "..."]

Both runs must have been scored on the same validation rows. The gain is new minus base PR-AUC
with a paired bootstrap (1,000 resamples, identical rows for both runs; per-seed scores are
averaged as in vaultic.eval.compare). Groups come from the uid's number of earlier transactions
(`uid_n_past`, point-in-time, from the base features of the configured uid variant):

  history  all; >= 5 past transactions; 1-4; cold start (0)
  gru      all; 1-4; 5-19; 20+; cold start (0)

Only the validation split is read. Writes research/tables/gain_<name>.md.
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
from vaultic.paths import RESEARCH_DIR

GROUPS = {
    "history": [("all", 0, None), (">= 5 past", 5, None), ("1-4 past", 1, 4), ("cold start", 0, 0)],
    "gru": [("all", 0, None), ("1-4 past", 1, 4), ("5-19 past", 5, 19), ("20+ past", 20, None),
            ("cold start", 0, 0)],
}  # fmt: skip
N_BOOT = 1000


def _validation(run_dir: Path) -> pd.DataFrame:
    preds = pd.read_parquet(run_dir / "predictions.parquet")
    return preds[preds["split"] == "validation"].reset_index(drop=True)


def _seeds(preds: pd.DataFrame) -> list[np.ndarray]:
    return [preds[c].to_numpy() for c in preds.columns if c.startswith("score_seed")]


def group_masks(n_past: np.ndarray, kind: str) -> dict[str, np.ndarray]:
    out = {}
    for name, lo, hi in GROUPS[kind]:
        mask = n_past >= lo
        if hi is not None:
            mask &= n_past <= hi
        out[name] = mask
    return out


def gain_table(
    base: pd.DataFrame, new: pd.DataFrame, n_past: np.ndarray, kind: str = "history"
) -> pd.DataFrame:
    """One row per group: rows, frauds, base and new PR-AUC, paired gain with 95% CI and p."""
    if not np.array_equal(base["TransactionID"].to_numpy(), new["TransactionID"].to_numpy()):
        raise ValueError("runs were scored on different validation rows")
    y = base["label"].to_numpy()
    sb, sn = _seeds(base), _seeds(new)
    rows = []
    for name, mask in group_masks(n_past, kind).items():
        yy = y[mask]
        row = {"group": name, "rows": int(mask.sum()), "frauds": int(yy.sum())}
        if row["frauds"] == 0 or row["frauds"] == row["rows"]:
            rows.append({**row, "note": "one class only: not scored"})
            continue
        res = paired_bootstrap(
            yy, [s[mask] for s in sn], [s[mask] for s in sb], pr_auc, n_boot=N_BOOT, seed=0
        )
        rows.append(
            {
                **row,
                "base PR-AUC": float(np.mean([pr_auc(yy, s[mask]) for s in sb])),
                "new PR-AUC": float(np.mean([pr_auc(yy, s[mask]) for s in sn])),
                "gain": res["diff"],
                "ci_low": res["ci_low"],
                "ci_high": res["ci_high"],
                "p": res["p_value"],
            }
        )
    return pd.DataFrame(rows)


def n_past_for(preds: pd.DataFrame, base_features: pd.DataFrame) -> np.ndarray:
    """uid_n_past of each prediction row, joined on TransactionID."""
    idx = pd.Index(base_features["TransactionID"]).get_indexer(preds["TransactionID"])
    if (idx < 0).any():
        raise ValueError("some prediction rows have no base features")
    return base_features["uid_n_past"].to_numpy()[idx]


def to_markdown(table: pd.DataFrame, title: str, base_run: Path, new_run: Path) -> str:
    def f(v, d=4):
        return "" if pd.isna(v) else f"{v:.{d}f}"

    lines = [
        f"# {title}",
        "",
        f"Generated {date.today().isoformat()} by `python -m vaultic.reports.gain` from development "
        "runs (**validation period only**). Gain = new − base PR-AUC, mean over seeds; paired "
        f"bootstrap ({N_BOOT} resamples, same rows for both runs). Groups by the uid's number of "
        "earlier transactions (point-in-time).",
        "",
        f"- base run: `{base_run.name}` ({base_run.parent.name})",
        f"- new run: `{new_run.name}` ({new_run.parent.name})",
        "",
        "| group | rows | frauds | base PR-AUC | new PR-AUC | gain | 95% CI | p |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in table.to_dict("records"):
        if "gain" not in r or pd.isna(r.get("gain")):
            lines.append(f"| {r['group']} | {r['rows']} | {r['frauds']} | | | | "
                         f"{r.get('note', '')} | |")  # fmt: skip
            continue
        lines.append(
            f"| {r['group']} | {r['rows']} | {r['frauds']} | {f(r['base PR-AUC'])} | "
            f"{f(r['new PR-AUC'])} | {r['gain']:+.4f} | [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] | "
            f"{f(r['p'], 3)} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    from vaultic.data.splits import load_splits
    from vaultic.features.pipeline import features_path

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("base_run", type=Path)
    parser.add_argument("new_run", type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--groups", default="history", choices=sorted(GROUPS))
    parser.add_argument("--title", default=None)
    args = parser.parse_args()
    base, new = _validation(args.base_run), _validation(args.new_run)
    feats = pd.read_parquet(
        features_path(load_splits().uid_variant), columns=["TransactionID", "uid_n_past"]
    )
    table = gain_table(base, new, n_past_for(base, feats), args.groups)
    title = args.title or f"Gain: {args.new_run.parent.name} over {args.base_run.parent.name}"
    out = RESEARCH_DIR / "tables" / f"gain_{args.name}.md"
    out.write_text(to_markdown(table, title, args.base_run, args.new_run), encoding="utf-8")
    (out.with_suffix(".json")).write_text(
        json.dumps(table.to_dict("records"), indent=1), encoding="utf-8"
    )
    print(f"wrote {out}")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
