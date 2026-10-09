"""SHAP summary of a tree baseline (default: tuned B5), on a VALIDATION sample.

Run:  python -m vaultic.reports.shap_summary experiments/configs/EXP-009.yaml --name b5

Trains the config's model with its first seed on the train period, explains a fixed random
sample of validation rows with TreeExplainer, and writes research/figures/<name>_shap_summary.png
and research/tables/<name>_shap_top20.md/.csv (mean |SHAP|). Features keep their raw names;
masked IEEE-CIS features (C, D, M, V, id_) have no published meaning and are not given one.
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from vaultic.data.splits import SPLITS_PATH, load_splits
from vaultic.eval.run import load_config, load_inputs
from vaultic.features.sets import design_matrix, drop_features
from vaultic.paths import RESEARCH_DIR
from vaultic.views.tabular import make_model

FIGURES_DIR = RESEARCH_DIR / "figures"
TABLES_DIR = RESEARCH_DIR / "tables"


def shap_values(model, X: pd.DataFrame) -> np.ndarray:
    import shap

    values = shap.TreeExplainer(model).shap_values(X)
    return values[1] if isinstance(values, list) else values


def importance(values: np.ndarray, columns, top: int = 20) -> pd.DataFrame:
    mean_abs = np.abs(values).mean(axis=0)
    order = np.argsort(-mean_abs, kind="stable")[:top]
    return pd.DataFrame({"feature": np.asarray(columns)[order], "mean_abs_shap": mean_abs[order]})


def save_plot(values: np.ndarray, X: pd.DataFrame, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    shap.summary_plot(values, X, max_display=20, show=False)
    plt.title(title, fontsize=10)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close("all")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", type=Path)
    parser.add_argument("--name", default="b5")
    parser.add_argument("--sample", type=int, default=5000)
    args = parser.parse_args()

    cfg = load_config(args.config)
    splits = load_splits(Path(cfg.get("splits", SPLITS_PATH)))
    df, base, _ = load_inputs(cfg, splits)
    X = drop_features(design_matrix(df, base, cfg["features"]), cfg.get("drop_features"))
    y = df["isFraud"].to_numpy()
    part = splits.assign(df["day"])
    tr, va = part == "train", part == "validation"

    seed = cfg["seeds"][0]
    model = make_model(cfg["model"]["name"], cfg["model"].get("params", {}), seed)
    model.fit(X[tr], y[tr])
    rng = np.random.default_rng(0)
    rows = np.sort(rng.choice(np.flatnonzero(va), size=min(args.sample, va.sum()), replace=False))
    X_s = X.iloc[rows]
    values = shap_values(model, X_s)

    title = f"{cfg['id']} ({args.name.upper()}): SHAP on {len(rows):,} validation rows, seed {seed}"
    save_plot(values, X_s, FIGURES_DIR / f"{args.name}_shap_summary.png", title)
    top = importance(values, X.columns)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    top.to_csv(TABLES_DIR / f"{args.name}_shap_top20.csv", index=False)
    lines = [
        f"# {args.name.upper()} SHAP top 20 (mean |SHAP|)",
        "",
        f"Generated {date.today().isoformat()} by `python -m vaultic.reports.shap_summary "
        f"{args.config.as_posix()}`: {cfg['id']}, seed {seed}, {len(rows):,} random validation "
        "rows. Masked IEEE-CIS features (C, D, M, V, id_) have no published meaning.",
        "",
        "| rank | feature | mean abs SHAP |",
        "|---|---|---|",
        *[f"| {i + 1} | `{f}` | {v:.4f} |" for i, (f, v) in enumerate(top.itertuples(index=False))],
        "",
    ]
    (TABLES_DIR / f"{args.name}_shap_top20.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote research/figures/{args.name}_shap_summary.png and the top-20 table")


if __name__ == "__main__":
    main()
