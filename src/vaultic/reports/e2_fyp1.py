"""E2: FYP-1's original evaluation next to its honest temporal result (B6).

Run:  python -m vaultic.reports.e2_fyp1

FYP-1 never stored its metrics (ml/train_model.py only printed them), so the original numbers
are reproduced exactly from FYP-1's own artifacts: the saved model
(legacy/fyp1/models_store/xgboost_baseline.json) scored on the saved random 80/20 test split
(legacy/fyp1/data/X_test.csv, y_test.csv), with FYP-1's metrics at its 0.5 threshold plus
PR-AUC and recall at 1% FPR. The temporal column is the latest B6 harness run (EXP-010):
the --final (test period) run if one exists, otherwise the development (validation) run,
labelled as such. Writes research/tables/e2_fyp1_comparison.md and .csv.
"""

from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from vaultic.eval.metrics import pr_auc, recall_at_fpr, roc_auc
from vaultic.paths import REPO_ROOT, RESEARCH_DIR
from vaultic.reports.table1 import latest_run

FYP1_DIR = REPO_ROOT / "legacy" / "fyp1"
B6_EXPERIMENT = "EXP-010"
TABLES_DIR = RESEARCH_DIR / "tables"


def fyp1_metrics(y: np.ndarray, proba: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    pred = (proba >= threshold).astype(int)
    return {
        "PR-AUC": pr_auc(y, proba),
        "ROC-AUC": roc_auc(y, proba),
        "Recall@1%FPR": recall_at_fpr(y, proba, 0.01),
        "Precision@0.5": float(precision_score(y, pred, zero_division=0)),
        "Recall@0.5": float(recall_score(y, pred)),
        "F1@0.5": float(f1_score(y, pred)),
        "Accuracy@0.5": float(accuracy_score(y, pred)),
    }


def original_evaluation() -> tuple[dict[str, float], int]:
    from xgboost import XGBClassifier

    model = XGBClassifier()
    model.load_model(FYP1_DIR / "models_store" / "xgboost_baseline.json")
    X_test = pd.read_csv(FYP1_DIR / "data" / "X_test.csv")
    y_test = pd.read_csv(FYP1_DIR / "data" / "y_test.csv").to_numpy().ravel()
    return fyp1_metrics(y_test, model.predict_proba(X_test)[:, 1]), len(y_test)


def temporal_evaluation() -> tuple[dict[str, float], str, str]:
    run = latest_run(B6_EXPERIMENT, "final")
    period = "test period, days 151-182 (--final)"
    if run is None:
        run = latest_run(B6_EXPERIMENT, "development")
        period = "validation period, days 128-150 (development run; final pending)"
    if run is None:
        raise FileNotFoundError("no B6 (EXP-010) run yet")
    result = json.loads((run / "metrics.json").read_text("utf-8"))
    m = result["test" if result["mode"] == "final" else "validation"]
    out = {
        "PR-AUC": m["pr_auc"]["mean"],
        "ROC-AUC": m["roc_auc"]["mean"],
        "Recall@1%FPR": m["recall_at_1pct_fpr"]["mean"],
    }
    return out, period, run.name


def main() -> None:
    original, n_test = original_evaluation()
    temporal, period, run_name = temporal_evaluation()
    rows = []
    for metric, value in original.items():
        rows.append(
            {"metric": metric, "FYP-1 original": value, "B6 temporal": temporal.get(metric)}
        )
    table = pd.DataFrame(rows)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLES_DIR / "e2_fyp1_comparison.csv", index=False)

    def cell(v):
        return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.4f}"

    lines = [
        "# E2: FYP-1's original evaluation vs. the temporal split",
        "",
        f"Generated {date.today().isoformat()} by `python -m vaultic.reports.e2_fyp1`.",
        "",
        f"- **FYP-1 original:** FYP-1's saved XGBoost on its saved random 80/20 test split "
        f"({n_test:,} rows), FYP-1's 0.5 threshold. FYP-1 never stored these numbers; they are "
        "reproduced from its own artifacts.",
        f"- **B6 temporal:** the same method ported to the time-based split (global XGBoost + "
        f"per-user Isolation Forest + 5x-max rule, fitted on days 1-120), {period}, "
        f"run `{run_name}`.",
        "",
        "| metric | FYP-1 original (random split) | B6 temporal |",
        "|---|---|---|",
        *[
            f"| {r['metric']} | {cell(r['FYP-1 original'])} | {cell(r['B6 temporal'])} |"
            for r in rows
        ],
        "",
        "Accuracy is shown only because FYP-1 reported it; with 3.5% fraud it says little. "
        "The harness does not compute thresholded metrics at 0.5, so those cells are empty "
        "for B6.",
        "",
        "Why the original numbers are optimistic: (1) the 80/20 split was random, so the model "
        "trained on transactions from after the ones it was tested on; (2) missing values were "
        "filled with medians computed on the whole dataset, test rows included, before splitting.",
        "",
    ]
    (TABLES_DIR / "e2_fyp1_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote research/tables/e2_fyp1_comparison.md and .csv")


if __name__ == "__main__":
    main()
