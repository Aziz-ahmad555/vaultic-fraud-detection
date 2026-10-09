"""Phase 13 statistics: Holm-corrected comparison tables, effect sizes, the cost model and its
sensitivity to the assumed costs.

Cost (roadmap Phase 13):  missed fraud amount + C_FP * #FP + C_rev * #reviews, with the stated
assumptions C_FP = $10 and C_rev = $5 and a sensitivity sweep over $2-$50. In the sweep the
decision threshold is re-chosen on validation for every cost setting (never on test).
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import product

import numpy as np
import pandas as pd

from vaultic.eval.bootstrap import holm
from vaultic.eval.metrics import C_FP, C_REVIEW, choose_cost_threshold

SENSITIVITY_GRID = (2.0, 5.0, 10.0, 20.0, 50.0)


def holm_table(comparisons: Sequence[dict], alpha: float = 0.05) -> pd.DataFrame:
    """Add Holm-adjusted p-values to a table of paired comparisons that share one table.

    Each comparison is a dict with at least `name` and `p_value` (e.g. the output of
    vaultic.eval.compare.compare_runs plus a name); other keys are kept.
    """
    table = pd.DataFrame(list(comparisons))
    table["p_holm"] = holm(table["p_value"].tolist())
    table["significant"] = table["p_holm"] < alpha
    return table


def effect_sizes(per_seed_a: Sequence[float], per_seed_b: Sequence[float]) -> dict[str, float]:
    """Effect of A over B from per-seed metric values (same seeds, paired).

    diff: mean of A - B; relative_pct: diff as a % of B's mean; cohens_d_paired: mean of the
    per-seed differences divided by their standard deviation (NaN with < 2 seeds or no spread).
    """
    a, b = np.asarray(per_seed_a, dtype=float), np.asarray(per_seed_b, dtype=float)
    d = a - b
    sd = d.std(ddof=1) if len(d) > 1 else np.nan
    return {
        "diff": float(d.mean()),
        "relative_pct": float(d.mean() / b.mean() * 100) if b.mean() != 0 else np.nan,
        "cohens_d_paired": float(d.mean() / sd) if sd and np.isfinite(sd) else np.nan,
    }


def cost_breakdown(
    y, flagged, amount, c_fp: float = C_FP, c_rev: float = C_REVIEW
) -> dict[str, float]:
    """The three parts of the cost for a given set of flagged (reviewed) transactions."""
    y = np.asarray(y).astype(int)
    flagged = np.asarray(flagged, dtype=bool)
    amount = np.asarray(amount, dtype=float)
    missed = float(amount[(y == 1) & ~flagged].sum())
    n_fp = int((flagged & (y == 0)).sum())
    n_reviews = int(flagged.sum())
    return {
        "c_fp": c_fp,
        "c_rev": c_rev,
        "missed_fraud_amount": missed,
        "n_false_positives": n_fp,
        "n_reviews": n_reviews,
        "cost_false_positives": c_fp * n_fp,
        "cost_reviews": c_rev * n_reviews,
        "total_cost": missed + c_fp * n_fp + c_rev * n_reviews,
    }


def cost_sensitivity(
    y_val, s_val, amount_val, y_eval, s_eval, amount_eval, grid: Sequence[float] = SENSITIVITY_GRID
) -> pd.DataFrame:
    """Cost on the evaluation rows for every (C_FP, C_rev) pair in grid x grid, with the
    threshold re-chosen on validation for each pair."""
    rows = []
    for c_fp, c_rev in product(grid, grid):
        t = choose_cost_threshold(
            np.asarray(y_val), np.asarray(s_val), np.asarray(amount_val, dtype=float), c_fp, c_rev
        )
        flagged = np.asarray(s_eval) >= t
        rows.append({"threshold": t, **cost_breakdown(y_eval, flagged, amount_eval, c_fp, c_rev)})
    return pd.DataFrame(rows)
