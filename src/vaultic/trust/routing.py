"""Phase 8.4 budget-aware routing: which K transactions per day go to analyst review.

Policies (priority per transaction, highest K per day are reviewed):
  R1  risk:                          p
  R2  expected loss:                 p * amount
  R3  R2 + uncertainty bonus:        p * amount + lambda_u * u * amount
  R4  R3 + disagreement bonus:       ... + lambda_d * d * amount

u in [0, 1] is uncertainty (1 for an {legit, fraud} conformal set; or 1 - |2p - 1| when no
sets are given) and d in [0, 1] is view disagreement. lambda_u and lambda_d are parameters to
choose on validation, never on test. Ties are broken by row order, so results are
deterministic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

POLICIES = ("R1", "R2", "R3", "R4")


def priority(
    policy: str,
    p: np.ndarray,
    amount: np.ndarray,
    u: np.ndarray | None = None,
    d: np.ndarray | None = None,
    lambda_u: float = 1.0,
    lambda_d: float = 1.0,
) -> np.ndarray:
    p, amount = np.asarray(p, dtype=np.float64), np.asarray(amount, dtype=np.float64)
    if u is None:
        u = 1 - np.abs(2 * p - 1)
    if policy == "R1":
        return p
    score = p * amount
    if policy == "R2":
        return score
    score = score + lambda_u * np.asarray(u, dtype=np.float64) * amount
    if policy == "R3":
        return score
    if policy == "R4":
        if d is None:
            raise ValueError("R4 needs view disagreement d")
        return score + lambda_d * np.asarray(d, dtype=np.float64) * amount
    raise ValueError(f"unknown policy {policy!r}")


def select_daily(day: np.ndarray, prio: np.ndarray, k: int) -> np.ndarray:
    """Boolean mask of the top-k rows by priority within each day (stable on ties)."""
    frame = pd.DataFrame({"day": np.asarray(day), "prio": prio, "row": np.arange(len(prio))})
    frame = frame.sort_values(["day", "prio", "row"], ascending=[True, False, True])
    chosen = frame.groupby("day", sort=False).head(k)["row"].to_numpy()
    mask = np.zeros(len(prio), dtype=bool)
    mask[chosen] = True
    return mask


def evaluate_policy(
    day, y, amount, p, policy: str, k: int, u=None, d=None, lambda_u=1.0, lambda_d=1.0
) -> dict[str, float]:
    y, amount = np.asarray(y).astype(int), np.asarray(amount, dtype=np.float64)
    reviewed = select_daily(day, priority(policy, p, amount, u, d, lambda_u, lambda_d), k)
    caught = reviewed & (y == 1)
    total_fraud_value = float(amount[y == 1].sum())
    return {
        "policy": policy,
        "K": k,
        "reviews": int(reviewed.sum()),
        "fraud_caught": int(caught.sum()),
        "fraud_value_caught": float(amount[caught].sum()),
        "recall_at_k": float(caught.sum() / max((y == 1).sum(), 1)),
        "value_recall": (
            float(amount[caught].sum() / total_fraud_value) if total_fraud_value else 0.0
        ),
        "money_saved_per_review": float(amount[caught].sum() / max(reviewed.sum(), 1)),
    }


def compare_policies(day, y, amount, p, ks=(50, 100, 200, 500), u=None, d=None, **lambdas):
    rows = [
        evaluate_policy(day, y, amount, p, pol, k, u, d, **lambdas)
        for k in ks
        for pol in POLICIES
        if not (pol == "R4" and d is None)
    ]
    return pd.DataFrame(rows)
