"""Phase 9.1 counterfactuals: a DiCE wrapper with immutable features locked.

"From a known device and under $120, risk falls to 31." A counterfactual may only change what
could have been different about THIS transaction; it may never change who the customer is or
what happened before (research/decisions.md D38):

  immutable  card fields, billing address, and every history feature (customer history,
             velocity, labels known, frequency, entity / graph features, masked counters C and
             time deltas D, which describe past activity). Never varied.
  derived    functions of the current transaction AND its history (amount z-score, ratio to
             the usual amount, novelty flags, anomaly scores, ...). Varying them freely would
             give impossible combinations (amount unchanged but its z-score changed), so they
             are locked unless a `derive` callback recomputes them from the changed inputs.
  actionable amount, product, email domains, device: the default features DiCE may vary.

The lock is enforced twice: DiCE is told the features it may vary, and every candidate it
returns is checked again; one that changed a locked feature is rejected and counted. Validity
is always re-checked with the model itself after `derive`.

dice-ml is imported lazily, so everything except DiceBackend works without it.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

IMMUTABLE_PATTERNS = (
    r"card[1-6]",
    r"addr[12]",
    r"dist[12]",
    r"C\d+",
    r"D\d+",
    r"uid.*",
    r"hist_.*",
    r"vel_.*",
    r"freq_.*",
    r"ent_uids_.*",
    r"g_.*",
    r"mean_gap",
    r"TransactionDT",
    r"day",
    r"hour",
)
DERIVED_PATTERNS = (
    r"amt_.*",
    r"secs_since_prev",
    r"gap_ratio",
    r"new_.*",
    r"n_new_attributes",
    r"hour_deviation",
    r"anomaly_.*",
)
ACTIONABLE = ("TransactionAmt", "ProductCD", "P_emaildomain", "R_emaildomain", "DeviceType",
              "DeviceInfo")  # fmt: skip


def _matches(col: str, patterns: Sequence[str]) -> bool:
    return any(re.fullmatch(p, col) for p in patterns)


def immutable_features(columns: Sequence[str]) -> list[str]:
    return [c for c in columns if _matches(c, IMMUTABLE_PATTERNS)]


def derived_features(columns: Sequence[str]) -> list[str]:
    return [c for c in columns if _matches(c, DERIVED_PATTERNS)]


def features_to_vary(
    columns: Sequence[str], vary: Sequence[str] | None = None, derive: Callable | None = None
) -> list[str]:
    """Features the generator may change. `vary` defaults to the actionable ones; a requested
    feature that is immutable (or derived without a `derive` callback) raises."""
    locked = set(immutable_features(columns))
    if derive is None:
        locked |= set(derived_features(columns))
    wanted = [c for c in (ACTIONABLE if vary is None else vary) if c in columns]
    bad = sorted(set(wanted) & locked)
    if vary is not None and bad:
        raise ValueError(f"locked features cannot be varied: {bad}")
    return [c for c in wanted if c not in locked]


@dataclass
class CounterfactualResult:
    query: pd.DataFrame  # one row
    risk_before: float
    candidates: pd.DataFrame  # accepted counterfactuals (after derive)
    risk_after: np.ndarray
    valid: np.ndarray  # risk_after < threshold
    rejected_locked: int  # candidates dropped because they changed a locked feature
    changed: list[list[str]] = field(default_factory=list)


class DiceBackend:
    """dice-ml (Mothilal et al. 2020) as the candidate generator."""

    def __init__(self, model, train: pd.DataFrame, outcome: str, continuous: Sequence[str],
                 method: str = "random", seed: int = 0):  # fmt: skip
        import dice_ml

        data = dice_ml.Data(dataframe=train, continuous_features=list(continuous),
                            outcome_name=outcome)  # fmt: skip
        wrapped = dice_ml.Model(model=model, backend="sklearn")
        self.explainer = dice_ml.Dice(data, wrapped, method=method)
        self.seed = seed

    def __call__(self, query: pd.DataFrame, total: int, vary: list[str]) -> pd.DataFrame:
        kwargs = {"random_seed": self.seed} if self.explainer.__class__.__name__.endswith(
            "Random"
        ) else {}  # fmt: skip
        result = self.explainer.generate_counterfactuals(
            query, total_CFs=total, desired_class=0, features_to_vary=vary, **kwargs
        )
        cfs = result.cf_examples_list[0].final_cfs_df
        return pd.DataFrame(columns=query.columns) if cfs is None else cfs[query.columns]


class CounterfactualExplainer:
    """model: anything with predict_proba(DataFrame) -> (n, 2); backend: a callable
    (query, total, vary) -> DataFrame of candidates (DiceBackend by default)."""

    def __init__(
        self,
        model,
        columns: Sequence[str],
        backend: Callable[[pd.DataFrame, int, list[str]], pd.DataFrame],
        threshold: float = 0.5,
        vary: Sequence[str] | None = None,
        derive: Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame] | None = None,
    ):
        self.model = model
        self.columns = list(columns)
        self.backend = backend
        self.threshold = threshold
        self.derive = derive
        self.vary = features_to_vary(self.columns, vary, derive)
        self.locked = [c for c in self.columns if c not in self.vary]

    def _risk(self, frame: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.model.predict_proba(frame[self.columns]))[:, 1]

    def explain(self, query: pd.DataFrame, total: int = 3) -> CounterfactualResult:
        if len(query) != 1:
            raise ValueError("explain one transaction at a time")
        query = query[self.columns]
        raw = self.backend(query, total, list(self.vary))
        raw = raw[self.columns].reset_index(drop=True)
        if self.derive is not None and len(raw):
            raw = self.derive(raw, query)[self.columns]
        q = query.iloc[0]
        changed_locked = np.zeros(len(raw), dtype=bool)
        recomputed = set(derived_features(self.columns)) if self.derive is not None else set()
        for col in self.locked:
            if col in recomputed:
                continue  # recomputed by derive, allowed to change
            same = [_same(a, q[col]) for a in raw[col]]
            changed_locked |= ~np.array(same, dtype=bool)
        accepted = raw[~changed_locked].reset_index(drop=True)
        risk_after = self._risk(accepted) if len(accepted) else np.array([])
        changed = [
            [c for c in self.vary if not _same(accepted.at[i, c], q[c])]
            for i in range(len(accepted))
        ]
        return CounterfactualResult(
            query=query,
            risk_before=float(self._risk(query)[0]),
            candidates=accepted,
            risk_after=risk_after,
            valid=risk_after < self.threshold,
            rejected_locked=int(changed_locked.sum()),
            changed=changed,
        )


def _same(a, b) -> bool:
    if pd.isna(a) and pd.isna(b):
        return True
    return bool(a == b)


def quality(result: CounterfactualResult, scale: dict[str, float] | None = None) -> dict:
    """Roadmap 9.3 counterfactual quality for one query: validity (share of candidates whose
    risk really falls below the threshold), proximity (mean over changed numeric features of
    |change| / scale, e.g. the training MAD; categorical changes count 1), features changed,
    and whether every changed feature is actionable."""
    if not len(result.candidates):
        return {"n": 0, "validity": np.nan, "proximity": np.nan, "n_changed": np.nan,
                "all_actionable": np.nan}  # fmt: skip
    scale = scale or {}
    q = result.query.iloc[0]
    dists, counts, actionable = [], [], []
    for i, cols in enumerate(result.changed):
        d = []
        for c in cols:
            a, b = result.candidates.at[i, c], q[c]
            if isinstance(a, (int, float, np.number)) and isinstance(b, (int, float, np.number)):
                d.append(abs(float(a) - float(b)) / scale.get(c, 1.0))
            else:
                d.append(1.0)
        dists.append(np.mean(d) if d else 0.0)
        counts.append(len(cols))
        actionable.append(all(c in ACTIONABLE for c in cols))
    return {
        "n": len(result.candidates),
        "validity": float(np.mean(result.valid)),
        "proximity": float(np.mean(dists)),
        "n_changed": float(np.mean(counts)),
        "all_actionable": float(np.mean(actionable)),
    }


def describe(result: CounterfactualResult, i: int | None = None) -> str | None:
    """'If the amount were $120.00 instead of $950.00, risk would fall from 87 to 31.'
    Uses the valid candidate with the fewest changes (lowest risk on ties); None if none."""
    if i is None:
        valid = np.flatnonzero(result.valid)
        if not len(valid):
            return None
        i = int(min(valid, key=lambda j: (len(result.changed[j]), result.risk_after[j])))
    q = result.query.iloc[0]
    parts = [f"{_label(c)} were {_fmt(c, result.candidates.at[i, c])} instead of "
             f"{_fmt(c, q[c])}" for c in result.changed[i]]  # fmt: skip
    if not parts:
        return None
    return (f"If the {' and the '.join(parts)}, the risk would fall from "
            f"{100 * result.risk_before:.0f} to {100 * result.risk_after[i]:.0f}.")  # fmt: skip


_LABELS = {"TransactionAmt": "amount", "ProductCD": "product code", "DeviceInfo": "device",
           "DeviceType": "device type", "P_emaildomain": "purchaser email domain",
           "R_emaildomain": "recipient email domain"}  # fmt: skip


def _label(col: str) -> str:
    return _LABELS.get(col, col)


def _fmt(col: str, value) -> str:
    if pd.isna(value):
        return "missing"
    if col == "TransactionAmt":
        return f"${float(value):,.2f}"
    return str(value)
