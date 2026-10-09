"""Fusion baselines F1-F7 (roadmap Phase 7.4; F7 added by decision D31/D33). MVAF must beat them or be reported honestly.

F1  simple average of the available view probabilities
F2  fixed weights per view, fitted on validation (renormalised over available views)
F3  logistic-regression stacking, missing views filled with a constant (logit 0, i.e. p = 0.5)
F4  LightGBM stacking with missing views left as NaN
F5  MVAF gate without the availability mask
F6  MVAF gate with the mask but without view dropout
F7  SimMLM-style gate: MVAF's gate trained with modality dropout and the MoFe ranking loss

All share one interface: fit(views, y, context=None, sample_weight=None) and
predict_proba(views, context=None), with views as an (n, V) array of probabilities and NaN for
missing. F3 and F4 also see the context, like the MVAF gate, so every method has the same
information. Class weighting is "balanced" everywhere by default, matching MVAF's weighted BCE.
"""

from __future__ import annotations

import numpy as np

from vaultic.fusion.mvaf import MVAF, _logit


def _stack(views: np.ndarray, context: np.ndarray | None, fill: float | None) -> np.ndarray:
    views = np.asarray(views, dtype=np.float64)
    cols = np.where(np.isnan(views), np.nan, _logit(np.nan_to_num(views, nan=0.5)))
    if fill is not None:
        cols = np.where(np.isnan(cols), fill, cols)
    return cols if context is None else np.hstack([cols, np.asarray(context, dtype=np.float64)])


class SimpleAverage:
    """F1: mean of the available views; the training fraud rate when no view is available."""

    def fit(self, views, y, context=None, sample_weight=None) -> SimpleAverage:
        self.prior_ = float(np.mean(y))
        return self

    def predict_proba(self, views, context=None) -> np.ndarray:
        views = np.asarray(views, dtype=np.float64)
        count = (~np.isnan(views)).sum(axis=1)
        total = np.nansum(views, axis=1)
        return np.where(count > 0, total / np.maximum(count, 1), self.prior_)


class LogisticStacking:
    """F3: logistic regression on view logits (missing -> 0) and context."""

    def __init__(self, class_weight="balanced", C=1.0, seed=0):
        self.class_weight, self.C, self.seed = class_weight, C, seed

    def fit(self, views, y, context=None, sample_weight=None) -> LogisticStacking:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        self.model_ = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=self.C, class_weight=self.class_weight, max_iter=1000, random_state=self.seed
            ),
        )
        self.model_.fit(
            _stack(views, context, fill=0.0), y, logisticregression__sample_weight=sample_weight
        )
        return self

    def predict_proba(self, views, context=None) -> np.ndarray:
        return self.model_.predict_proba(_stack(views, context, fill=0.0))[:, 1]


class LightGBMStacking:
    """F4: LightGBM on view logits (missing stays NaN) and context."""

    def __init__(self, class_weight="balanced", n_estimators=200, num_leaves=15, seed=0):
        self.class_weight, self.seed = class_weight, seed
        self.n_estimators, self.num_leaves = n_estimators, num_leaves

    def fit(self, views, y, context=None, sample_weight=None) -> LightGBMStacking:
        from lightgbm import LGBMClassifier

        self.model_ = LGBMClassifier(
            n_estimators=self.n_estimators,
            num_leaves=self.num_leaves,
            class_weight=self.class_weight,
            random_state=self.seed,
            deterministic=True,
            verbose=-1,
        )
        self.model_.fit(_stack(views, context, fill=None), y, sample_weight=sample_weight)
        return self

    def predict_proba(self, views, context=None) -> np.ndarray:
        return self.model_.predict_proba(_stack(views, context, fill=None))[:, 1]


def make_fusion(name: str, seed: int = 0, **kwargs):
    """F1-F7 or MVAF by name; kwargs go to the underlying model."""
    if name == "F1":
        return SimpleAverage()
    if name == "F2":
        return MVAF(gate_inputs="constant", dropout=0.0, seed=seed, **kwargs)
    if name == "F3":
        return LogisticStacking(seed=seed, **kwargs)
    if name == "F4":
        return LightGBMStacking(seed=seed, **kwargs)
    if name == "F5":
        return MVAF(use_mask=False, seed=seed, **kwargs)
    if name == "F6":
        return MVAF(dropout=0.0, seed=seed, **kwargs)
    if name == "F7":
        from vaultic.fusion.simmlm import SimMLMGate

        return SimMLMGate(seed=seed, **kwargs)
    if name == "MVAF":
        return MVAF(seed=seed, **kwargs)
    raise ValueError(f"unknown fusion method {name!r}")
