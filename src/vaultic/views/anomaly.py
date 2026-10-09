"""Phase 6 anomaly view: label-free scores that can flag patterns the supervised views have
never seen. Three scores, each rank-normalised to [0, 1] on the training period (higher = more
anomalous):

  anomaly_if      global Isolation Forest trained on LEGIT training rows only
  anomaly_ae      autoencoder (3-layer MLP) trained to reconstruct LEGIT training rows;
                  score = mean squared reconstruction error of the standardised row
  anomaly_uid_if  per-customer Isolation Forest (the FYP-1 idea), fitted on each uid's
                  training-period rows; only for transactions whose uid has >= 20 past
                  transactions and a fitted model, NaN (masked) otherwise (CLAUDE.md rule 11)

The autoencoder uses scikit-learn's MLPRegressor for now; it may move to PyTorch together with
Phase 5 (research/decisions.md D35).

Categorical columns are turned into codes by the shared CategoryEncoder fitted on the training
period (features.categories, D37): pass it to AnomalyView together with DataFrame inputs, so a
category gets the same code in training, validation and replay; unseen values become 1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from vaultic.features.categories import CategoryEncoder

MIN_HISTORY = 20


class RankNormalizer:
    """Position of a score among the training-period scores: share of them <= the score."""

    def fit(self, scores) -> RankNormalizer:
        self.sorted_ = np.sort(np.asarray(scores, dtype=float))
        return self

    def transform(self, scores) -> np.ndarray:
        s = np.asarray(scores, dtype=float)
        out = np.searchsorted(self.sorted_, s, side="right") / len(self.sorted_)
        return np.where(np.isnan(s), np.nan, out)


class GlobalIsolationForest:
    def __init__(self, n_estimators: int = 200, seed: int = 0):
        self.n_estimators, self.seed = n_estimators, seed

    def fit(self, X_train, y_train) -> GlobalIsolationForest:
        legit = np.asarray(y_train) == 0
        self.model_ = make_pipeline(
            SimpleImputer(strategy="median"),
            IsolationForest(n_estimators=self.n_estimators, random_state=self.seed),
        ).fit(np.asarray(X_train, dtype=float)[legit])
        self.n_fit_ = int(legit.sum())
        self.norm_ = RankNormalizer().fit(self.raw(X_train))
        return self

    def raw(self, X) -> np.ndarray:
        return -self.model_.score_samples(np.asarray(X, dtype=float))  # higher = more anomalous

    def transform(self, X) -> np.ndarray:
        return self.norm_.transform(self.raw(X))


class AutoencoderAnomaly:
    def __init__(self, hidden=(64, 16, 64), max_iter: int = 200, seed: int = 0):
        self.hidden, self.max_iter, self.seed = hidden, max_iter, seed

    def fit(self, X_train, y_train) -> AutoencoderAnomaly:
        legit = np.asarray(y_train) == 0
        self.prep_ = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
        Z = self.prep_.fit_transform(np.asarray(X_train, dtype=float)[legit])
        self.model_ = MLPRegressor(
            hidden_layer_sizes=self.hidden, max_iter=self.max_iter, random_state=self.seed
        ).fit(Z, Z)
        self.n_fit_ = int(legit.sum())
        self.norm_ = RankNormalizer().fit(self.raw(X_train))
        return self

    def raw(self, X) -> np.ndarray:
        Z = self.prep_.transform(np.asarray(X, dtype=float))
        return np.mean((self.model_.predict(Z) - Z) ** 2, axis=1)

    def transform(self, X) -> np.ndarray:
        return self.norm_.transform(self.raw(X))


class PerUidIsolationForest:
    """One Isolation Forest per uid with at least MIN_HISTORY training-period rows."""

    def __init__(self, min_history: int = MIN_HISTORY, n_estimators: int = 200, seed: int = 0):
        self.min_history, self.n_estimators, self.seed = min_history, n_estimators, seed

    def fit(self, X_train, uid_train) -> PerUidIsolationForest:
        X = np.asarray(X_train, dtype=float)
        uid = np.asarray(uid_train)
        self.fill_ = np.nan_to_num(np.nanmedian(X, axis=0))  # per-feature median, all rows
        X = np.where(np.isnan(X), self.fill_, X)
        counts = pd.Series(uid).value_counts()
        self.models_ = {}
        raw = np.full(len(X), np.nan)
        for user in counts.index[counts >= self.min_history]:
            rows = uid == user
            forest = IsolationForest(n_estimators=self.n_estimators, random_state=self.seed)
            self.models_[user] = forest.fit(X[rows])
            raw[rows] = -forest.score_samples(X[rows])
        self.norm_ = RankNormalizer().fit(raw[~np.isnan(raw)]) if self.models_ else None
        return self

    def transform(self, X, uid, n_past) -> np.ndarray:
        X = np.where(np.isnan(np.asarray(X, dtype=float)), self.fill_, np.asarray(X, dtype=float))
        uid, n_past = np.asarray(uid), np.asarray(n_past)
        raw = np.full(len(X), np.nan)
        for user, forest in self.models_.items():
            rows = (uid == user) & (n_past >= self.min_history)
            if rows.any():
                raw[rows] = -forest.score_samples(X[rows])
        return raw if self.norm_ is None else self.norm_.transform(raw)


def numeric_matrix(X, encoder: CategoryEncoder | None = None) -> np.ndarray:
    """Model input as float: DataFrame categoricals via the shared training-period encoder."""
    if isinstance(X, pd.DataFrame):
        if encoder is not None:
            X = encoder.transform(X)
        bad = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
        if bad:
            raise ValueError(f"non-numeric columns need the CategoryEncoder: {bad}")
        return X.to_numpy(dtype=float)
    return np.asarray(X, dtype=float)


class AnomalyView:
    """All three scores; fit on the training period only."""

    def __init__(
        self,
        seed: int = 0,
        min_history: int = MIN_HISTORY,
        encoder: CategoryEncoder | None = None,
        **ae_kwargs,
    ):
        self.encoder = encoder
        self.global_if = GlobalIsolationForest(seed=seed)
        self.autoencoder = AutoencoderAnomaly(seed=seed, **ae_kwargs)
        self.per_uid = PerUidIsolationForest(min_history=min_history, seed=seed)

    def fit(self, X_train, y_train, uid_train) -> AnomalyView:
        X_train = numeric_matrix(X_train, self.encoder)
        self.global_if.fit(X_train, y_train)
        self.autoencoder.fit(X_train, y_train)
        self.per_uid.fit(X_train, uid_train)
        return self

    def transform(self, X, uid, n_past) -> pd.DataFrame:
        X = numeric_matrix(X, self.encoder)
        return pd.DataFrame(
            {
                "anomaly_if": self.global_if.transform(X),
                "anomaly_ae": self.autoencoder.transform(X),
                "anomaly_uid_if": self.per_uid.transform(X, uid, n_past),
            }
        )
