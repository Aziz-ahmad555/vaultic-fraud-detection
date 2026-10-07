"""Tabular models used by the baselines (config key `model`)."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_model(name: str, params: dict[str, Any], seed: int):
    """A fresh, unfitted classifier with predict_proba. All randomness comes from `seed`."""
    params = dict(params or {})
    if name == "logistic_regression":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(random_state=seed, **params),
        )
    if name == "random_forest":
        # scikit-learn 1.3 forests do not accept NaN; -999 is outside every raw range
        return make_pipeline(
            SimpleImputer(strategy="constant", fill_value=-999),
            RandomForestClassifier(random_state=seed, n_jobs=-1, **params),
        )
    if name == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(random_state=seed, n_jobs=-1, tree_method="hist", **params)
    if name == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(random_state=seed, n_jobs=-1, verbose=-1, deterministic=True, **params)
    raise ValueError(f"unknown model {name!r}")
