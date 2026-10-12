"""Tabular models used by the baselines (config key `model`)."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DEVICES = ("cpu", "cuda")
GPU_MODELS = {"xgboost", "lightgbm"}  # the others always run on CPU


def resolve_device(cli: str | None = None, config: str | None = None) -> str:
    """Requested device: --device flag, then VAULTIC_DEVICE, then the config, then cpu."""
    device = cli or os.environ.get("VAULTIC_DEVICE") or config or "cpu"
    if device not in DEVICES:
        raise ValueError(f"device must be one of {DEVICES}, got {device!r}")
    return device


def effective_device(name: str, device: str) -> str:
    """Device a model actually trains on (CPU-only models ignore a GPU request)."""
    return device if name in GPU_MODELS else "cpu"


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Clip each column to its quantiles learned at fit time (the training period only).

    B1 (D61): the raw masked columns are extremely heavy-tailed on the training rows (192 of 406
    reach |z| > 50), so a linear model extrapolates without bound on extreme rows; clipping to
    the training range keeps the logit finite and lets lbfgs converge."""

    def __init__(self, low: float = 0.001, high: float = 0.999):
        self.low, self.high = low, high

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lo_ = np.nanquantile(X, self.low, axis=0)
        self.hi_ = np.nanquantile(X, self.high, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)


def make_model(name: str, params: dict[str, Any], seed: int, device: str = "cpu"):
    """A fresh, unfitted classifier with predict_proba. All randomness comes from `seed`.

    device="cuda" trains XGBoost and LightGBM on the GPU; results can differ slightly from CPU
    (different floating-point summation order), so the device is recorded with every run.
    """
    params = dict(params or {})
    if name == "logistic_regression":
        clip = params.pop("clip_quantiles", None)  # D61: clip to training quantiles first
        steps = [SimpleImputer(strategy="median")]
        if clip is not None:
            steps.append(QuantileClipper(*clip))
        return make_pipeline(
            *steps, StandardScaler(), LogisticRegression(random_state=seed, **params)
        )
    if name == "random_forest":
        # scikit-learn 1.3 forests do not accept NaN; -999 is outside every raw range
        return make_pipeline(
            SimpleImputer(strategy="constant", fill_value=-999),
            RandomForestClassifier(random_state=seed, n_jobs=-1, **params),
        )
    if name == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            random_state=seed, n_jobs=-1, tree_method="hist", device=device, **params
        )
    if name == "lightgbm":
        from lightgbm import LGBMClassifier

        gpu = {}
        if device == "cuda":
            # "gpu" = LightGBM's OpenCL build; set VAULTIC_LIGHTGBM_GPU=cuda for a CUDA build
            gpu = {"device_type": os.environ.get("VAULTIC_LIGHTGBM_GPU", "gpu")}
        return LGBMClassifier(
            random_state=seed, n_jobs=-1, verbose=-1, deterministic=True, **gpu, **params
        )
    if name == "fyp1":
        from vaultic.views.fyp1 import FYP1Model

        return FYP1Model(seed=seed)
    raise ValueError(f"unknown model {name!r}")
