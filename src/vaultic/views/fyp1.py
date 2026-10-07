"""B6: the FYP-1 models, ported to IEEE-CIS under the temporal split (experiment E2).

FYP-1 (legacy/fyp1/ml_model.py, global_model.py, ml/preprocess_data.py) scored a transaction
as follows, and this port keeps every rule:
  - global XGBoost on 14 selected columns (median-imputed numbers, one-hot text columns),
    200 trees, depth 6, learning rate 0.1, scale_pos_weight = #legit / #fraud;
  - a per-user Isolation Forest (200 trees, contamination 0.01) for users with >= 30
    transactions, on 8 behavioural features; personal score = 1 / (1 + exp(raw));
  - blend 0.5 * personal + 0.5 * global when a personal model exists, else global;
  - rule: amount >= 5x the user's previous maximum (>= 3 earlier transactions) -> >= 0.95.

Differences forced by the temporal split, all on the side of correctness: every model and
imputation is fitted on the train period only (FYP-1 used a random split and full-data
medians); behavioural features use only earlier transactions; users are the reconstructed
uid; device = DeviceInfo; hour/weekday are relative (IEEE-CIS has no calendar time); no
analyst labels exist, so contamination stays at FYP-1's default 0.01.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from vaultic.data.load import SECONDS_PER_DAY
from vaultic.features.pit import PastIndex

GLOBAL_NUMERIC = ["TransactionAmt", "card1", "card2", "card3", "card5", "addr1", "addr2"]
GLOBAL_NUMERIC += ["C1", "C2", "C13"]
GLOBAL_TEXT = ["ProductCD", "card4", "card6", "P_emaildomain"]
IF_FEATURES = [
    "amount", "log_amount", "hour", "dow", "delta_prev_seconds",
    "rolling_mean_10", "rolling_count_10", "device_change",
]  # fmt: skip
MIN_HISTORY = 30
EXTREME_MULTIPLIER = 5.0
MIN_HISTORY_FOR_RULE = 3


def build_fyp1_frame(df: pd.DataFrame, uid: pd.Series) -> pd.DataFrame:
    """Model input for B6: global columns, point-in-time behavioural features, uid."""
    time = df["TransactionDT"].to_numpy(dtype=np.int64)
    amount = df["TransactionAmt"].to_numpy(dtype=np.float64)
    device = df["DeviceInfo"].astype(object).where(df["DeviceInfo"].notna(), "").to_numpy()
    past = PastIndex(uid.to_numpy(), time)

    n_past = past.count_before(time)
    last_time = past.last_time_before(time)
    sum10, count10 = past.last_k_sum_count(amount, time, k=10)
    prev_device = past.value_before(device, time, fill="")
    out = df[GLOBAL_NUMERIC + GLOBAL_TEXT].copy()
    for col in GLOBAL_TEXT:
        out[col] = out[col].astype(object).where(out[col].notna(), "unknown").astype(str)
    out["amount"] = amount
    out["log_amount"] = np.log1p(np.clip(amount, 0, None))
    out["hour"] = (time // 3600) % 24
    out["dow"] = (time // SECONDS_PER_DAY) % 7
    out["delta_prev_seconds"] = np.where(n_past > 0, time - np.nan_to_num(last_time), 0.0)
    out["rolling_mean_10"] = np.where(count10 > 0, sum10 / np.maximum(count10, 1), 0.0)
    out["rolling_count_10"] = count10.astype(float)
    out["device_change"] = ((device != "") & (device != prev_device)).astype(float)
    out["_uid"] = uid.to_numpy()
    out["_n_past"] = n_past
    out["_hist_max"] = past.max_before(amount, time)
    return out


class FYP1Model:
    """Global XGBoost + per-user Isolation Forests + extreme-amount rule (see module doc)."""

    def __init__(self, seed: int = 0):
        self.seed = seed

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> FYP1Model:
        from xgboost import XGBClassifier

        y = np.asarray(y)
        prep = ColumnTransformer(
            [
                ("num", SimpleImputer(strategy="median"), GLOBAL_NUMERIC),
                ("txt", OneHotEncoder(handle_unknown="ignore"), GLOBAL_TEXT),
            ]
        )
        self.global_ = make_pipeline(
            prep,
            XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                scale_pos_weight=float((y == 0).sum() / max((y == 1).sum(), 1)),
                eval_metric="logloss",
                random_state=self.seed,
                n_jobs=-1,
            ),
        )
        self.global_.fit(X[GLOBAL_NUMERIC + GLOBAL_TEXT], y)

        self.personal_ = {}
        counts = X["_uid"].value_counts()
        for user in counts.index[counts >= MIN_HISTORY]:
            rows = X.loc[X["_uid"] == user, IF_FEATURES].to_numpy(dtype=float)
            forest = IsolationForest(n_estimators=200, contamination=0.01, random_state=self.seed)
            self.personal_[user] = forest.fit(rows)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        # float64: XGBoost returns float32, which would turn the rule's 0.95 into 0.9499999
        score = self.global_.predict_proba(X[GLOBAL_NUMERIC + GLOBAL_TEXT])[:, 1].astype(np.float64)
        users = X["_uid"].to_numpy()
        for user, forest in self.personal_.items():
            rows = users == user
            if rows.any():
                raw = forest.decision_function(X.loc[rows, IF_FEATURES].to_numpy(dtype=float))
                personal = 1.0 / (1.0 + np.exp(np.clip(raw, -700, 700)))
                score[rows] = 0.5 * personal + 0.5 * score[rows]
        hist_max = X["_hist_max"].to_numpy()
        rule = (
            (X["_n_past"].to_numpy() >= MIN_HISTORY_FOR_RULE)
            & (hist_max > 0)
            & (X["amount"].to_numpy() >= EXTREME_MULTIPLIER * np.nan_to_num(hist_max))
        )
        score[rule] = np.maximum(score[rule], 0.95)
        return np.column_stack([1 - score, score])
