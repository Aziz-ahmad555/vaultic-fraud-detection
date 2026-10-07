"""Phase 6 anomaly view on synthetic data where the anomalies are known."""

import numpy as np
import pytest

from vaultic.eval.metrics import roc_auc
from vaultic.views.anomaly import (
    AnomalyView,
    AutoencoderAnomaly,
    GlobalIsolationForest,
    PerUidIsolationForest,
    RankNormalizer,
)


def _data(n=3000, d=6, seed=0):
    """Legit rows around 0; fraud rows shifted far away in the first three features."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.05).astype(int)
    X = rng.normal(size=(n, d))
    X[y == 1, :3] += 4.0
    X[rng.random((n, d)) < 0.02] = np.nan  # some missing values
    return X, y


def test_rank_normalizer_by_hand():
    norm = RankNormalizer().fit([1.0, 2.0, 3.0, 4.0])
    assert norm.transform([0.0, 2.5, 4.0, 10.0]).tolist() == [0.0, 0.5, 1.0, 1.0]
    assert np.isnan(norm.transform([np.nan])[0])


@pytest.mark.parametrize("cls", [GlobalIsolationForest, AutoencoderAnomaly])
def test_global_models_train_on_legit_rows_and_flag_unseen_fraud(cls):
    X, y = _data()
    model = cls(seed=0).fit(X, y)
    assert model.n_fit_ == int((y == 0).sum())  # fraud rows never reach the model
    train_scores = model.transform(X)
    assert train_scores.min() > 0 and train_scores.max() == 1.0  # rank-normalised on train
    X_new, y_new = _data(seed=1)
    assert roc_auc(y_new, model.transform(X_new)) > 0.9  # label-free, still separates fraud


def test_per_uid_forest_masks_short_histories():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 3))
    uid = np.array(["heavy"] * 40 + ["light"] * 20)  # light has exactly 20 rows
    uid[-5:] = "rare"  # light now has 15 -> no model
    model = PerUidIsolationForest(min_history=20).fit(X, uid)
    assert set(model.models_) == {"heavy"}
    X_new = np.vstack([np.zeros((1, 3)), np.full((1, 3), 8.0), np.zeros((2, 3))])
    out = model.transform(X_new, np.array(["heavy", "heavy", "heavy", "light"]),
                          n_past=np.array([25, 25, 5, 30]))  # fmt: skip
    assert out[1] > out[0]  # the outlier scores higher for its own customer
    assert np.isnan(out[2])  # fewer than 20 past transactions -> masked
    assert np.isnan(out[3])  # no model for this uid -> masked
    assert 0 <= out[0] <= 1 and out[1] == 1.0


def test_anomaly_view_columns_and_determinism():
    X, y = _data(n=800)
    uid = np.array([f"u{i % 20}" for i in range(800)])  # 40 rows each
    a = AnomalyView(seed=3, max_iter=50).fit(X, y, uid).transform(X, uid, np.full(800, 30))
    b = AnomalyView(seed=3, max_iter=50).fit(X, y, uid).transform(X, uid, np.full(800, 30))
    assert list(a.columns) == ["anomaly_if", "anomaly_ae", "anomaly_uid_if"]
    assert a.equals(b)
    assert a.notna().all().all()
