"""Phase 5 GRU temporal view on synthetic sequences only. Needs PyTorch (.venv-torch, D40);
skipped in the main venv."""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from vaultic.eval.metrics import pr_auc  # noqa: E402
from vaultic.features.categories import CategoryEncoder  # noqa: E402
from vaultic.features.sequences import build_sequences  # noqa: E402
from vaultic.views.temporal import GRUTemporalView, TemporalData  # noqa: E402


def _synthetic(n_uids=300, per_uid=12, seed=0):
    """Each uid makes small payments; a fraud is a payment far above that uid's own history
    (the "amount jumped after small transactions" pattern), so the label needs the history."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_uids):
        base = rng.uniform(5, 200)
        t = np.sort(rng.integers(0, 60 * 86_400, per_uid))
        for i in range(per_uid):
            fraud = i >= 3 and rng.random() < 0.12
            amt = base * (rng.uniform(6, 12) if fraud else rng.uniform(0.7, 1.3))
            rows.append((u, int(t[i]), amt, rng.choice(["W", "C"]), int(fraud)))
    df = pd.DataFrame(rows, columns=["uid", "TransactionDT", "TransactionAmt", "ProductCD",
                                     "isFraud"])  # fmt: skip
    df = df.sort_values("TransactionDT", kind="stable").reset_index(drop=True)
    return df


def _data(df, enc, n_steps=10):
    s = build_sequences(df, df["uid"], enc, n_steps=n_steps)
    return TemporalData.from_sequences(s, df["isFraud"].to_numpy())


@pytest.fixture(scope="module")
def trained():
    df = _synthetic()
    cut = df["TransactionDT"].quantile(0.7)
    enc = CategoryEncoder().fit(df[df["TransactionDT"] <= cut], columns=("ProductCD",))
    data = _data(df, enc)
    tr = np.flatnonzero(df["TransactionDT"] <= cut)
    va = np.flatnonzero(df["TransactionDT"] > cut)
    view = GRUTemporalView(n_products=enc.n_codes("ProductCD"), max_epochs=40, patience=6,
                           seed=0).fit(data.subset(tr), data.subset(va))  # fmt: skip
    return view, data, tr, va, enc, df


def test_learns_a_pattern_that_needs_the_history(trained):
    view, data, tr, va, enc, df = trained
    val = data.subset(va)
    p = view.predict_proba(val)
    has = val.mask.any(axis=1)
    prevalence = val.y[has].mean()
    assert pr_auc(val.y[has], p[has]) > max(0.8, 4 * prevalence)


def test_no_history_is_masked_not_scored(trained):
    view, data, tr, va, enc, df = trained
    p = view.predict_proba(data)
    first = ~data.mask.any(axis=1)
    assert first.any() and np.isnan(p[first]).all()
    assert np.isfinite(p[~first]).all() and ((p[~first] > 0) & (p[~first] < 1)).all()


def test_padding_never_changes_the_score(trained):
    view, data, tr, va, enc, df = trained
    rows = np.flatnonzero(data.mask.any(axis=1) & ~data.mask.all(axis=1))[:50]
    sub = data.subset(rows)
    noisy = TemporalData(sub.values.copy(), sub.mask, sub.current, sub.y)
    noisy.values[~sub.mask] = 123.0  # garbage in padded positions
    assert np.allclose(view.predict_proba(sub), view.predict_proba(noisy), atol=1e-6)
    # the same history left-padded to 20 steps gives the same score as with 10
    wide = TemporalData(
        np.concatenate([np.zeros((len(rows), 10, sub.values.shape[2]), np.float32), sub.values], 1),
        np.concatenate([np.zeros((len(rows), 10), bool), sub.mask], 1),
        sub.current,
    )
    assert np.allclose(view.predict_proba(sub), view.predict_proba(wide), atol=1e-6)


def test_early_stopping_keeps_the_best_epoch(trained):
    view, data, tr, va, enc, df = trained
    scores = [h["val_pr_auc"] for h in view.history_]
    assert view.best_val_pr_auc_ == max(scores)
    assert len(scores) < view.max_epochs or view.best_epoch_ == len(scores) - 1
    if len(scores) < view.max_epochs:  # stopped: exactly `patience` epochs without a gain
        assert len(scores) - 1 - view.best_epoch_ == view.patience
    val = data.subset(va)
    has = val.mask.any(axis=1)
    p = view.predict_proba(val)
    assert pr_auc(val.y[has], p[has]) == pytest.approx(view.best_val_pr_auc_)


def test_weighted_bce_uses_balanced_class_weight(trained):
    view, data, tr, va, enc, df = trained
    train = data.subset(tr)
    y = train.y[train.mask.any(axis=1)]
    assert view.pos_weight_ == pytest.approx((y == 0).sum() / (y == 1).sum())
    assert view.n_parameters() > 0


def test_same_seed_same_scores():
    df = _synthetic(n_uids=60, per_uid=8, seed=1)
    enc = CategoryEncoder().fit(df, columns=("ProductCD",))
    data = _data(df, enc, n_steps=5)
    half = len(df) // 2
    tr, va = np.arange(half), np.arange(half, len(df))
    a = GRUTemporalView(enc.n_codes("ProductCD"), max_epochs=3, seed=7).fit(data.subset(tr),
                                                                            data.subset(va))  # fmt: skip
    b = GRUTemporalView(enc.n_codes("ProductCD"), max_epochs=3, seed=7).fit(data.subset(tr),
                                                                            data.subset(va))  # fmt: skip
    assert np.array_equal(a.predict_proba(data), b.predict_proba(data), equal_nan=True)


def test_scores_do_not_depend_on_future_rows(trained):
    """Leakage, end to end: rewriting every transaction after a cut leaves the scores of the
    earlier transactions unchanged (sequences are strictly past; the model is fixed)."""
    view, data, tr, va, enc, df = trained
    cut = df["TransactionDT"].quantile(0.5)
    changed = df.copy()
    future = changed["TransactionDT"] > cut
    changed.loc[future, "TransactionAmt"] = 1e6
    changed.loc[future, "ProductCD"] = "S"
    before = view.predict_proba(_data(df, enc))
    after = view.predict_proba(_data(changed, enc))
    keep = (~future).to_numpy()
    assert np.array_equal(before[keep], after[keep], equal_nan=True)


def test_training_options_and_small_batch_overfit():
    """D74: sqrt class weight, lr decay and gradient clipping train; the model can overfit a
    small batch (sanity check for the GRU diagnosis)."""
    rng = np.random.default_rng(0)
    n, steps, feats = 600, 5, 3
    values = rng.normal(size=(n, steps, feats)).astype(np.float32)
    values[..., 2] = rng.integers(1, 4, size=(n, steps))
    mask = np.ones((n, steps), bool)
    current = rng.normal(size=(n, feats)).astype(np.float32)
    current[:, 2] = 1
    y = (values[:, -1, 0] > 1.0).astype(int)
    data = TemporalData(values, mask, current, y)
    view = GRUTemporalView(n_products=5, pos_weight="sqrt", lr_decay=0.9, grad_clip=1.0,
                           max_epochs=3, seed=0)  # fmt: skip
    view.fit(data.subset(np.arange(400)), data.subset(np.arange(400, 600)))
    assert view.pos_weight_ == pytest.approx(np.sqrt((y[:400] == 0).sum() / y[:400].sum()))
    check = view.overfit_check(data, n=64, epochs=150)
    assert check["train_pr_auc"] > 0.95


def test_robust_inputs_tame_heavy_tails():
    """D85: signed log1p + clipping keeps one huge counter from dominating the inputs."""
    rng = np.random.default_rng(0)
    n, steps = 300, 4
    values = np.abs(rng.normal(size=(n, steps, 3))).astype(np.float32)
    values[..., 2] = 1
    values[0, 0, 0] = 1e6  # one extreme counter value
    data = TemporalData(values, np.ones((n, steps), bool), values[:, -1].copy(),
                        (rng.random(n) < 0.2).astype(int))  # fmt: skip
    plain = GRUTemporalView(n_products=3)
    plain._fit_scaler(data)
    robust = GRUTemporalView(n_products=3, robust_inputs=True)
    robust._fit_scaler(data)
    v_plain = plain._tensors(data)[0].numpy()[..., 0]
    v_robust = robust._tensors(data)[0].numpy()[..., 0]
    assert np.std(v_plain.ravel()[1:]) < 0.001  # every other value squashed to ~one point
    assert np.abs(v_robust).max() <= 5.0 and np.median(np.abs(v_robust)) > 0.3
