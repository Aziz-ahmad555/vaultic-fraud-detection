"""MVAF gate and fusion baselines on synthetic data where the right behaviour is known."""

import numpy as np
import pytest

from vaultic.eval.metrics import pr_auc
from vaultic.fusion.baselines import SimpleAverage, make_fusion
from vaultic.fusion.mvaf import MVAF, masked_softmax, view_dropout, view_inputs

NAN = np.nan


def _synthetic(n=3000, seed=0):
    """View A: strong signal, present 50%. View B: weak signal, always present.
    View C: pure noise, present 70%. One context column."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.1).astype(int)
    a = np.clip(0.15 + 0.7 * y + rng.normal(0, 0.12, n), 0.01, 0.99)
    b = np.clip(0.4 + 0.2 * y + rng.normal(0, 0.15, n), 0.01, 0.99)
    c = np.clip(rng.random(n), 0.01, 0.99)
    a[rng.random(n) < 0.5] = NAN
    c[rng.random(n) < 0.3] = NAN
    context = rng.normal(size=(n, 1))
    return np.column_stack([a, b, c]), y, context


# ---- masked softmax and inputs -------------------------------------------------------------


def test_masked_softmax_exact_zero_and_one():
    g = np.array([[1.0, 2.0, 3.0], [5.0, -1.0, 0.0], [0.0, 0.0, 0.0]])
    mask = np.array([[True, False, True], [False, True, False], [False, False, False]])
    w = masked_softmax(g, mask)
    assert w[0, 1] == 0.0 and w[1, 0] == 0.0 and w[1, 2] == 0.0  # missing -> exactly 0
    assert w[1, 1] == 1.0  # the only available view -> exactly 1
    assert (w[2] == 0).all()  # nothing available -> no weight at all
    assert w[0].sum() == pytest.approx(1.0)


def test_view_inputs():
    mask, logits, conf, dis = view_inputs(np.array([[0.5, NAN, 0.9], [NAN, NAN, 0.2]]))
    assert mask.tolist() == [[True, False, True], [False, False, True]]
    assert logits[0, 0] == pytest.approx(0.0) and logits[0, 1] == 0.0
    assert conf[0].tolist() == pytest.approx([0.0, 0.0, 0.8])
    assert dis[0] == pytest.approx(0.2) and dis[1] == 0.0  # std of 0.5, 0.9; one view -> 0


def test_view_dropout_never_empties_a_row():
    rng = np.random.default_rng(0)
    views = np.array([[0.2, NAN, NAN]] * 200 + [[0.2, 0.3, 0.4]] * 200 + [[NAN, NAN, NAN]] * 5)
    out = view_dropout(views, rate=0.9, rng=rng)
    available = ~np.isnan(out)
    assert available[:400].any(axis=1).all()  # every row that had a view keeps one
    assert not available[400:].any()  # rows without views stay empty
    assert not (available & np.isnan(views)).any()  # nothing missing comes back
    rate = 1 - available[200:400].sum() / 600
    # 3 views at rate 0.9: all three drop with probability 0.9**3 and one is then restored,
    # so the expected share dropped is (3 * 0.9 - 0.9**3) / 3 = 0.657
    assert rate == pytest.approx((3 * 0.9 - 0.9**3) / 3, abs=0.04)


# ---- MVAF -----------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def trained():
    views, y, ctx = _synthetic()
    return MVAF(epochs=20, seed=0).fit(views, y, ctx), views, y, ctx


def test_missing_views_get_exactly_zero_and_single_views_exactly_one(trained):
    model, views, _, ctx = trained
    w = model.view_weights(views, ctx)
    assert (w[np.isnan(views)] == 0.0).all()
    single = (~np.isnan(views)).sum(axis=1) == 1
    assert single.any() and (w[single].max(axis=1) == 1.0).all()
    assert np.allclose(w.sum(axis=1), 1.0)


def test_no_view_available_falls_back_to_context_bias(trained):
    model, _, _, ctx = trained
    p = model.predict_proba(np.full((3, 3), NAN), ctx[:3])
    assert np.isfinite(p).all() and ((p > 0) & (p < 1)).all()
    assert (model.view_weights(np.full((3, 3), NAN), ctx[:3]) == 0).all()


def test_gate_learns_which_view_to_trust(trained):
    model, views, y, ctx = trained
    w = model.view_weights(views, ctx)
    all_three = (~np.isnan(views)).all(axis=1)
    assert w[all_three, 0].mean() > w[all_three, 1].mean() > w[all_three, 2].mean()
    no_a = np.isnan(views[:, 0]) & ~np.isnan(views[:, 2])
    assert w[no_a, 1].mean() > w[no_a, 2].mean()  # without A, the weak signal beats noise
    test_views, test_y, test_ctx = _synthetic(seed=1)
    mvaf = pr_auc(test_y, model.predict_proba(test_views, test_ctx))
    average = pr_auc(test_y, SimpleAverage().fit(views, y).predict_proba(test_views))
    assert mvaf > average


def test_gradients_match_finite_differences():
    views, y, ctx = _synthetic(n=40, seed=3)
    model = MVAF(hidden=(4, 3), dropout=0.0, l2=1e-3, seed=0)
    model.fit(views, y, ctx)  # sets standardisation and shapes
    rng = np.random.default_rng(5)
    params = [p + rng.normal(0, 0.3, p.shape) for p in model.params_]
    z, logits, mask = model._raw_inputs(views, ctx)
    z = model._standardize(z)
    c = model._row_weights(y.astype(float), None)
    _, grads = model._loss_and_grads(params, z, logits, mask, y.astype(float), c)
    h = 1e-6
    for k, P in enumerate(params):
        for idx in [tuple(rng.integers(0, s) for s in P.shape) for _ in range(5)]:
            plus = [q.copy() for q in params]
            minus = [q.copy() for q in params]
            plus[k][idx] += h
            minus[k][idx] -= h
            lp, _ = model._loss_and_grads(plus, z, logits, mask, y.astype(float), c)
            lm, _ = model._loss_and_grads(minus, z, logits, mask, y.astype(float), c)
            numeric = (lp - lm) / (2 * h)
            assert grads[k][idx] == pytest.approx(numeric, rel=1e-4, abs=1e-7)


def test_weighted_bce_row_weights():
    m = MVAF(class_weight="balanced")
    y = np.array([1, 0, 0, 0])
    c = m._row_weights(y, None)
    assert c[0] * 1 == pytest.approx(c[1:].sum())  # both classes carry equal total weight
    amount = np.array([100.0, 1.0, 1.0, 1.0])
    assert m._row_weights(y, amount)[0] == pytest.approx(100 * c[0])  # cost-weighted variant


def test_same_seed_same_model():
    views, y, ctx = _synthetic(n=800)
    a = MVAF(epochs=3, seed=7).fit(views, y, ctx).predict_proba(views, ctx)
    b = MVAF(epochs=3, seed=7).fit(views, y, ctx).predict_proba(views, ctx)
    assert np.array_equal(a, b)


# ---- baselines ------------------------------------------------------------------------------


def test_f1_average_and_prior():
    f1 = SimpleAverage().fit(None, np.array([0, 0, 0, 1]))
    out = f1.predict_proba(np.array([[0.2, NAN, 0.6], [NAN, NAN, NAN]]))
    assert out.tolist() == pytest.approx([0.4, 0.25])


def test_f2_weights_are_fixed_per_availability_pattern():
    views, y, ctx = _synthetic(n=1500)
    f2 = make_fusion("F2", epochs=10).fit(views, y, ctx)
    w = f2.view_weights(views, ctx)
    full = (~np.isnan(views)).all(axis=1)
    assert np.allclose(w[full], w[full][0])  # identical whatever the context or scores
    only_ab = ~np.isnan(views[:, 0]) & np.isnan(views[:, 2])
    expected = w[full][0, :2] / w[full][0, :2].sum()  # renormalised over available views
    assert np.allclose(w[only_ab][:, :2], expected)


@pytest.mark.parametrize("name", ["F3", "F4"])
def test_stacking_baselines_handle_missing_views(name):
    views, y, ctx = _synthetic(n=1500)
    model = make_fusion(name).fit(views, y, ctx)
    p = model.predict_proba(views, ctx)
    assert p.shape == (1500,) and np.isfinite(p).all()
    assert pr_auc(y, p) > y.mean()  # better than random


def test_f5_has_no_mask_and_f6_has_no_dropout():
    views, y, ctx = _synthetic(n=1500)
    f5 = make_fusion("F5", epochs=5).fit(views, y, ctx)
    w = f5.view_weights(views, ctx)
    assert (w[np.isnan(views)] > 0).any()  # missing views still receive weight
    assert make_fusion("F6").dropout == 0.0 and make_fusion("MVAF").dropout == 0.25


def test_unknown_fusion_name():
    with pytest.raises(ValueError):
        make_fusion("F9")
