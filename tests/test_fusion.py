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


# ---- F7: SimMLM-style gate with the MoFe ranking loss ---------------------------------------

from vaultic.fusion.simmlm import SimMLMGate, mofe_term, row_bce  # noqa: E402


def test_mofe_is_zero_when_more_views_give_lower_loss():
    more = np.array([0.10, 0.20, 0.05])
    fewer = np.array([0.30, 0.50, 0.05])  # more views never worse (equal on the last row)
    assert mofe_term(more, fewer) == 0.0
    # when more views are worse, the hinge is the excess, averaged over rows
    assert mofe_term(np.array([0.5, 0.1]), np.array([0.2, 0.3])) == pytest.approx(0.3 / 2)


def test_f7_gradients_match_finite_differences():
    views, y, ctx = _synthetic(n=40, seed=4)
    model = SimMLMGate(hidden=(4, 3), mofe_weight=0.7, l2=1e-3, epochs=1, seed=0)
    model.fit(views, y, ctx)
    rng = np.random.default_rng(6)
    params = [p + rng.normal(0, 0.3, p.shape) for p in model.params_]
    more = model._inputs(views, ctx)
    fewer = model._inputs(view_dropout(views, 0.5, np.random.default_rng(1)), ctx)
    yf, c = y.astype(float), model._row_weights(y.astype(float), None)
    _, grads = model._mofe_loss_and_grads(params, more, fewer, yf, c)
    h = 1e-6
    for k, P in enumerate(params):
        for idx in [tuple(rng.integers(0, s) for s in P.shape) for _ in range(4)]:
            plus, minus = [q.copy() for q in params], [q.copy() for q in params]
            plus[k][idx] += h
            minus[k][idx] -= h
            lp, _ = model._mofe_loss_and_grads(plus, more, fewer, yf, c)
            lm, _ = model._mofe_loss_and_grads(minus, more, fewer, yf, c)
            assert grads[k][idx] == pytest.approx((lp - lm) / (2 * h), rel=1e-4, abs=1e-7)


def test_f7_keeps_the_mask_and_needs_dropout():
    views, y, ctx = _synthetic(n=1500)
    f7 = make_fusion("F7", epochs=5).fit(views, y, ctx)
    assert isinstance(f7, SimMLMGate)
    assert (f7.view_weights(views, ctx)[np.isnan(views)] == 0.0).all()
    with pytest.raises(ValueError, match="dropout"):
        SimMLMGate(dropout=0.0)


@pytest.mark.parametrize("seed", [0, 1])
def test_f7_reduces_mofe_violations_against_mvaf(seed):
    views, y, ctx = _synthetic()
    test_views, test_y, test_ctx = _synthetic(seed=1)
    fewer = view_dropout(test_views, 0.5, np.random.default_rng(9))

    def held_out_mofe(model):
        lm = row_bce(model.predict_proba(test_views, test_ctx), test_y)
        lf = row_bce(model.predict_proba(fewer, test_ctx), test_y)
        return mofe_term(lm, lf)

    mvaf = make_fusion("MVAF", epochs=20, seed=seed).fit(views, y, ctx)
    f7 = make_fusion("F7", epochs=20, seed=seed).fit(views, y, ctx)
    assert held_out_mofe(f7) < held_out_mofe(mvaf)


def test_masked_softmax_never_overflows_on_masked_entries():
    """A masked entry far above the available ones must not raise an overflow warning."""
    import warnings

    from vaultic.fusion.mvaf import masked_softmax

    g = np.array([[0.0, 1.0, 1e4], [2.0, -1e4, 0.0]])
    mask = np.array([[True, True, False], [True, False, True]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        w = masked_softmax(g, mask)
    e = np.exp([0.0, 1.0])
    assert w[0].tolist() == pytest.approx([e[0] / e.sum(), e[1] / e.sum(), 0.0])
    assert w[1, 1] == 0.0 and w[1].sum() == pytest.approx(1.0)
