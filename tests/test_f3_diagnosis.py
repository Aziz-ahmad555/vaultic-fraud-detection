"""Post-hoc F3 diagnosis helpers (D117) on synthetic data."""

import numpy as np

from vaultic.fusion.f3_diagnosis import (
    CONTEXT,
    INPUTS,
    VIEWS,
    fit_f3,
    psi,
    quantile_map,
    shift_table,
    swap_table,
)


def _rows(n, seed, hist_scale):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.1).astype(int)
    views = np.clip(0.1 + 0.6 * y[:, None] + rng.normal(0, 0.2, (n, len(VIEWS))), 0.01, 0.99)
    views[rng.random((n, len(VIEWS))) < 0.3] = np.nan
    context = rng.normal(0, 1, (n, len(CONTEXT)))
    context[:, CONTEXT.index("ctx_hist_n_past")] = rng.poisson(hist_scale, n)
    return views, context, y


def test_psi_and_quantile_map():
    rng = np.random.default_rng(0)
    a, b = rng.normal(0, 1, 5000), rng.normal(2, 1, 5000)
    assert psi(a, a) < 0.01 < 1.0 < psi(a, b)
    m = quantile_map(np.array([5.0, np.nan, 1.0, 3.0]), a)
    assert np.isnan(m[1]) and m[2] < m[3] < m[0]  # ranks kept, NaN stays NaN
    assert abs(np.nanmean(quantile_map(b, a))) < 0.1  # marginal moved onto the reference


def test_shift_and_swap_find_a_drifted_count():
    gv, gc, gy = _rows(4000, 1, hist_scale=3)
    tv, tc, ty = _rows(4000, 2, hist_scale=30)  # the history count drifts upward
    d = {"gate_views": gv, "gate_context": gc, "gate_y": gy,
         "test_views": tv, "test_context": tc, "test_y": ty}  # fmt: skip
    models = fit_f3(d)
    shifts = {r["input"]: r for r in shift_table(models, d)}
    assert set(shifts) == set(INPUTS)
    hist = shifts["ctx_hist_n_past"]["all rows"]
    assert hist["psi"] > 1.0 and hist["max_test"] > hist["max_gate"]
    swaps = {r["swap"]: r for r in swap_table(models, d)}
    assert {"none (as in the final run)", "ctx_hist_n_past", "all five views"} <= set(swaps)
    assert all(0 <= r["all rows"] <= 1 for r in swaps.values())
