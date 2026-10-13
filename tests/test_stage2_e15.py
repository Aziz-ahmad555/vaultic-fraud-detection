"""E15 explanation tests on a small synthetic frozen system (D121 section 1)."""

import numpy as np
import pytest
from stage2_fake import fake_system, synthetic_frame
from xgboost import DMatrix

from vaultic.stage2.e15 import (
    B5Explained,
    MvafExplained,
    deletion_aopc,
    faithfulness,
    global_order,
    order_by,
    paired_mean_test,
    permutation_importance,
    random_orders,
    seed_consistency,
    stability,
    tree_shap,
    view_deletion,
)


@pytest.fixture(scope="module")
def fake():
    system, train, y, ctx = fake_system()
    rows, _ = synthetic_frame(300, seed=5)
    avail = system.availability(rows)
    ctx_rows = ctx[:300]
    return system, rows, ctx_rows, avail, train, y, ctx


def test_tree_shap_adds_up_to_the_margin(fake):
    system = fake[0]
    m, rows = system.b5_models[0], fake[1][system.b5_columns]
    contrib = m.get_booster().predict(DMatrix(rows), pred_contribs=True)
    margin = m.get_booster().predict(DMatrix(rows), output_margin=True)
    assert contrib.sum(axis=1) == pytest.approx(margin, rel=1e-4, abs=1e-4)
    assert tree_shap(m, rows).shape == (len(rows), len(system.b5_columns))


def test_mvaf_attribution_is_gate_weight_times_view_shap(fake):
    system, rows, ctx, avail = fake[:4]
    expl = MvafExplained(system)
    shap = expl.shap_by_view(rows, avail)
    p = system.calibrated(system.raw(rows, avail))
    w = system.gate_weights(p, ctx)
    a = expl.attributions(rows, ctx, avail)
    j = expl.columns.index("b1")  # behavioral only
    assert a[:, j] == pytest.approx(np.nan_to_num(w[:, 1]) * shap["behavioral"][:, 3])
    beh_missing = ~avail[:, 1]
    assert np.all(a[beh_missing, j] == 0)  # a missing view contributes nothing
    per_seed = expl.attributions(rows, ctx, avail, per_seed=True)
    assert len(per_seed) == 5


def test_availability_is_kept_when_features_change(fake):
    system, rows, ctx, avail = fake[:4]
    changed = rows.copy()
    changed["hist_n_past"] = 0.0  # would make the behavioral view missing
    raw = system.raw(changed, avail)
    assert np.array_equal(~np.isnan(raw), avail)


def test_attribution_deletion_beats_random(fake):
    system, rows, ctx, avail = fake[:4]
    expl = MvafExplained(system)
    attr = expl.attributions(rows, ctx, avail)
    a_attr, curve = deletion_aopc(expl, rows, ctx, avail, system.medians, order_by(attr))
    assert curve.shape == (10, len(rows))
    a_rand = np.mean([deletion_aopc(expl, rows, ctx, avail, system.medians, o)[0]
                      for o in random_orders(len(rows), attr.shape[1], draws=3)], axis=0)  # fmt: skip
    t = paired_mean_test(a_attr, a_rand, n_boot=200)
    assert t["diff"] > 0 and t["ci_low"] > 0


def test_faithfulness_report_and_holm(fake):
    system, rows, ctx, avail, train, y, ctx_all = fake
    expl = B5Explained(system)
    imp = permutation_importance(expl, train[:600], None, None, y[:600], repeats=1)
    assert np.argmax(imp) == expl.columns.index("x1")  # the planted signal
    small = rows[:60]
    out = faithfulness(expl, small, None, None, system.medians, imp, n_boot=100, log=lambda m: None)
    h = out["hypotheses"]
    assert set(h) == {"H6a", "H6b"} and all(h[k]["p_holm"] >= h[k]["p_value"] for k in h)
    assert len(out["deletion_curve"]) == 10
    sec = out["secondary_positive_order"]  # D167: declared secondary, own Holm family
    assert sec["label"].startswith("SECONDARY") and len(sec["deletion_curve"]) == 10
    assert set(sec["tests"]) == {"positive - random", "positive - permutation"}
    assert all(t["p_holm"] >= t["p_value"] for t in sec["tests"].values())
    assert global_order(imp, 3).shape == (3, len(imp))


def test_mvaf_permutation_cache_equals_full_rescoring(fake):
    system, rows, ctx, avail = fake[:4]
    expl = MvafExplained(system)
    raw = expl.raw_cache(rows, avail)
    changed = rows.copy()
    changed["b1"] = changed["b1"].to_numpy()[::-1]
    assert expl.risk_changed(changed, ctx, avail, raw, "b1") == pytest.approx(
        expl.risk(changed, ctx, avail)
    )


def test_stability_and_seed_consistency_identities(fake):
    system, rows, ctx, avail = fake[:4]
    expl = MvafExplained(system)
    a = expl.attributions(rows, ctx, avail)
    st = stability(expl, a, {1.0: a.copy()})
    assert st["x1.0"]["top3_jaccard"]["mean"] == 1.0 and st["x1.0"]["top1_changed"]["mean"] == 0.0
    assert seed_consistency([a, a, a])["mean"] == 1.0
    assert seed_consistency([a, a])["pairs"] == 1


def test_view_deletion_needs_enough_views(fake):
    system, rows, ctx, avail = fake[:4]
    out = view_deletion(system, rows, ctx, avail)
    assert set(out) >= {"top-1 view", "top-2 views", "random view"}
    n_av = avail.sum(axis=1)
    assert out["top-2 views"]["n"] == int((n_av >= 3).sum())


def test_refit_checks_require_every_row():
    import pandas as pd

    from vaultic.fusion.final_eval import refit_check
    from vaultic.stage2.system import refit_complete

    stored = pd.Series([0.1, 0.2, 0.3], index=[1, 2, 3])
    assert refit_complete(refit_check(stored, [1, 2, 3], [0.1, 0.2, 0.3]), 3)
    # one row missing from the stored run: the remaining rows match, but the check fails (D173)
    assert not refit_complete(refit_check(stored, [1, 2, 4], [0.1, 0.2, 0.9]), 3)
    assert not refit_complete(refit_check(stored, [1, 2, 3], [0.1, 0.2, float("nan")]), 3)
    assert not refit_complete(refit_check(stored, [1, 2, 3], [0.1, 0.2, 0.31]), 3)
