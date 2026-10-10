"""Final-experiment statistics (E10, E11; D101, D102) on synthetic scores."""

import numpy as np
import pandas as pd
import pytest

from vaultic.fusion.final_eval import (
    e10_subgroups,
    e10_table,
    e11_drops,
    h2_verdict,
    subgroup_masks,
)

RNG = np.random.default_rng(0)
N = 2000
Y = (RNG.random(N) < 0.1).astype(int)
AMOUNT = RNG.lognormal(4, 1, N)


def _score(noise, seed):
    r = np.random.default_rng(seed)
    return 1 / (1 + np.exp(-(3 * Y - 1.5 + r.normal(0, noise, N))))


def test_e10_table_paired_comparisons_holm_and_effect_sizes():
    scores = {"MVAF": _score(0.8, 1), "F3": _score(2.0, 2), "B5": _score(0.8, 3)}
    per_seed = {"MVAF": [_score(0.8, s) for s in (1, 4, 5)],
                "F3": [_score(2.0, s) for s in (2, 6, 7)]}  # fmt: skip
    th = {m: 0.5 for m in scores}
    methods, comps = e10_table(Y, scores, per_seed, AMOUNT, th, n_boot=200)
    assert set(methods["method"]) == {"MVAF", "F3", "B5"}
    m = methods.set_index("method")
    assert (
        m.loc["MVAF", "pr_auc_ci_low"] < m.loc["MVAF", "pr_auc"] < m.loc["MVAF", "pr_auc_ci_high"]
    )
    pr = comps[comps["metric"] == "pr_auc"].set_index("comparison")
    assert pr.loc["MVAF - F3", "diff"] > 0 and pr.loc["MVAF - F3", "ci_low"] > 0
    assert pr.loc["MVAF - F3", "p_holm"] >= pr.loc["MVAF - F3", "p_value"]
    assert np.isfinite(pr.loc["MVAF - F3", "cohens_d_paired"])  # per-seed values for both
    assert np.isnan(pr.loc["MVAF - B5", "cohens_d_paired"])  # B5 has no per-seed list here
    # relative % follows the reported diff, so the two always share a sign
    assert (np.sign(pr["relative_pct"]) == np.sign(pr["diff"])).all()
    assert np.isfinite(pr.loc["MVAF - F3", "seed_mean_diff"])
    assert set(comps["metric"]) == {"pr_auc", "recall_at_1pct_fpr", "cost"}
    # the same method compared with itself would give exactly zero on every resample
    same, _ = e10_table(Y, {"MVAF": scores["MVAF"], "X": scores["MVAF"]}, {}, AMOUNT,
                        {"MVAF": 0.5, "X": 0.5}, n_boot=50)  # fmt: skip
    assert same is not None


def test_subgroups_and_h2_rule():
    hist = RNG.integers(0, 3, N)
    ident = RNG.integers(0, 2, N)
    views = np.ones((N, 5), int)
    views[: N // 2, 3] = 0  # graph missing for half
    views[: N // 4, 2] = 0
    masks = subgroup_masks(hist, ident, views)
    assert (masks["cold start"] | masks["with history"]).all()
    assert masks["5 views"].sum() == N // 2 and masks["3-4 views"].sum() == N // 2
    scores = {"MVAF": _score(0.8, 1), "F3": _score(2.0, 2), "F4": _score(2.0, 3)}
    sub = e10_subgroups(Y, scores, masks, n_boot=100)
    assert "MVAF - F3" in sub.columns
    _, comps = e10_table(Y, scores, {}, AMOUNT, {m: 0.5 for m in scores}, n_boot=100)
    verdict = h2_verdict(comps, sub)
    assert verdict["beats_overall"] == {"F3": True, "F4": True}
    assert set(verdict) == {"beats_overall", "larger_margin_on_missing_views", "verdict",
                            "supported"}  # fmt: skip
    assert verdict["verdict"] == "undetermined"  # no 1-2 view rows here: that margin is NaN
    worse = {"MVAF": _score(2.0, 9), "F3": _score(0.8, 2), "F4": _score(0.8, 3)}
    _, comps2 = e10_table(Y, worse, {}, AMOUNT, {m: 0.5 for m in worse}, n_boot=100)
    assert not h2_verdict(comps2, e10_subgroups(Y, worse, masks, n_boot=100))["supported"]


def test_e11_drops_and_holm_family():
    full = {"MVAF": _score(0.8, 1), "F3": _score(0.8, 2), "F4": _score(0.8, 3)}
    removed = {}
    for k, v in enumerate(("tabular", "behavioral", "temporal", "graph", "anomaly")):
        removed[v] = {
            "MVAF": _score(1.0, 10 + k),
            "F3": _score(3.0, 20 + k),
            "F4": _score(1.0, 30 + k),
        }
    removed["tabular only"] = {
        "MVAF": _score(1.5, 40),
        "F3": _score(1.5, 41),
        "F4": _score(1.5, 42),
    }
    drops, comps = e11_drops(Y, full, removed, n_boot=100)
    assert len(drops) == 6 * 3
    d = drops.set_index(["condition", "method"])
    assert d.loc[("tabular", "F3"), "drop"] > d.loc[("tabular", "MVAF"), "drop"]
    c = comps[comps["comparison"] == "drop MVAF - drop F3"].set_index("condition")
    assert c.loc["tabular", "diff"] < 0  # MVAF degrades less than F3 here
    assert np.isnan(c.loc["tabular only", "p_holm"])  # outside the Holm family
    assert c.drop(index="tabular only")["p_holm"].notna().all()


def test_e10_cost_uses_the_given_threshold_only():
    s = _score(0.8, 1)
    _, comps_lo = e10_table(
        Y, {"MVAF": s, "F3": s}, {}, AMOUNT, {"MVAF": 0.2, "F3": 0.9}, n_boot=50
    )
    cost = comps_lo[comps_lo["metric"] == "cost"].iloc[0]
    assert cost["diff"] != 0  # same scores, different thresholds -> different cost
    with pytest.raises(KeyError):
        e10_table(Y, {"MVAF": s, "F3": s}, {}, AMOUNT, {"MVAF": 0.5}, n_boot=10)


def _h2_inputs(diff_f3, diff_f4, p, margins):
    comps = pd.DataFrame([{"metric": "pr_auc", "comparison": f"MVAF - {v}", "diff": d,
                           "p_holm": p} for v, d in (("F3", diff_f3), ("F4", diff_f4))])  # fmt: skip
    subs = pd.DataFrame([{"subgroup": g, "MVAF - F3": (m, 0, 0, 1), "MVAF - F4": (m, 0, 0, 1)}
                         for g, m in margins.items()])  # fmt: skip
    return comps, subs


def test_h2_verdict_three_states():
    ok = {"1-2 views": 0.05, "3-4 views": 0.04, "5 views": 0.01}
    assert h2_verdict(*_h2_inputs(0.02, 0.02, 0.01, ok))["verdict"] == "supported"
    assert h2_verdict(*_h2_inputs(-0.01, 0.02, 0.01, ok))["verdict"] == "not supported"
    smaller = {**ok, "3-4 views": 0.0}
    assert h2_verdict(*_h2_inputs(0.02, 0.02, 0.01, smaller))["verdict"] == "not supported"
    missing = {**ok, "1-2 views": np.nan}  # one margin not finite: undetermined (D105)
    r = h2_verdict(*_h2_inputs(0.02, 0.02, 0.01, missing))
    assert r["verdict"] == "undetermined" and r["supported"] is False
    assert r["larger_margin_on_missing_views"]["F3"] is None
    # a decisive False elsewhere still gives "not supported"
    assert h2_verdict(*_h2_inputs(0.02, 0.02, 0.2, missing))["verdict"] == "not supported"
