"""E23 subgroup gaps on synthetic data (D121 section 3)."""

import numpy as np
import pandas as pd
import pytest

from vaultic.eval.metrics import pr_auc, recall_at_fpr
from vaultic.stage2.e23 import attribute_groups, e23_tables, global_fpr_threshold


def _synthetic(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.05).astype(int)
    group = np.where(rng.random(n) < 0.5, "A", "B").astype(object)
    group[:300] = "tiny"
    y[:300] = 0
    y[:10] = 1  # 10 frauds: below the 30-fraud flag
    noise = np.where(group == "A", 0.3, 1.0)  # MVAF is better in A
    mvaf = y + rng.normal(0, noise)
    b5 = y + rng.normal(0, 1.0, n)
    return y, {"MVAF": mvaf, "B5": b5}, {"g": group}


def test_global_threshold_gives_the_recall_at_fpr():
    y, s, _ = _synthetic()
    t = global_fpr_threshold(y, s["MVAF"])
    flagged = s["MVAF"] >= t
    assert flagged[y == 0].mean() <= 0.01
    assert flagged[y == 1].mean() == pytest.approx(recall_at_fpr(y, s["MVAF"], 0.01))


def test_groups_gaps_and_paired_differences():
    y, s, groups = _synthetic()
    out = e23_tables(y, s, groups, {"MVAF": 0.5, "B5": 0.5}, n_boot=200)
    g = out["groups"].set_index(["group", "method"])
    a = groups["g"] == "A"
    assert g.loc[("A", "MVAF"), "pr_auc"] == pytest.approx(pr_auc(y[a], s["MVAF"][a]))
    fpr = ((s["B5"] >= 0.5) & (y == 0) & a).sum() / ((y == 0) & a).sum()
    assert g.loc[("A", "B5"), "fpr_at_cost_threshold"] == pytest.approx(fpr)
    assert not g.loc[("tiny", "MVAF"), "reliable"] and g.loc[("A", "MVAF"), "reliable"]
    assert g.loc[("A", "MVAF"), "pr_auc_ci_low"] <= g.loc[("A", "MVAF"), "pr_auc"]
    gap = out["gaps"].query("method == 'MVAF' and metric == 'pr_auc'").iloc[0]
    assert gap["groups"] == 2  # the 10-fraud group is left out of the gaps
    want = abs(pr_auc(y[a], s["MVAF"][a]) - pr_auc(y[~a & (groups["g"] == "B")],
                                                   s["MVAF"][~a & (groups["g"] == "B")]))  # fmt: skip
    assert gap["max_minus_min"] == pytest.approx(want)
    assert gap["max_minus_min_ci_low"] <= gap["max_minus_min"] <= gap["max_minus_min_ci_high"]
    d = out["mvaf_minus_b5"].set_index("group")
    assert d.loc["A", "diff"] > 0 and d.loc["A", "ci_low"] > 0  # MVAF clearly better in A
    assert (d["p_holm"].dropna() >= d["p_value"].dropna()).all()


def test_attribute_groups_follow_the_roadmap_list():
    raw = pd.DataFrame({
        "ProductCD": ["W", "C", "R", "H", "S", "W"],
        "DeviceType": ["desktop", "mobile", None, "desktop", None, "mobile"],
        "P_emaildomain": ["gmail.com", "yahoo.com", None, "rare.org", "gmail.com", "aol.com"],
    })  # fmt: skip
    train = pd.Series(["gmail.com"] * 5 + ["yahoo.com"] * 4 + ["aol.com"] * 3 + ["a.com"] * 2
                      + ["b.com"] * 2 + ["rare.org"])  # fmt: skip
    g = attribute_groups(raw, train, hist_n_past=[0, 3, 0, 1, np.nan, 2])
    assert list(g["DeviceType"]) == ["desktop", "mobile", "missing", "desktop", "missing", "mobile"]
    # rare.org is 6th in the training period: "other"
    assert list(g["P_emaildomain"]) == ["gmail.com", "yahoo.com", "missing", "other",
                                        "gmail.com", "aol.com"]  # fmt: skip
    assert list(g["cold start"]) == ["cold start", "history", "cold start", "history",
                                     "cold start", "history"]  # fmt: skip
