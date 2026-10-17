"""E15 counterfactuals on a small synthetic frozen system (D158 (2), D159, D163)."""

import numpy as np
import pandas as pd
import pytest
from stage2_fake import fake_system, synthetic_frame

from vaultic.stage2.amount import AmountHistory
from vaultic.stage2.counterfactuals import (
    Problem,
    ProductCodes,
    ProductHistory,
    apply_change,
    choose_setting,
    evaluate_alert,
    grid_settings,
)
from vaultic.trust.decision import STEP_UP

PRODUCTS = ["W", "C", "R"]


@pytest.fixture(scope="module")
def setup():
    system, train, y, ctx = fake_system()
    n = 200
    frame, _ = synthetic_frame(n, seed=7)
    rng = np.random.default_rng(7)
    df = pd.DataFrame({"TransactionDT": np.sort(rng.integers(0, 10**6, n)),
                       "TransactionAmt": frame["TransactionAmt"].to_numpy(float),
                       "ProductCD": [PRODUCTS[int(c)] for c in frame["ProductCD"]]})  # fmt: skip
    uid = frame["uid"].to_numpy()
    frame["new_product"] = np.float32(0)
    rows = np.arange(n)
    codes = ProductCodes(df, frame, encoder=None)
    hist, prods = AmountHistory(df, uid, rows), ProductHistory(df, uid, rows)
    return system, df, frame, ctx[:n].copy(), system.availability(frame), hist, prods, codes


def test_product_change_recomputes_its_dependents(setup):
    system, df, frame, ctx, avail, hist, prods, codes = setup
    from vaultic.views.orchestrate import CONTEXT

    i = 5
    new = "R" if df.at[i, "ProductCD"] != "R" else "W"
    f, c = apply_change(system, frame, ctx, hist, prods, codes, [df.at[i, "TransactionAmt"]],
                        [new], np.array([i]))  # fmt: skip
    assert f.at[0, "ProductCD"] == codes.design[new]
    assert f.at[0, "new_product"] == float(new not in prods.seen[i])
    assert c[0, CONTEXT.index("ctx_product")] == codes.encoded[new]
    # unchanged inputs give back the original row
    f0, _ = apply_change(system, frame, ctx, hist, prods, codes, [df.at[i, "TransactionAmt"]],
                         [df.at[i, "ProductCD"]], np.array([i]))  # fmt: skip
    for col in ("TransactionAmt", "ProductCD", "x1", "hist_n_past"):
        assert f0.at[0, col] == pytest.approx(frame.at[i, col])


def test_validity_is_rechecked_end_to_end(setup):
    system, df, frame, ctx, avail, hist, prods, codes = setup
    pb = Problem(system, "MVAF", df, frame, ctx, avail, hist, prods, codes)
    alerts = np.flatnonzero(pb.action >= STEP_UP)
    assert len(alerts), "the synthetic set needs MVAF alerts"
    # E12 also alerts on an uncertain conformal set or high view disagreement, whatever the
    # amount; a zero amount clears only alerts raised by the expected loss (D159)
    zero = pb.actions(alerts, np.zeros(len(alerts)), pb.product[alerts])
    assert (zero < STEP_UP).any() and (pb.action[alerts] >= STEP_UP).all()
    i = int(alerts[np.flatnonzero(zero < STEP_UP)[0]])
    cands = pd.DataFrame({"TransactionAmt": [0.0, pb.amount[i]], "ProductCD": [pb.product[i]] * 2})
    out = evaluate_alert(pb, i, cands, total=4, mad=10.0)
    assert out["validity"] == pytest.approx(0.25)  # 1 valid of 4 requested
    assert out["any_valid"] == 1.0 and out["returned"] == 2
    assert out["sparsity"] == pytest.approx(0.5)
    empty = evaluate_alert(pb, i, cands.iloc[:0], total=3, mad=10.0)
    assert empty["validity"] == 0.0  # no candidate counts as invalid


def test_selection_rule():
    a, b, c = ({"total_cfs": 3, "sample_size": 1000, "amount_range": "below"},
               {"total_cfs": 5, "sample_size": 1000, "amount_range": "below"},
               {"total_cfs": 3, "sample_size": 5000, "amount_range": "p1-p99"})  # fmt: skip

    def res(v, prox):
        return pd.DataFrame({"validity": v, "proximity": prox, "sparsity": [1.0] * len(v)})

    pick = choose_setting([(a, res([0.90, 0.90], [2.0, 2.0])), (b, res([0.95, 0.95], [3.0, 3.0])),
                           (c, res([0.945, 0.945], [1.0, 1.0]))])  # fmt: skip
    assert pick == c  # within 0.01 of the best validity, lower proximity wins
    assert len(grid_settings()) == 8


def test_dice_search_on_synthetic_alerts(setup):
    pytest.importorskip("dice_ml")
    from vaultic.stage2.counterfactuals import run_setting

    system, df, frame, ctx, avail, hist, prods, codes = setup
    pb = Problem(system, "MVAF", df, frame, ctx, avail, hist, prods, codes)
    alerts = np.flatnonzero(pb.action >= STEP_UP)[:3]
    sub = Problem(system, "MVAF", df.iloc[alerts].reset_index(drop=True),
                  frame.iloc[alerts].reset_index(drop=True), ctx[alerts], avail[alerts],
                  _subset(hist, alerts), _subset_p(prods, alerts), codes)  # fmt: skip
    train = pd.DataFrame({"TransactionAmt": df["TransactionAmt"], "ProductCD": df["ProductCD"],
                          "alert": 0})  # fmt: skip
    train.loc[:20, "alert"] = 1
    setting = {"total_cfs": 3, "sample_size": 300, "amount_range": "below"}
    out = run_setting(sub, train, setting, mad=10.0, train_amounts=df["TransactionAmt"].to_numpy())
    assert len(out) == len(alerts)
    assert out["validity"].between(0, 1).all() and out["actionable_share"].dropna().eq(1).all()


def _subset(hist, idx):
    h = AmountHistory.__new__(AmountHistory)
    h.rows, h.past = hist.rows[idx], [hist.past[i] for i in idx]
    return h


def _subset_p(prods, idx):
    p = ProductHistory.__new__(ProductHistory)
    p.seen = [prods.seen[i] for i in idx]
    return p
