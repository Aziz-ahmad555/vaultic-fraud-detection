"""E12 trust-layer evaluation (D101, D102) on synthetic, time-ordered data."""

import numpy as np

from vaultic.trust.final_trust import phase8_gate, trust_report


def _rows(n, seed, day0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.08).astype(int)
    p = 1 / (1 + np.exp(-(2.5 * y - 2.5 + rng.normal(0, 1.0, n))))
    day = np.sort(rng.integers(day0, day0 + 20, n))
    return {"y": y, "p": p, "amount": rng.lognormal(4, 1, n), "day": day,
            "d": rng.random(n) * 0.3}  # fmt: skip


def test_trust_report_sections_and_gate():
    cal, ev = _rows(3000, 0, 100), _rows(3000, 1, 130)
    rep = trust_report(cal, ev, label_delay_days=5, sweep=False, n_boot=50)
    assert set(rep) >= {"calibrator", "calibration", "conformal", "adaptive", "decision", "routing"}
    assert rep["calibrator"]["fallback"] is False  # > 100 frauds in the calibration rows
    for t in ("0.90", "0.95"):
        c = rep["conformal"][t]
        assert 0 <= c["coverage"] <= 1 and {"coverage_legit", "coverage_fraud"} <= set(c)
        assert abs(c["coverage"] - float(t)) < 0.05  # exchangeable synthetic data: near target
        blocks = rep["adaptive"][t]["blocks"]
        assert len(blocks) == 3 and blocks[0]["first_day"] == 130  # 20 days in 7-day blocks
    pol = {(r["policy"], r["K"]) for r in rep["routing"]["policies"]}
    assert ("R4", 50) in pol and ("R1", 500) in pol
    assert {h["comparison"] for h in rep["routing"]["h5"]} == {"R3 - R1", "R4 - R1"}
    assert rep["decision"]["default"]["evaluation_cost"] > 0
    gate = phase8_gate({"MVAF": rep})
    assert set(gate["per_method"]["MVAF"]) == {"0.90", "0.95"}


def test_no_disagreement_means_no_r4_and_small_slices_use_platt():
    cal, ev = _rows(600, 2, 100), _rows(600, 3, 130)
    cal["d"] = ev["d"] = None
    rep = trust_report(cal, ev, label_delay_days=5, sweep=True, n_boot=20)
    assert rep["calibrator"]["fallback"] is True  # < 100 frauds: Platt (D83)
    assert all(r["policy"] != "R4" for r in rep["routing"]["policies"])
    assert len(rep["decision"]["sensitivity"]) == 5 * 3 * 3 * 3
