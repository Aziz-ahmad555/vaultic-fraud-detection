"""Scenario suite on the real data (Phase B, D191): every scenario at a fixed seed, run through the
Test Lab engine with the history overlay. Asserts the explicit false-positive budgets of D191
with the D196 thresholds, determinism, the synthetic-row markers and the scenario reports.
Synthetic scenario - not the evaluation. Skipped without the data files, the final run on record
and the cached frozen system (drive E)."""

import numpy as np
import pytest

from vaultic.scenarios import LABEL, SYNTHETIC_ID_START


def _real_available() -> bool:
    from vaultic.live.engine import STAGE2_DEV_SYSTEM, SYSTEM_CACHE
    from vaultic.paths import MERGED_PATH, REPO_ROOT
    from vaultic.stage2.final import E10_RUN

    return (MERGED_PATH.exists() and (REPO_ROOT / E10_RUN / "predictions.parquet").exists()
            and (SYSTEM_CACHE.exists() or STAGE2_DEV_SYSTEM.exists()))  # fmt: skip


real = pytest.mark.skipif(not _real_available(), reason="needs the data files, the final run and "
                          "the cached frozen system (drive E)")  # fmt: skip
SEED, INTENSITY = 0, "low"


@pytest.fixture(scope="module")
def loaded():
    from vaultic.scenarios.run import load_all

    return load_all(log=lambda m: None)


@real
@pytest.mark.parametrize("name", ["card_testing", "account_takeover", "mule_ring", "bust_out",
                                  "mimicry"])  # fmt: skip
def test_scenario_meets_the_fp_budget(loaded, name):
    from vaultic.scenarios.generators import GENERATORS, MULE_DEVICE_NOTE
    from vaultic.scenarios.run import LIVE, STORED, run_scenario

    system, store, stored = loaded
    a = GENERATORS[name](store, seed=SEED, intensity=INTENSITY)
    b = GENERATORS[name](store, seed=SEED, intensity=INTENSITY)
    assert a.rows.equals(b.rows) and np.array_equal(a.uids, b.uids)  # fixed seed -> same rows
    assert (a.rows["TransactionID"] >= SYNTHETIC_ID_START).all()
    res = run_scenario(store, system, stored, a, log=lambda m: None)
    assert res["label"] == LABEL
    rows = res["rows"]
    assert set(rows["source"]) <= {LIVE, STORED}
    assert (rows.loc[rows["synthetic"], "source"] == LIVE).all()
    assert (rows.loc[~rows["synthetic"], "TransactionID"] < SYNTHETIC_ID_START).all()
    primary = res["results"]["D196 (primary)"]
    assert "D195 (sensitivity)" in res["results"]
    assert primary["budget_added_fpr_ok"], primary["added_fpr"]  # D191: <= +0.5 pp over model
    assert primary["budget_rules_only_ok"], primary["rules_only"]["fpr_legit"]  # D191: <= 2%
    for mode in ("rules_only", "model_only", "rules_plus_model"):
        assert "hard_recall_synthetic" in primary[mode]
    if name == "mule_ring":
        assert res["info"]["device_note"] == MULE_DEVICE_NOTE
        assert 2 <= res["info"]["device_rows_in_data"] <= 5
    if name == "bust_out":
        assert "alerted_before_first_cash_out" in primary["rules_plus_model"]
        assert (rows.loc[rows["synthetic"], "label"] == 0).sum() == res["info"]["build_up_rows"]
