"""Paired gain by history group and the label-dependency finder (items 2-6, D65)."""

import numpy as np
import pandas as pd
import pytest

from vaultic.reports.gain import gain_table, group_masks, n_past_for
from vaultic.reports.label_features import label_dependent_columns

D = 86_400


def test_group_masks_partition_by_history():
    n = np.array([0, 1, 4, 5, 19, 20, 100])
    g = group_masks(n, "gru")
    assert g["cold start"].tolist() == [1, 0, 0, 0, 0, 0, 0]
    assert g["1-4 past"].sum() == 2 and g["5-19 past"].sum() == 2 and g["20+ past"].sum() == 2
    h = group_masks(n, "history")
    assert h[">= 5 past"].sum() == 4 and h["all"].all()


def _preds(ids, y, scores):
    cols = {f"score_seed{i}": s for i, s in enumerate(scores)}
    return pd.DataFrame({"TransactionID": ids, "label": y, **cols})


def test_gain_is_new_minus_base_and_marks_one_class_groups():
    rng = np.random.default_rng(0)
    n = 400
    y = (rng.random(n) < 0.2).astype(int)
    ids = np.arange(n)
    base = _preds(ids, y, [rng.random(n), rng.random(n)])  # random scores
    new = _preds(ids, y, [y + rng.normal(0, 0.3, n)] * 2)  # informative scores
    n_past = np.where(ids < 380, 10, 0)
    table = gain_table(base, new, n_past, "history")
    row = table.set_index("group").loc["all"]
    assert row["gain"] > 0.3 and row["ci_low"] > 0 and row["new PR-AUC"] > row["base PR-AUC"]
    assert table.set_index("group").loc["1-4 past", "rows"] == 0
    assert "one class" in str(table.set_index("group").loc["1-4 past", "note"])
    with pytest.raises(ValueError, match="different"):
        gain_table(base, new.iloc[::-1].reset_index(drop=True), n_past)


def test_n_past_join_by_id():
    preds = pd.DataFrame({"TransactionID": [3, 1]})
    feats = pd.DataFrame({"TransactionID": [1, 2, 3], "uid_n_past": [0.0, 5.0, 7.0]})
    assert n_past_for(preds, feats).tolist() == [7.0, 0.0]
    with pytest.raises(ValueError):
        n_past_for(pd.DataFrame({"TransactionID": [9]}), feats)


def test_label_dependent_columns_found_by_perturbation():
    df = pd.DataFrame(
        {
            "TransactionID": np.arange(8),
            "TransactionDT": np.array([0, 1, 2, 3, 5, 6, 7, 9]) * D,
            "day": [0, 1, 2, 3, 5, 6, 7, 9],
            "TransactionAmt": [10.0, 20, 30, 40, 50, 60, 70, 80],
            "isFraud": [1, 0, 1, 0, 0, 1, 0, 0],
            "card1": [1.0] * 8,
            "addr1": [5.0] * 8,
            "P_emaildomain": ["a.com"] * 8,
        }
    )
    uid = pd.Series(["u"] * 8)
    found = label_dependent_columns(df, uid, delay_days=1, longer_delay_days=3)
    assert set(found["values"]) == {"uid_fraud_known", "uid_fraud_rate_known"}
    assert found["timing"] == ["uid_n_labels_known"]


def test_emerging_beats_random_and_slugs():
    """D94: 'beats random' = PR-AUC CI lower bound above the base rate."""
    from vaultic.reports.emerging import GROUPS, beats_random, slug

    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.05).astype(int)
    assert beats_random(y, y + rng.normal(0, 0.5, 2000))["beats_random"]
    assert not beats_random(y, rng.random(2000))["beats_random"]
    assert (
        slug("ProductCD", "R") == "R"
        and slug("card4", "american express") == "card4-american-express"
    )
    assert len(GROUPS) == 9


def test_emerging_configs_drop_one_groups_frauds(tmp_path, monkeypatch):
    import yaml

    import vaultic.reports.emerging as em

    monkeypatch.setattr(em, "CONFIG_DIR", tmp_path)
    written = em.write_configs()
    assert len(written) == 18
    cfg = yaml.safe_load((tmp_path / "EXP-108-holdout-card4-american-express.yaml").read_text())
    assert cfg["train_drop_fraud"] == {"card4": "american express"}
    assert cfg["extends"] == "EXP-108-frozen.yaml"


def test_gain_table_full_kind_adds_extra_subgroups():
    rng = np.random.default_rng(1)
    n = 500
    y = (rng.random(n) < 0.2).astype(int)
    ids = np.arange(n)
    base = _preds(ids, y, [rng.random(n)])
    new = _preds(ids, y, [y + rng.normal(0, 0.4, n)])
    ident = rng.random(n) < 0.3
    t = gain_table(base, new, np.full(n, 3), "full", {"has_identity yes": ident,
                                                      "has_identity no": ~ident})  # fmt: skip
    groups = t["group"].tolist()
    assert groups[:4] == ["all", ">= 5 past", "1-4 past", "cold start"]
    assert groups[-2:] == ["has_identity yes", "has_identity no"]
    assert t.set_index("group").loc["has_identity yes", "rows"] == int(ident.sum())
