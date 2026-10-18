"""Test Lab live scoring (src/vaultic/live): batch validation on a small fake store, and, when the
real data and the final run are present, online-feature parity, custom-transaction uid resolution
and batch scoring against the stored final-run scores (D182-D186)."""

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from vaultic.live.batch import MAX_ROWS, BatchError, parse, template_csv


def _fake_store():
    df = pd.DataFrame({
        "TransactionID": pd.Series([1, 2], dtype="int32"),
        "isFraud": pd.Series([0, 1], dtype="int8"),
        "TransactionDT": pd.Series([151 * 86400, 152 * 86400], dtype="int32"),
        "TransactionAmt": pd.Series([10.0, 20.0], dtype="float32"),
        "ProductCD": pd.Categorical(["W", "C"], categories=["C", "H", "R", "S", "W"]),
        "card1": pd.Series([1000.0, 2000.0], dtype="float32"),
        "addr1": pd.Series([np.nan, 300.0], dtype="float32"),
        "D1": pd.Series([0.0, 5.0], dtype="float32"),
        "DeviceType": pd.Categorical([None, "mobile"], categories=["desktop", "mobile"]),
        "has_identity": pd.Series([0, 1], dtype="int8"),
        "day": pd.Series([151, 152], dtype="int16"),
    })  # fmt: skip
    return SimpleNamespace(df=df, splits=SimpleNamespace(test=SimpleNamespace(first=151, last=182)))


GOOD = (
    "TransactionID,TransactionDT,TransactionAmt,ProductCD,card1,isFraud\n5,13046400,12.5,W,1500,1\n"
)


def test_template_has_the_ieee_columns_but_not_the_derived_ones():
    head = template_csv(_fake_store()).strip().split(",")
    assert "TransactionAmt" in head and "isFraud" in head
    assert "day" not in head and "has_identity" not in head


def test_a_good_file_parses_with_the_stored_dtypes():
    rows = parse(_fake_store(), GOOD)
    assert rows.attrs["has_label"] and rows["day"].iat[0] == 151
    assert str(rows["ProductCD"].dtype) == "category" and np.isnan(rows["addr1"].iat[0])


@pytest.mark.parametrize("text,msg", [
    ("", "not a readable CSV"),
    ("TransactionID,TransactionDT,TransactionAmt,ProductCD,card1,bogus\n5,13046400,1,W,1,2\n", "unknown columns"),
    ("TransactionID,TransactionDT,TransactionAmt,card1\n5,13046400,1,1\n", "missing required"),
    ("TransactionID,TransactionDT,TransactionAmt,ProductCD,card1\n5,100,1,W,1\n", "outside the test period"),
    ("TransactionID,TransactionDT,TransactionAmt,ProductCD,card1\n5,13046400,1,Z,1\n", "never seen"),
    ("TransactionID,TransactionDT,TransactionAmt,ProductCD,card1\n5,13046400,abc,W,1\n", "non-numeric"),
    ("TransactionID,TransactionDT,TransactionAmt,ProductCD,card1\n5,13046400,,W,1\n", "is empty"),
    ("TransactionID,TransactionDT,TransactionAmt,ProductCD,card1,isFraud\n5,13046400,1,W,1,3\n", "0 or 1"),
    ("TransactionID,TransactionDT,TransactionAmt,ProductCD,card1,isFraud\n5,13046400,1,W,1,1\n6,13046400,1,W,1,\n", r"isFraud is empty in rows \[3\]"),
])  # fmt: skip
def test_bad_files_are_refused_with_a_clear_reason(text, msg):
    with pytest.raises(BatchError, match=msg):
        parse(_fake_store(), text)


def test_too_many_rows_are_refused():
    body = "".join(f"{i},13046400,1,W,1\n" for i in range(MAX_ROWS + 1))
    with pytest.raises(BatchError, match="at most"):
        parse(_fake_store(), "TransactionID,TransactionDT,TransactionAmt,ProductCD,card1\n" + body)


# ---- real data (skipped without the data files and the final run on record) -----------------


def _real_available() -> bool:
    from vaultic.live.engine import STAGE2_DEV_SYSTEM, SYSTEM_CACHE
    from vaultic.paths import MERGED_PATH, REPO_ROOT

    run = REPO_ROOT / "experiments/runs/EXP-200-final/20261011-002400-202994/predictions.parquet"
    return (
        MERGED_PATH.exists()
        and run.exists()
        and (SYSTEM_CACHE.exists() or STAGE2_DEV_SYSTEM.exists())
    )


real = pytest.mark.skipif(not _real_available(), reason="needs the data files, the final run and "
                          "the cached frozen system (drive E)")  # fmt: skip


@pytest.fixture(scope="module")
def engine():
    from vaultic.live import engine as eng
    from vaultic.live.store import build_store
    from vaultic.paths import REPO_ROOT
    from vaultic.stage2.final import E10_RUN

    store = build_store(log=lambda m: None, days=[151, 152])  # two test days keep it quick
    stored = pd.read_parquet(REPO_ROOT / E10_RUN / "predictions.parquet").set_index("TransactionID")
    return eng.LiveEngine(store, eng.load_system(log=lambda m: None), stored, 0.8812372071871475)


@real
def test_online_features_and_scores_equal_the_frozen_run_on_a_sample(engine):
    """The parity gate on a sample of days 151-152 (the snapshots this fixture built); the full
    2,000-row gate runs at Test Lab startup (live/parity.cached_parity)."""
    from vaultic.live.parity import FEATURE_TOLERANCE, SCORE_TOLERANCE, run_parity

    day = engine.store.df["day"].to_numpy()
    pool = np.flatnonzero((day >= 151) & (day <= 152))
    n = int(os.environ.get("VAULTIC_TESTLAB_PARITY_N", "60"))
    out = run_parity(engine, n=n, log=None, pool=pool)
    assert out["passed"], out
    assert out["max_feature_diff"] <= FEATURE_TOLERANCE and out["n"] == n
    assert max(out["max_score_diff"].values()) <= SCORE_TOLERANCE


@real
def test_custom_transaction_for_an_existing_customer_resolves_to_that_customer(engine):
    from vaultic.live.custom import build_custom
    from vaultic.live.store import uid_key_strings

    store = engine.store
    day = store.df["day"].to_numpy()
    # customers with a transaction before day 152 and one on day 152 (history exists)
    cand = np.flatnonzero(day == 152)
    rng = np.random.default_rng(3)
    checked = 0
    for p in rng.choice(cand, 50, replace=False):
        u = int(store.uid[p])
        if not len(store.before(store.uid_rows[u], 152 * 86400 + 86000)):
            continue
        spec = {"customer": u, "day": 152, "seconds": 86000, "amount": 42.0, "product": "W",
                "card4": "visa", "card6": "debit", "email": "gmail.com"}  # fmt: skip
        row, uid, notes = build_custom(store, spec)
        assert uid == u and store.key_to_uid[uid_key_strings(row)[0]] == u
        assert row["TransactionAmt"].iat[0] == 42.0 and np.isnan(row["D2"].iat[0])
        checked += 1
        if checked == 10:
            break
    assert checked >= 5
    new_row, new_uid, _ = build_custom(store, {"customer": "new", "day": 152, "amount": 5,
                                               "product": "C"})  # fmt: skip
    assert new_uid not in store.uid_rows and new_row["D1"].iat[0] == 0


@real
def test_batch_csv_of_real_rows_reproduces_the_stored_scores(engine):
    from vaultic.live.batch import score_batch, template_columns

    store = engine.store
    pos = np.flatnonzero(store.df["day"].to_numpy() == 151)[:8]
    text = store.df.iloc[pos][template_columns(store)].to_csv(index=False)
    table, metrics = score_batch(engine, parse(store, text))
    ref = engine.stored.loc[table["TransactionID"], "score_MVAF"].to_numpy()
    assert np.max(np.abs(table["risk"].to_numpy() - ref)) <= 1e-6
    assert metrics["has_label"] and sum(metrics["confusion_e12_alert"].values()) == len(pos)


def test_provenance_marks_stored_edited_and_new_rows():
    from vaultic.live.batch import EDITED, NEW, STORED, provenance

    store = _fake_store()
    head = "TransactionID,TransactionDT,TransactionAmt,ProductCD,card1\n"
    same = "1,13046400,10,W,1000\n"  # stored row 1, uploaded fields equal (addr1 left out)
    edited = "2,13132800,25,C,2000\n"  # stored row 2 with a different amount
    new = "99,13046400,10,W,1000\n"
    rows = parse(store, head + same + edited + new)
    assert provenance(store, rows) == [STORED, EDITED, NEW]
    relabelled = parse(store, head.strip() + ",isFraud\n" + "1,13046400,10,W,1000,1\n")
    assert provenance(store, relabelled) == [
        EDITED
    ]  # stored label is 0: a changed label is an edit


def test_parity_fails_without_the_stored_scores():
    from vaultic.live.parity import run_parity

    out = run_parity(SimpleNamespace(stored=None, store=None, system=None), log=None)
    assert out["passed"] is False and "not available" in out["reason"]


def test_cache_root_follows_the_environment(monkeypatch, tmp_path):
    from vaultic.live.store import DEFAULT_CACHE_ROOT, cache_root

    monkeypatch.delenv("VAULTIC_CACHE_DIR", raising=False)
    assert cache_root() == Path(DEFAULT_CACHE_ROOT)
    monkeypatch.setenv("VAULTIC_CACHE_DIR", str(tmp_path))
    assert cache_root() == tmp_path


def test_parity_key_changes_with_the_stored_predictions_and_final_results(tmp_path):
    from vaultic.live.parity import cache_key

    pred, final = tmp_path / "predictions.parquet", tmp_path / "final_results.json"
    pred.write_bytes(b"scores v1")
    final.write_text("{}")
    k1 = cache_key(None, [pred, final])
    pred.write_bytes(b"scores v2")
    k2 = cache_key(None, [pred, final])
    final.write_text('{"meta": 1}')
    k3 = cache_key(None, [pred, final])
    assert len({k1, k2, k3}) == 3
