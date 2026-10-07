import numpy as np
import pandas as pd
import pytest

from vaultic.data.load import SECONDS_PER_DAY
from vaultic.data.splits import (
    SPLITS_PATH,
    DayRange,
    label_matured,
    load_splits,
    matured_labels,
)


def test_repo_splits_file_is_valid_and_ordered():
    s = load_splits(SPLITS_PATH)
    assert (s.train.first, s.train.last) == (1, 120)
    assert (s.validation.first, s.validation.last) == (128, 150)
    assert (s.test.first, s.test.last) == (151, 182)
    assert s.validation.first - s.train.last - 1 == 7  # 7-day gap
    assert s.label_delay_days == 30
    assert s.uid_variant in {"uid", "uid2", "uid_card"}


def test_assign_covers_boundaries_and_gap():
    s = load_splits(SPLITS_PATH)
    days = np.array([1, 120, 121, 127, 128, 150, 151, 182])
    assert s.assign(days).tolist() == [
        "train",
        "train",
        "unused",
        "unused",
        "validation",
        "validation",
        "test",
        "test",
    ]


def test_every_day_belongs_to_at_most_one_split():
    s = load_splits(SPLITS_PATH)
    days = np.arange(1, 183)
    membership = sum(
        getattr(s, p).contains(days).astype(int) for p in ("train", "validation", "test")
    )
    assert membership.max() == 1


def test_overlapping_splits_are_rejected(tmp_path):
    bad = tmp_path / "splits.yaml"
    bad.write_text(
        "uid_variant: uid\nlabel_delay_days: 30\n"
        "fixed: {train: [1, 130], validation: [128, 150], test: [151, 182]}\n"
    )
    with pytest.raises(ValueError, match="validation must start after train"):
        load_splits(bad)


def test_rolling_fold_must_test_after_training(tmp_path):
    bad = tmp_path / "splits.yaml"
    bad.write_text(
        "uid_variant: uid\nlabel_delay_days: 30\n"
        "fixed: {train: [1, 120], validation: [128, 150], test: [151, 182]}\n"
        "rolling: [{train: [1, 100], test: [91, 120]}]\n"
    )
    with pytest.raises(ValueError, match="rolling fold"):
        load_splits(bad)


def test_empty_day_range_is_rejected():
    with pytest.raises(ValueError):
        DayRange(10, 5)


def test_label_delay_boundary_is_inclusive():
    t = 100 * SECONDS_PER_DAY
    assert label_matured(t, t + 30 * SECONDS_PER_DAY, 30)  # exactly L later: known
    assert not label_matured(t, t + 30 * SECONDS_PER_DAY - 1, 30)  # one second short
    assert label_matured(t, t, 0)  # no delay: known immediately


def test_matured_labels_hides_recent_labels():
    df = pd.DataFrame(
        {
            "TransactionDT": [0, 10 * SECONDS_PER_DAY, 25 * SECONDS_PER_DAY],
            "isFraud": [1, 0, 1],
        }
    )
    labels = matured_labels(df, as_of_time=40 * SECONDS_PER_DAY, delay_days=30)
    assert labels.iloc[0] == 1 and labels.iloc[1] == 0  # 30+ days old
    assert np.isnan(labels.iloc[2])  # 15 days old: not known yet
