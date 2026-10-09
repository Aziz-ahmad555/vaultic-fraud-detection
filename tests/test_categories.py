"""Shared category codes fitted on the training period (D37)."""

import numpy as np
import pandas as pd

from vaultic.data.splits import DayRange
from vaultic.features.categories import MISSING, UNKNOWN, CategoryEncoder, fit_on_training_period


def _frame():
    return pd.DataFrame(
        {
            "day": [1, 1, 2, 2, 3, 10, 11, 12],
            "ProductCD": pd.Categorical(["W", "C", "W", None, "H", "R", "W", "S"]),
            "card4": ["visa", "visa", "mastercard", np.nan, "visa", "discover", "visa", None],
            "amt": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        }
    )


class _Splits:  # only .train is used
    train = DayRange(1, 3)


def test_codes_by_hand():
    df = _frame()
    enc = fit_on_training_period(df, _Splits(), columns=("ProductCD", "card4"))
    assert enc.categories_ == {"ProductCD": ["C", "H", "W"], "card4": ["mastercard", "visa"]}
    # C=2, H=3, W=4; R and S appear only after training -> unknown (1); None -> missing (0)
    assert enc.encode(df["ProductCD"], "ProductCD").tolist() == [4, 2, 4, 0, 3, 1, 4, 1]
    assert enc.encode(df["card4"], "card4").tolist() == [3, 3, 2, 0, 3, 1, 3, 0]
    assert enc.n_codes("ProductCD") == 5
    assert (MISSING, UNKNOWN) == (0, 1)


def test_codes_do_not_depend_on_the_frame_or_dtype():
    df = _frame()
    enc = fit_on_training_period(df, _Splits(), columns=("ProductCD",))
    full = enc.encode(df["ProductCD"], "ProductCD")
    tail = enc.encode(df["ProductCD"].iloc[5:], "ProductCD")  # e.g. one replayed day
    assert tail.tolist() == full[5:].tolist()
    as_object = enc.encode(df["ProductCD"].astype(object), "ProductCD")
    assert as_object.tolist() == full.tolist()


def test_future_rows_never_change_the_mapping():
    df = _frame()
    a = fit_on_training_period(df, _Splits())
    later = df.copy()
    later.loc[5:, "ProductCD"] = "C"
    later.loc[5:, "card4"] = "amex"
    b = fit_on_training_period(later, _Splits())
    assert a.categories_ == b.categories_


def test_transform_save_and_load(tmp_path):
    df = _frame()
    enc = fit_on_training_period(df, _Splits(), columns=("ProductCD", "card4"))
    out = enc.transform(df)
    assert out["amt"].equals(df["amt"]) and out["ProductCD"].dtype == np.int32
    enc.save(tmp_path / "codes.json")
    again = CategoryEncoder.load(tmp_path / "codes.json")
    assert again.transform(df).equals(out)
