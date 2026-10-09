"""One shared category -> integer code mapping per categorical column, fitted on the training
period only and reused by every view that needs codes (sequences, anomaly; GNN node features
later). Fixes the D36 caveat: codes no longer depend on which frame a view happens to receive.

Codes:  0 = missing (NaN / None)
        1 = unknown (a value never seen in the training period)
        2.. = training-period categories, sorted by their string form (deterministic)
A value is matched by its string form, so pandas category, object and numeric dtypes of the
same value get the same code. The mapping carries no label information.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

MISSING = 0
UNKNOWN = 1
FIRST_CODE = 2

# IEEE-CIS string columns (pandas category after vaultic.data.load) plus ProductCD.
DEFAULT_COLUMNS = (
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain",
    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
    "M6",
    "M7",
    "M8",
    "M9",
    "DeviceType",
    "DeviceInfo",
)


def _as_text(values) -> tuple[np.ndarray, np.ndarray]:
    s = pd.Series(values).astype(object)
    missing = s.isna().to_numpy()
    return s.where(~missing, "").astype(str).to_numpy(dtype=object), missing


class CategoryEncoder:
    def __init__(self) -> None:
        self.categories_: dict[str, list[str]] = {}

    @property
    def columns(self) -> list[str]:
        return list(self.categories_)

    def fit(self, train: pd.DataFrame, columns=DEFAULT_COLUMNS) -> CategoryEncoder:
        """`train` must hold training-period rows only (see fit_on_training_period)."""
        for col in columns:
            if col not in train:
                continue
            text, missing = _as_text(train[col])
            self.categories_[col] = sorted(set(text[~missing].tolist()))
        return self

    def n_codes(self, col: str) -> int:
        """Size of an embedding table for this column (missing + unknown + categories)."""
        return FIRST_CODE + len(self.categories_[col])

    def encode(self, values, col: str) -> np.ndarray:
        if col not in self.categories_:
            raise KeyError(f"{col!r} was not fitted")
        text, missing = _as_text(values)
        pos = pd.Index(self.categories_[col]).get_indexer(text)
        codes = np.where(pos >= 0, pos + FIRST_CODE, UNKNOWN)
        return np.where(missing, MISSING, codes).astype(np.int32)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fitted columns replaced by their codes; other columns unchanged."""
        out = df.copy()
        for col in self.categories_:
            if col in out:
                out[col] = self.encode(out[col], col)
        return out

    def to_dict(self) -> dict:
        return {"categories": self.categories_}

    @classmethod
    def from_dict(cls, data: dict) -> CategoryEncoder:
        enc = cls()
        enc.categories_ = {k: list(v) for k, v in data["categories"].items()}
        return enc

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> CategoryEncoder:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def fit_on_training_period(df: pd.DataFrame, splits, columns=DEFAULT_COLUMNS) -> CategoryEncoder:
    """Fit on the rows whose `day` lies in the training period of `splits` (data.splits)."""
    train = df[splits.train.contains(df["day"].to_numpy())]
    return CategoryEncoder().fit(train, columns)
