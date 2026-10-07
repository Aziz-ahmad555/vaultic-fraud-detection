"""Phase 1.3: time-based splits and the label-delay rule.

The split definition lives in experiments/configs/splits.yaml; nothing else may hard-code
split days.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from vaultic.data.load import SECONDS_PER_DAY
from vaultic.paths import CONFIG_DIR

SPLITS_PATH = CONFIG_DIR / "splits.yaml"
FIXED_PARTS = ("train", "validation", "test")


@dataclass(frozen=True)
class DayRange:
    first: int
    last: int

    def __post_init__(self) -> None:
        if self.first > self.last:
            raise ValueError(f"empty day range {self.first}-{self.last}")

    def contains(self, day: pd.Series | np.ndarray) -> np.ndarray:
        day = np.asarray(day)
        return (day >= self.first) & (day <= self.last)


@dataclass(frozen=True)
class Splits:
    uid_variant: str
    label_delay_days: int
    train: DayRange
    validation: DayRange
    test: DayRange
    rolling: tuple[tuple[DayRange, DayRange], ...]

    def assign(self, day: pd.Series | np.ndarray) -> np.ndarray:
        """Label every row 'train', 'validation', 'test' or 'unused' (gap days)."""
        out = np.full(len(day), "unused", dtype=object)
        for part in FIXED_PARTS:
            out[getattr(self, part).contains(day)] = part
        return out


def _range(pair: list[int]) -> DayRange:
    if len(pair) != 2:
        raise ValueError(f"a day range needs [first, last], got {pair}")
    return DayRange(int(pair[0]), int(pair[1]))


def load_splits(path: Path = SPLITS_PATH) -> Splits:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    fixed = cfg["fixed"]
    splits = Splits(
        uid_variant=str(cfg["uid_variant"]),
        label_delay_days=int(cfg["label_delay_days"]),
        train=_range(fixed["train"]),
        validation=_range(fixed["validation"]),
        test=_range(fixed["test"]),
        rolling=tuple((_range(f["train"]), _range(f["test"])) for f in cfg.get("rolling", [])),
    )
    validate(splits)
    return splits


def validate(s: Splits) -> None:
    """Splits must be in time order with no overlap: train < validation < test."""
    if s.label_delay_days < 0:
        raise ValueError("label_delay_days must be >= 0")
    if not s.train.last < s.validation.first:
        raise ValueError("validation must start after train ends")
    if not s.validation.last < s.test.first:
        raise ValueError("test must start after validation ends")
    for train, test in s.rolling:
        if not train.last < test.first:
            raise ValueError(
                f"rolling fold tests on days before its training ends: {train}, {test}"
            )


def label_matured(
    label_time: pd.Series | np.ndarray | int,
    as_of_time: pd.Series | np.ndarray | int,
    delay_days: int,
) -> np.ndarray | bool:
    """True where a label from label_time (seconds) is known at as_of_time (seconds).

    The rule: its_time + L <= t, with L in days.
    """
    return np.asarray(label_time) + delay_days * SECONDS_PER_DAY <= np.asarray(as_of_time)


def matured_labels(df: pd.DataFrame, as_of_time: int, delay_days: int) -> pd.Series:
    """isFraud where the label is known at as_of_time, NaN where it is not yet known."""
    known = label_matured(df["TransactionDT"].to_numpy(), as_of_time, delay_days)
    return df["isFraud"].where(known)
