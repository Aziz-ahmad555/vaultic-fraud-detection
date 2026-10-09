"""Fold plans for out-of-sample view predictions (roadmap 7.3).

A fold trains every view on `train` days and predicts the `predict` days, which lie strictly
after them. A fold is one set of view MODELS. Schemes, built from experiments/configs/splits.yaml:

  fixed    ONE fold ("fixed"): views trained on the training period predict validation and,
           only for final runs, the test period with the SAME models (review N2, D77)
  gate     the plan for the fusion gate and everything calibrated (E10-E12): the fixed fold
           plus the rolling folds whose block ends before validation starts, which only add
           earlier gate-training rows (each with its own per-view calibration slice)
  rolling  splits.yaml's rolling out-of-time folds as they are (drift work, Phase 10); not for
           calibration: its later folds are other models predicting validation and test

Calibration scope (D77): a calibrator, conformal set, threshold or routing lambda fitted on the
rows of one fold (model) may only be applied to rows of that same fold; `check_calibration_scope`
refuses anything else. Per-view calibrators are therefore fitted per fold, and everything
fitted on fused scores (calibrate_fused rows) is fitted on the fixed fold and applied to its
validation and test rows.

Row roles (D53, split in time by D76):
  gate_train       rows the fusion gate is trained (and, through an inner time split, tuned) on
  calibrate_views  earlier slice of the calibrate tail (days 144-146): per-view calibrators only
  calibrate_fused  later slice (days 147-150): fused calibration, conformal calibration,
                   decision thresholds and routing lambdas only
  test             rows that evaluate the gate (final runs only)
Neither calibrate slice is ever used for gate training. A fold's own role is gate_train or test;
inside a gate_train fold, predicted days in the slices get their slice's row role. A plan that
predicts test-period days without final=True is refused (CLAUDE.md rule 6).

Optional label maturity: with label_maturity_days = L, a fold trains only on rows whose
label is known when its predicted block starts (its_time + L <= start of block), the same
rule as the harness option `train_label_maturity_days` (D39).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np

from vaultic.data.load import SECONDS_PER_DAY

ROLES = ("gate_train", "test")  # fold roles
ROW_ROLES = ("gate_train", "calibrate_views", "calibrate_fused", "test")  # view-table rows


@dataclass(frozen=True)
class Fold:
    name: str
    train: tuple[int, int]  # inclusive day range
    predict: tuple[int, int]
    role: str
    label_maturity_days: int | None = None
    calibrate_views: tuple[int, int] | None = None  # days for per-view calibrators (D76)
    calibrate_fused: tuple[int, int] | None = None  # days for fused calibration etc. (D76)
    test: tuple[int, int] | None = None  # predicted days that are test rows (fixed fold, final)

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}")
        if not self.train[1] < self.predict[0]:
            raise ValueError(f"fold {self.name}: predicted days must come after training days")
        if self.test is not None and not (
            self.predict[0] < self.test[0] <= self.test[1] == self.predict[1]
        ):
            raise ValueError(f"fold {self.name}: test days must be the tail of its predicted days")

    def train_rows(self, day: np.ndarray, time: np.ndarray) -> np.ndarray:
        day, time = np.asarray(day), np.asarray(time)
        rows = (day >= self.train[0]) & (day <= self.train[1])
        if self.label_maturity_days is not None:
            start = self.predict[0] * SECONDS_PER_DAY
            rows &= time + self.label_maturity_days * SECONDS_PER_DAY <= start
        return rows

    def predict_rows(self, day: np.ndarray) -> np.ndarray:
        day = np.asarray(day)
        return (day >= self.predict[0]) & (day <= self.predict[1])

    def row_roles(self, day: np.ndarray) -> np.ndarray:
        """Role of each predicted row: the fold's role, or a calibrate slice's role."""
        day = np.asarray(day)
        roles = np.full(len(day), self.role, dtype=object)
        if self.test is not None:
            roles[(day >= self.test[0]) & (day <= self.test[1])] = "test"
        if self.role == "gate_train":
            for role, days in (("calibrate_views", self.calibrate_views),
                               ("calibrate_fused", self.calibrate_fused)):  # fmt: skip
                if days is not None:
                    roles[(day >= days[0]) & (day <= days[1])] = role
        return roles


def _overlaps_test(days: tuple[int, int], splits) -> bool:
    return days[1] >= splits.test.first


def check_plan(plan: list[Fold], splits, final: bool) -> None:
    names = [f.name for f in plan]
    if len(set(names)) != len(names):
        raise ValueError("fold names must be unique")
    for role in ROLES:
        blocks = sorted(f.predict for f in plan if f.role == role)
        if any(a[1] >= b[0] for a, b in zip(blocks, blocks[1:], strict=False)):
            raise ValueError(f"predicted blocks of role {role} overlap")
    for f in plan:
        if _overlaps_test(f.predict, splits) and not final:
            raise ValueError(f"fold {f.name} predicts test-period days; only final runs may")
        if f.role == "gate_train" and _overlaps_test(f.predict, splits):
            # only the fixed fold may, and only if every test-period day it predicts is a test row
            if f.test is None or f.test[0] > splits.test.first:
                raise ValueError(
                    f"fold {f.name}: gate-training rows may not come from the test period"
                )
        if _overlaps_test(f.train, splits):
            raise ValueError(f"fold {f.name} trains on test-period days")


def _calibrate(splits) -> dict:
    """The calibrate slices as Fold keyword arguments (empty if splits.yaml has none)."""
    v, f = getattr(splits, "calibrate_views", None), getattr(splits, "calibrate_fused", None)
    if v is None:
        return {}
    return {"calibrate_views": (v.first, v.last), "calibrate_fused": (f.first, f.last)}


def fixed_plan(splits, final: bool = False, label_maturity_days: int | None = None) -> list[Fold]:
    """One fold: the same view models predict validation and (final runs only) test (D77)."""
    t, v, s = splits.train, splits.validation, splits.test
    last = s.last if final else v.last
    plan = [Fold("fixed", (t.first, t.last), (v.first, last), "gate_train", label_maturity_days,
                 test=(s.first, s.last) if final else None, **_calibrate(splits))]  # fmt: skip
    check_plan(plan, splits, final)
    return plan


def gate_plan(splits, final: bool = False, label_maturity_days: int | None = None) -> list[Fold]:
    """The fixed fold plus the rolling folds that end before validation (D77). Each rolling fold
    keeps the last days of its block (as many as calibrate_views has) to calibrate its own
    views; the rest of its block are extra gate-training rows."""
    fixed = fixed_plan(splits, final, label_maturity_days)
    cal = _calibrate(splits)
    plan = []
    for i, (train, block) in enumerate(splits.rolling):
        if block.last >= splits.validation.first:
            continue  # validation and test predictions come from the fixed fold only
        own = {}
        if cal:
            width = cal["calibrate_views"][1] - cal["calibrate_views"][0]
            own = {"calibrate_views": (block.last - width, block.last)}
        plan.append(Fold(f"rolling_{i}", (train.first, train.last), (block.first, block.last),
                         "gate_train", label_maturity_days, **own))  # fmt: skip
    plan += fixed
    check_plan(plan, splits, final)
    return plan


def check_calibration_scope(fitted_on, applied_to, what: str = "calibration") -> None:
    """Refuse a calibrator / conformal set / threshold fitted on one fold's rows and applied to
    rows of another fold (another set of view models), D77."""
    fitted, applied = set(fitted_on), set(applied_to)
    if len(fitted) != 1:
        raise ValueError(
            f"{what} must be fitted on rows of exactly one fold (model); got {sorted(fitted)}"
        )
    if not applied <= fitted:
        raise ValueError(
            f"{what} fitted on fold {sorted(fitted)[0]} cannot be applied to rows of "
            f"{sorted(applied - fitted)} (a different model, D77)"
        )


def rolling_plan(splits, final: bool = False, label_maturity_days: int | None = None) -> list[Fold]:
    plan = []
    for i, (train, block) in enumerate(splits.rolling):
        if block.last < splits.test.first:
            role = "gate_train"
        elif block.first >= splits.test.first:
            if not final:
                continue
            role = "test"
        else:
            raise ValueError(f"rolling fold {i} straddles the start of the test period")
        plan.append(Fold(f"rolling_{i}", (train.first, train.last), (block.first, block.last),
                         role, label_maturity_days,
                         **(_calibrate(splits) if role == "gate_train" else {})))  # fmt: skip
    check_plan(plan, splits, final)
    return plan


def plan_to_json(plan: list[Fold]) -> str:
    return json.dumps([asdict(f) for f in plan], indent=1)


def plan_from_json(text: str) -> list[Fold]:
    def days(d, key):
        return None if d.get(key) is None else tuple(d[key])

    return [Fold(**{**d, "train": tuple(d["train"]), "predict": tuple(d["predict"]),
                    "calibrate_views": days(d, "calibrate_views"),
                    "calibrate_fused": days(d, "calibrate_fused"), "test": days(d, "test")})
            for d in json.loads(text)]  # fmt: skip
