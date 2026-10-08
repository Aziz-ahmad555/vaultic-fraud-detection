"""Fold plans for out-of-sample view predictions (roadmap 7.3).

A fold trains every view on `train` days and predicts the `predict` days, which lie strictly
after them. Two schemes, both built from experiments/configs/splits.yaml:

  fixed    views trained on the training period predict validation (the rows the gate is
           trained on) and, only for final runs, the test period
  rolling  splits.yaml's rolling out-of-time folds: every fold whose predicted block ends
           before the test period gives gate-training rows (more of them than validation
           alone); the fold that predicts the test period is used only for final runs

Role "gate_train" rows train the fusion gate; role "test" rows evaluate it. A plan that
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

ROLES = ("gate_train", "test")


@dataclass(frozen=True)
class Fold:
    name: str
    train: tuple[int, int]  # inclusive day range
    predict: tuple[int, int]
    role: str
    label_maturity_days: int | None = None

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}")
        if not self.train[1] < self.predict[0]:
            raise ValueError(f"fold {self.name}: predicted days must come after training days")

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
            raise ValueError(f"fold {f.name}: gate-training rows may not come from the test period")
        if _overlaps_test(f.train, splits):
            raise ValueError(f"fold {f.name} trains on test-period days")


def fixed_plan(splits, final: bool = False, label_maturity_days: int | None = None) -> list[Fold]:
    t, v, s = splits.train, splits.validation, splits.test
    plan = [Fold("fixed_validation", (t.first, t.last), (v.first, v.last), "gate_train",
                 label_maturity_days)]  # fmt: skip
    if final:
        plan.append(Fold("fixed_test", (t.first, t.last), (s.first, s.last), "test",
                         label_maturity_days))  # fmt: skip
    check_plan(plan, splits, final)
    return plan


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
                         role, label_maturity_days))  # fmt: skip
    check_plan(plan, splits, final)
    return plan


def plan_to_json(plan: list[Fold]) -> str:
    return json.dumps([asdict(f) for f in plan], indent=1)


def plan_from_json(text: str) -> list[Fold]:
    return [Fold(**{**d, "train": tuple(d["train"]), "predict": tuple(d["predict"])})
            for d in json.loads(text)]  # fmt: skip
