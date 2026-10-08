"""Phase 8.3 cost-based decision engine: Allow / Monitor / Step-up / Hold / Block.

Inputs per transaction: calibrated fraud probability p, amount, the conformal prediction set
(legit / fraud / uncertain / empty, trust/conformal.py) and view disagreement d. Expected loss
EL = p x amount. Rules, first match wins (roadmap 8.3 table):

  Block    EL >= t_block and the set is confidently {fraud}      very high risk, confident
  Hold     EL >= t_hold, or d >= t_dis                            high risk or disagreement
  Step-up  EL >= t_step, or the set is uncertain / empty          medium risk or uncertain
  Monitor  EL >= t_monitor                                        low-medium risk
  Allow    otherwise                                              low risk

with t_step <= t_hold <= t_block. Without conformal sets only EL and d are used.

Cost model (roadmap Phase 13): missed fraud amount + C_FP x false positives + C_rev x reviews,
C_FP = $10, C_rev = $5. How each action enters it (research/decisions.md D43):
  Allow, Monitor  fraud is missed (its amount); legit costs nothing
  Step-up         a simulated OTP costing c_step (default = C_rev) for every customer asked;
                  fraud is stopped with probability step_up_success_rate (default 0.9) and
                  missed otherwise; a legit customer also bears legit_step_up_friction
                  (default $1) (D45)
  Hold            one analyst review (C_rev); fraud caught; a legit transaction is released
  Block           a case is opened (C_rev); fraud caught; a legit transaction is a false
                  positive (C_FP)
Monitor changes no cost term, so cost cannot choose its threshold: t_monitor is set so a fixed
share (monitor_share, default 5%) of the validation transactions that would otherwise be
allowed is monitored. All other thresholds are chosen on validation by minimising total cost
over a quantile grid; never set by hand and never on test.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

ACTIONS = ("allow", "monitor", "step_up", "hold", "block")
ALLOW, MONITOR, STEP_UP, HOLD, BLOCK = range(5)


@dataclass(frozen=True)
class CostModel:
    c_fp: float = 10.0
    c_rev: float = 5.0
    c_step: float | None = None  # None: same as c_rev
    step_up_success_rate: float = 0.9  # share of fraud a step-up stops (D45)
    legit_step_up_friction: float = 1.0  # cost to a legit customer asked to step up (D45)

    @property
    def step(self) -> float:
        return self.c_rev if self.c_step is None else self.c_step

    def per_action(self, y: np.ndarray, amount: np.ndarray) -> np.ndarray:
        """(n, 5) cost of taking each action for each transaction."""
        y, a = np.asarray(y, dtype=float), np.asarray(amount, dtype=float)
        missed = y * a
        return np.column_stack(
            [
                missed,  # allow
                missed,  # monitor
                self.step
                + (1 - self.step_up_success_rate) * missed
                + (1 - y) * self.legit_step_up_friction,  # step-up
                np.full(len(y), self.c_rev),  # hold
                self.c_rev + (1 - y) * self.c_fp,  # block
            ]
        )


DEFAULT_COST = CostModel()  # C_FP $10, C_rev $5 (roadmap Phase 13)


@dataclass(frozen=True)
class Thresholds:
    t_monitor: float
    t_step: float
    t_hold: float
    t_block: float
    t_dis: float = np.inf

    def __post_init__(self) -> None:
        if not self.t_step <= self.t_hold <= self.t_block:
            raise ValueError("thresholds must satisfy t_step <= t_hold <= t_block")


def _sets(conformal_set, n: int):
    if conformal_set is None:
        return np.zeros(n, bool), np.zeros(n, bool)
    s = np.asarray(conformal_set, dtype=object)
    return s == "fraud", np.isin(s, ["uncertain", "empty"])


def decide(p, amount, th: Thresholds, conformal_set=None, disagreement=None) -> np.ndarray:
    """Action code per transaction (index into ACTIONS)."""
    el = np.asarray(p, dtype=float) * np.asarray(amount, dtype=float)
    n = len(el)
    d = np.zeros(n) if disagreement is None else np.asarray(disagreement, dtype=float)
    confident_fraud, uncertain = _sets(conformal_set, n)
    if conformal_set is None:
        confident_fraud = np.ones(n, bool)  # no sets: EL alone decides blocking
    action = np.full(n, ALLOW)
    action[el >= th.t_monitor] = MONITOR
    action[(el >= th.t_step) | uncertain] = STEP_UP
    action[(el >= th.t_hold) | (d >= th.t_dis)] = HOLD
    action[(el >= th.t_block) & confident_fraud] = BLOCK
    return action


def total_cost(action, y, amount, cost: CostModel = DEFAULT_COST) -> float:
    per = cost.per_action(y, amount)
    return float(per[np.arange(len(per)), np.asarray(action)].sum())


def breakdown(action, y, amount, cost: CostModel = DEFAULT_COST) -> dict:
    action, y, a = np.asarray(action), np.asarray(y), np.asarray(amount, dtype=float)
    passed = np.isin(action, [ALLOW, MONITOR])
    stepped = action == STEP_UP
    miss = (1 - cost.step_up_success_rate) * stepped + passed  # expected share missed
    out = {name: int((action == i).sum()) for i, name in enumerate(ACTIONS)}
    out.update(
        missed_fraud_value=float((a * y * miss).sum()),
        missed_fraud_count=float((y * miss).sum()),  # expected count when step-up can fail
        false_positives=int(((action == BLOCK) & (y == 0)).sum()),
        legit_step_ups=int((stepped & (y == 0)).sum()),
        reviews=int(np.isin(action, [HOLD, BLOCK]).sum()),
        total_cost=total_cost(action, y, a, cost),
    )
    return out


def _grid(values: np.ndarray, size: int) -> np.ndarray:
    qs = np.quantile(values, np.linspace(0, 1, size)) if len(values) else np.array([])
    return np.unique(np.concatenate([qs, [np.inf]]))


def choose_thresholds(
    p,
    amount,
    y,
    conformal_set=None,
    disagreement=None,
    cost: CostModel = DEFAULT_COST,
    grid_size: int = 15,
    monitor_share: float = 0.05,
) -> tuple[Thresholds, dict]:
    """Thresholds minimising total cost on VALIDATION rows (pass validation arrays only)."""
    el = np.asarray(p, dtype=float) * np.asarray(amount, dtype=float)
    per = cost.per_action(y, amount)
    n = len(el)
    d = np.zeros(n) if disagreement is None else np.asarray(disagreement, dtype=float)
    confident_fraud, uncertain = _sets(conformal_set, n)
    if conformal_set is None:
        confident_fraud = np.ones(n, bool)
    grid = _grid(el, grid_size)
    dis_grid = _grid(d, grid_size) if disagreement is not None else np.array([np.inf])

    # cost of rows by the action they get; evaluated directly, monitor counted as allow
    best_cost, best = np.inf, None
    rows = np.arange(n)
    for t_dis in dis_grid[::-1]:  # inf first: no disagreement rule unless it pays
        hold_dis = d >= t_dis
        for t_block in grid[::-1]:
            block = (el >= t_block) & confident_fraud
            for t_hold in grid[grid <= t_block][::-1]:
                hold = ((el >= t_hold) | hold_dis) & ~block
                for t_step in grid[grid <= t_hold][::-1]:
                    step = ((el >= t_step) | uncertain) & ~block & ~hold
                    action = np.where(block, BLOCK, np.where(hold, HOLD, np.where(step, STEP_UP,
                                                                                   ALLOW)))  # fmt: skip
                    c = per[rows, action].sum()
                    if c < best_cost - 1e-9:
                        best_cost, best = c, (t_step, t_hold, t_block, t_dis)
    t_step, t_hold, t_block, t_dis = best
    allowed = el[(el < t_step) & ~uncertain & ~(d >= t_dis)]
    t_monitor = float(np.quantile(allowed, 1 - monitor_share)) if len(allowed) else t_step
    th = Thresholds(min(t_monitor, t_step), t_step, t_hold, t_block, t_dis)
    info = {"validation_cost": float(best_cost), "cost_model": asdict(cost),
            "grid_size": grid_size, "monitor_share": monitor_share}  # fmt: skip
    return th, info


def cost_sensitivity(
    p,
    amount,
    y,
    c_fp_values=(2, 5, 10, 20, 50),
    success_rates=(0.7, 0.9, 1.0),
    frictions=(0.0, 1.0, 5.0),
    **kwargs,
) -> list[dict]:
    """Roadmap 13 sensitivity: re-choose thresholds on validation for every combination of
    C_FP ($2-$50), step-up success rate and legit step-up friction (D45)."""
    base = kwargs.pop("cost", DEFAULT_COST)
    out = []
    for c_fp in c_fp_values:
        for rate in success_rates:
            for friction in frictions:
                cost = CostModel(c_fp, base.c_rev, base.c_step, rate, friction)
                th, info = choose_thresholds(p, amount, y, cost=cost, **kwargs)
                out.append({"c_fp": c_fp, "step_up_success_rate": rate,
                            "legit_step_up_friction": friction, "thresholds": asdict(th),
                            "validation_cost": info["validation_cost"]})  # fmt: skip
    return out
