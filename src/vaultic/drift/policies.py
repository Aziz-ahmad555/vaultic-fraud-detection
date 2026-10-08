"""Phase 10.3 retraining policies under delayed labels, with a champion-challenger gate.

  P0  never retrain
  P1  retrain every `retrain_every` days (30: monthly)
  P2  retrain when a label-free detector fires (PSI of the served scores vs the champion's
      reference scores, over a trailing window)
  P3  retrain when a label-based detector fires (ADWIN on the error stream of labels as they
      mature, L days after each transaction)

Day by day from deployment: the champion scores the day's transactions; a transaction is
reviewed when its score reaches the champion's cost threshold, and the day's cost is
missed fraud + C_FP x false positives + C_rev x reviews (eval.stats.cost_breakdown).

Training only ever uses labels known at that moment: rows with its_time + L <= the start of
the training day; with `train_window_days` only the most recent such days (default: all
history; after a drift a long history dilutes the new pattern). Each model is fitted on those rows minus the most recent `holdout_days` of
them; its cost threshold and reference scores come from that holdout. Champion-challenger
gate: the challenger replaces the champion only if, on the challenger's holdout (the most
recent matured labels, excluding any row the champion trained on), its PR-AUC is at least
the champion's + `margin` AND its cost is not higher. A rejected challenger is logged.
P2/P3 wait `cooldown_days` between retraining attempts. An alarm stays pending until a
challenger is accepted, retried every `cooldown_days` up to `max_retries` attempts: right after
a label-based alarm few post-drift labels have matured, so the first challenger is usually
still a pre-drift model and is (correctly) rejected.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from vaultic.data.load import SECONDS_PER_DAY
from vaultic.drift.detectors import ADWIN
from vaultic.drift.signals import score_drift
from vaultic.eval.metrics import C_FP, C_REVIEW, choose_cost_threshold, pr_auc
from vaultic.eval.stats import cost_breakdown

POLICIES = ("P0", "P1", "P2", "P3")


@dataclass
class Model:
    estimator: object
    threshold: float
    trained_day: int
    train_rows: np.ndarray  # row positions used for fitting
    holdout_rows: np.ndarray
    reference_scores: np.ndarray


@dataclass
class PolicyResult:
    policy: str
    daily: pd.DataFrame
    events: list[dict] = field(default_factory=list)

    @property
    def total_cost(self) -> float:
        return float(self.daily["cost"].sum())

    @property
    def n_retrains(self) -> int:
        return sum(e["accepted"] for e in self.events)

    @property
    def n_attempts(self) -> int:
        return len(self.events)

    def pr_auc_by_block(self, scores: pd.DataFrame, block_days: int = 30) -> pd.DataFrame:
        """PR-AUC of the served scores per block of `block_days` days from deployment."""
        block = (scores["day"] - scores["day"].min()) // block_days
        rows = []
        for b, g in scores.groupby(block):
            value = pr_auc(g["y"], g["p"]) if g["y"].nunique() > 1 else np.nan
            rows.append({"block": int(b), "first_day": int(g["day"].min()), "pr_auc": value})
        return pd.DataFrame(rows)


def simulate(
    df: pd.DataFrame,
    features: list[str],
    policy: str,
    deploy_day: int,
    end_day: int,
    fit: Callable[[pd.DataFrame, np.ndarray], object],
    delay_days: int = 30,
    retrain_every: int = 30,
    cooldown_days: int = 7,
    holdout_days: int = 14,
    margin: float = 0.0,
    c_fp: float = C_FP,
    c_rev: float = C_REVIEW,
    psi_threshold: float = 0.2,
    psi_window_days: int = 7,
    adwin_delta: float = 0.002,
    max_retries: int = 6,
    train_window_days: int | None = None,
) -> tuple[PolicyResult, pd.DataFrame]:
    """Returns the result and the served scores (day, row, y, p, flag = reviewed, model)."""
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}")
    day = df["day"].to_numpy()
    time = df["TransactionDT"].to_numpy(dtype=np.int64)
    y = df["isFraud"].to_numpy()
    amount = df["TransactionAmt"].to_numpy(dtype=float)
    X = df[features]

    def train(as_of: int) -> Model:
        known = np.flatnonzero(time + delay_days * SECONDS_PER_DAY <= as_of * SECONDS_PER_DAY)
        if train_window_days is not None:
            known = known[time[known] > time[known].max() - train_window_days * SECONDS_PER_DAY]
        cut = time[known].max() - holdout_days * SECONDS_PER_DAY
        fit_rows, hold = known[time[known] <= cut], known[time[known] > cut]
        est = fit(X.iloc[fit_rows], y[fit_rows])
        s_hold = est.predict_proba(X.iloc[hold])[:, 1]
        thr = choose_cost_threshold(y[hold], s_hold, amount[hold], c_fp, c_rev)
        return Model(est, thr, as_of, fit_rows, hold, s_hold)

    def score(model: Model, rows: np.ndarray) -> np.ndarray:
        return model.estimator.predict_proba(X.iloc[rows])[:, 1]

    def gate_cost(model: Model, rows: np.ndarray) -> float:
        flagged = score(model, rows) >= model.threshold
        return cost_breakdown(y[rows], flagged, amount[rows], c_fp, c_rev)["total_cost"]

    champion, model_id = train(deploy_day), 0
    adwin = ADWIN(delta=adwin_delta)
    served_p = np.full(len(df), np.nan)
    served_flag = np.zeros(len(df), bool)
    served_by = np.full(len(df), -1)
    last_attempt = deploy_day
    pending, tries = False, 0  # P2/P3: an alarm waiting for an accepted challenger
    daily, events = [], []

    for d in range(deploy_day, end_day + 1):
        rows = np.flatnonzero(day == d)
        if len(rows):
            served_p[rows] = score(champion, rows)
            served_flag[rows] = served_p[rows] >= champion.threshold
            served_by[rows] = model_id
        c = cost_breakdown(y[rows], served_flag[rows], amount[rows], c_fp, c_rev)
        daily.append({"day": d, "n": len(rows), "frauds": int(y[rows].sum()), "model": model_id,
                      "cost": c["total_cost"], "reviews": c["n_reviews"]})  # fmt: skip

        alarm = False
        if policy == "P2":
            window = np.flatnonzero(
                (day > d - psi_window_days) & (day <= d) & (served_by == model_id)
            )
            alarm = len(window) > 0 and score_drift(champion.reference_scores,
                                                    served_p[window]) > psi_threshold  # fmt: skip
        elif policy == "P3":  # labels that matured during day d, of transactions we served
            arrival = time + delay_days * SECONDS_PER_DAY
            matured = np.flatnonzero((arrival >= d * SECONDS_PER_DAY)
                                     & (arrival < (d + 1) * SECONDS_PER_DAY) & (served_by >= 0))  # fmt: skip
            for r in matured[np.argsort(arrival[matured], kind="stable")]:
                alarm |= adwin.update(float(served_flag[r] != bool(y[r])))

        if alarm and not pending:
            pending, tries = True, 0
        due = (policy == "P1" and d + 1 - last_attempt >= retrain_every) or (
            policy in ("P2", "P3") and pending and d + 1 - last_attempt >= cooldown_days
        )
        if not due:
            continue
        last_attempt = d + 1
        tries += 1
        challenger = train(d + 1)
        gate = np.setdiff1d(challenger.holdout_rows, champion.train_rows)
        ev = {"day": d + 1, "trigger": "schedule" if policy == "P1" else "alarm",
              "gate_rows": len(gate)}  # fmt: skip
        if len(gate) and y[gate].min() != y[gate].max():
            pr_new, pr_old = pr_auc(y[gate], score(challenger, gate)), pr_auc(
                y[gate], score(champion, gate)
            )
            cost_new, cost_old = gate_cost(challenger, gate), gate_cost(champion, gate)
            accepted = pr_new >= pr_old + margin and cost_new <= cost_old
            ev.update(pr_auc_challenger=pr_new, pr_auc_champion=pr_old, cost_challenger=cost_new,
                      cost_champion=cost_old, accepted=bool(accepted))  # fmt: skip
        else:
            ev.update(accepted=False, reason="gate window has one class only")
        events.append(ev)
        if ev["accepted"]:
            champion, model_id = challenger, model_id + 1
            adwin.reset()
        if ev["accepted"] or tries >= max_retries:
            pending = False

    served = np.flatnonzero(served_by >= 0)
    scores = pd.DataFrame({"day": day[served], "row": served, "y": y[served],
                           "p": served_p[served], "flag": served_flag[served],
                           "model": served_by[served]})  # fmt: skip
    return PolicyResult(policy, pd.DataFrame(daily), events), scores
