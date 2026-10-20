"""Feedback loop with champion-challenger retraining under delayed labels (EXP-310; D210, D211).

Every setting below is fixed by the pre-registration (research/preregistration_feedback.md) and
its amendment 1; nothing here is tuned.

One simulation = one label delay L and one seed s; every policy runs in the same pass so that
challengers are shared (a challenger depends only on the attempt day and L):

  P0          never retrain
  P1          attempt at every 7-day slot from deployment
  P2          label-free: PSI of the scores served by the current champion over its last 7
              served days vs its reference scores > 0.2, checked daily; the reference and the
              window restart after a promotion (D211 b); an alarm makes the next slot an attempt
  P3          label-based: ADWIN (delta 0.002) on the 0/1 error stream (alert = step-up or
              higher vs the label) of the current champion's served decisions, in label-arrival
              order (time + L); the detector restarts after a promotion
  P1_ungated  P1 with every fitted challenger accepted (secondary, H-FB4)
  P1_monthly  attempts every 30 days from deployment (the monthly cadence, secondary; D214)
  P1_4weekly  every 28 days: the design run EXP-310-dev only (kept so its config stays true)

A pending P2 / P3 alarm is attempted at every slot until a challenger is accepted, at most 6
attempts (D44 retry rule; slots per D211).

Label curtain (strict, D214): a label is used (training, threshold, gate) only when
its_time + L < the simulated time; training at attempt day t uses rows with
TransactionDT + L * 86400 < t * 86400 (the design run used <=, which differs only for rows
exactly on a day boundary at L = 0). ADWIN reads, at the end of day d, the labels that arrived
during day d (arrival time + L before the end of day d). Matured rows are split by time (D58): fit = all but the
most recent 14 matured days, threshold slice = the next 7 (E12 thresholds and P2's reference),
gate window = the most recent 7. The gate compares challenger and current champion on the gate
window after removing every row either trained on; fewer than 50 frauds there = rejected
("insufficient evidence", checked before fitting); otherwise accepted only if the PR-AUC gain is
>= 0.005 with its 95% paired-bootstrap CI (1,000 resamples, seed 0) entirely above 0 AND the
challenger's E12 cost is not higher (D211 c: E12 decide on p x amount, choose_thresholds on the
threshold slice, total_cost).

Serving: the scores of day d come from the model serving at the start of day d.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from vaultic.data.load import SECONDS_PER_DAY
from vaultic.drift.detectors import ADWIN
from vaultic.drift.signals import score_drift
from vaultic.eval.bootstrap import paired_bootstrap
from vaultic.eval.metrics import pr_auc
from vaultic.trust.decision import STEP_UP, choose_thresholds, decide, total_cost

SLOT_DAYS = 7
FOUR_WEEKS = 28  # design run only (P1_4weekly)
MONTH_DAYS = 30  # P1_monthly (D214)
FIT_GAP_DAYS = 14  # threshold slice + gate window
THRESHOLD_DAYS = 7
GATE_DAYS = 7
MARGIN = 0.005
MIN_FRAUD = 50
PSI_THRESHOLD = 0.2
PSI_WINDOW_DAYS = 7
ADWIN_DELTA = 0.002
MAX_RETRIES = 6
GATE_N_BOOT = 1000
POLICIES = ("P0", "P1", "P2", "P3", "P1_ungated", "P1_monthly", "P1_4weekly")
DAY = SECONDS_PER_DAY


@dataclass
class Data:
    """One simulation's inputs: rows sorted by time, the B5 inputs for delay L."""

    time: np.ndarray  # TransactionDT (seconds)
    y: np.ndarray
    amount: np.ndarray
    X: pd.DataFrame
    delay: int  # L, days

    def __post_init__(self):
        self.time = np.asarray(self.time, dtype=np.int64)
        if np.any(np.diff(self.time) < 0):
            raise ValueError("rows must be sorted by TransactionDT")
        self.y = np.asarray(self.y).astype(int)
        self.amount = np.asarray(self.amount, dtype=float)
        self.day = self.time // DAY
        self.arrival_day = (self.time + self.delay * DAY) // DAY  # when each label is known


@dataclass
class Model:
    model_id: str
    estimator: object
    thresholds: object  # trust.decision.Thresholds
    reference: np.ndarray  # P2 reference scores (threshold slice)
    train_mask: np.ndarray  # rows the model trained on (bool, all rows)
    trained_day: int
    cache: np.ndarray | None = None  # score of every row, NaN until computed
    threshold_mask: np.ndarray | None = None  # rows of its threshold slice (bool, all rows)

    def score(self, data: Data, rows: np.ndarray) -> np.ndarray:
        rows = np.asarray(rows, dtype=np.int64)
        if self.cache is None:
            self.cache = np.full(len(data.time), np.nan)
        todo = rows[np.isnan(self.cache[rows])]
        if len(todo):
            self.cache[todo] = self.estimator.predict_proba(data.X.iloc[todo])[:, 1]
        return self.cache[rows].copy()


# ---- the label curtain and the D58 split ---------------------------------------------------
def matured_rows(data: Data, as_of_day: int) -> np.ndarray:
    """Rows whose label is known at the start of as_of_day: time + L days < as_of_day (strict,
    D214: a row exactly at the boundary is not yet known)."""
    return np.flatnonzero(data.time + data.delay * DAY < as_of_day * DAY)


def split_matured(data: Data, as_of_day: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(fit, threshold slice, gate window) of the matured rows, by time (D58)."""
    known = matured_rows(data, as_of_day)
    if not len(known):
        return known, known, known
    t = data.time[known]
    gate_cut = t.max() - GATE_DAYS * DAY
    fit_cut = gate_cut - THRESHOLD_DAYS * DAY
    return known[t <= fit_cut], known[(t > fit_cut) & (t <= gate_cut)], known[t > gate_cut]


def gate_rows(data: Data, as_of_day: int, *models: Model) -> np.ndarray:
    """The gate window minus every row the given models trained on."""
    _, _, gate = split_matured(data, as_of_day)
    keep = np.ones(len(gate), bool)
    for m in models:
        keep &= ~m.train_mask[gate]
    return gate[keep]


def e12_actions(model: Model, p: np.ndarray, amount: np.ndarray) -> np.ndarray:
    return decide(p, amount, model.thresholds)


def make_model(model_id, estimator, data: Data, train_rows, threshold_rows, trained_day) -> Model:
    """A serving model: E12 thresholds and P2 reference chosen on its threshold slice."""
    p_thr = estimator.predict_proba(data.X.iloc[threshold_rows])[:, 1]
    th, _ = choose_thresholds(p_thr, data.amount[threshold_rows], data.y[threshold_rows])
    mask = np.zeros(len(data.time), bool)
    mask[np.asarray(train_rows, dtype=np.int64)] = True
    m = Model(model_id, estimator, th, p_thr, mask, trained_day)
    m.threshold_mask = np.zeros(len(data.time), bool)
    m.threshold_mask[np.asarray(threshold_rows, dtype=np.int64)] = True
    m.cache = np.full(len(data.time), np.nan)
    m.cache[np.asarray(threshold_rows, dtype=np.int64)] = p_thr
    return m


def fit_challenger(data: Data, as_of_day: int, fit: Callable, seed_tag: str = "") -> Model:
    fit_rows, thr_rows, _ = split_matured(data, as_of_day)
    if np.any(data.time[fit_rows] + data.delay * DAY >= as_of_day * DAY):
        raise AssertionError("label curtain violated")  # cannot happen; guards the tests
    est = fit(data.X.iloc[fit_rows], data.y[fit_rows])
    return make_model(
        f"challenger_d{as_of_day}{seed_tag}", est, data, fit_rows, thr_rows, as_of_day
    )


def gate(data: Data, champion: Model, challenger: Model, rows: np.ndarray) -> dict:
    """D210 gate on `rows` (already without either model's training rows)."""
    y = data.y[rows]
    pc, pn = champion.score(data, rows), challenger.score(data, rows)
    bs = paired_bootstrap(y, [pn], [pc], pr_auc, n_boot=GATE_N_BOOT, seed=0, n_jobs=1)
    cost_c = total_cost(e12_actions(champion, pc, data.amount[rows]), y, data.amount[rows])
    cost_n = total_cost(e12_actions(challenger, pn, data.amount[rows]), y, data.amount[rows])
    auc_ok = bs["diff"] >= MARGIN and bs["ci_low"] > 0
    cost_ok = cost_n <= cost_c
    reason = (
        "accepted"
        if (auc_ok and cost_ok)
        else ("PR-AUC gain below the margin or CI not above 0" if not auc_ok else "cost higher")
    )
    return {"accepted": bool(auc_ok and cost_ok), "reason": reason, "gain": bs["diff"],
            "ci_low": bs["ci_low"], "ci_high": bs["ci_high"], "cost_champion": float(cost_c),
            "cost_challenger": float(cost_n), "gate_rows": int(len(rows)),
            "gate_frauds": int(y.sum()),
            # rows where the champion's E12 thresholds were chosen: its cost there is in-sample
            "champion_threshold_rows_in_gate": int(champion.threshold_mask[rows].sum())
            if champion.threshold_mask is not None else 0}  # fmt: skip


# ---- the simulation -----------------------------------------------------------------------
@dataclass
class _State:
    champion: Model
    pending: bool = False
    retries: int = 0
    served_days: list = field(default_factory=list)  # days served by the current champion
    adwin: ADWIN = field(default_factory=lambda: ADWIN(delta=ADWIN_DELTA))


def slots(deploy_day: int, end_day: int, every: int = SLOT_DAYS) -> list[int]:
    return list(range(deploy_day + every, end_day + 1, every))


def simulate(data: Data, champion: Model, deploy_day: int, end_day: int, fit: Callable,
             policies=POLICIES, log=None) -> tuple[pd.DataFrame, list[dict]]:  # fmt: skip
    """Returns (served: one row per policy x served transaction, events: every attempt and alarm)."""
    log = log or (lambda m: None)
    weekly, four = set(slots(deploy_day, end_day)), set(slots(deploy_day, end_day, FOUR_WEEKS))
    monthly = set(slots(deploy_day, end_day, MONTH_DAYS))
    state = {p: _State(champion) for p in policies}
    challengers: dict[int, Model] = {}
    served, events = [], []
    served_model = {p: np.full(len(data.time), "", dtype=object) for p in policies}
    rows_of_day = {int(d): np.flatnonzero(data.day == d) for d in range(deploy_day, end_day + 1)}

    def attempt(pol: str, d: int, st: _State) -> None:
        rows = gate_rows(data, d, st.champion)  # the challenger never trained on the gate window
        frauds = int(data.y[rows].sum())
        ev = {"policy": pol, "day": d, "delay": data.delay, "kind": "attempt",
              "champion": st.champion.model_id}  # fmt: skip
        if frauds < MIN_FRAUD:
            ev.update(accepted=False, reason="insufficient evidence", gate_rows=int(len(rows)),
                      gate_frauds=frauds)  # fmt: skip
        else:
            if d not in challengers:
                log(f"L={data.delay} day {d}: fitting the challenger")
                challengers[d] = fit_challenger(data, d, fit)
            ch = challengers[d]
            rows = gate_rows(data, d, st.champion, ch)
            if pol == "P1_ungated":
                ev.update(accepted=True, reason="ungated", gate_rows=int(len(rows)),
                          gate_frauds=int(data.y[rows].sum()))  # fmt: skip
            else:
                ev.update(gate(data, st.champion, ch, rows))
            ev["challenger"] = ch.model_id
            if ev["accepted"]:
                st.champion, st.pending, st.retries = ch, False, 0
                st.served_days, st.adwin = [], ADWIN(delta=ADWIN_DELTA)  # D211 b
        if not ev["accepted"] and pol in ("P2", "P3"):
            st.retries += 1
            if st.retries >= MAX_RETRIES:
                st.pending, st.retries = False, 0
                ev["alarm_dropped"] = True
        events.append(ev)

    for d in range(deploy_day, end_day + 1):
        for pol, st in state.items():  # attempts at the start of day d
            due = ((pol in ("P1", "P1_ungated") and d in weekly) or (pol == "P1_4weekly" and d in four)
                   or (pol == "P1_monthly" and d in monthly)
                   or (pol in ("P2", "P3") and st.pending and d in weekly))  # fmt: skip
            if due:
                attempt(pol, d, st)
        rows = rows_of_day[d]
        for pol, st in state.items():
            if not len(rows):
                continue
            p = st.champion.score(data, rows)
            act = e12_actions(st.champion, p, data.amount[rows])
            served.append(pd.DataFrame({"policy": pol, "row": rows, "day": d, "p": p,
                                        "action": act, "model": st.champion.model_id}))  # fmt: skip
            served_model[pol][rows] = st.champion.model_id
            st.served_days.append(d)
            if pol == "P2" and not st.pending and len(st.served_days) >= PSI_WINDOW_DAYS:
                window = [x for x in st.served_days[-PSI_WINDOW_DAYS:]]
                cur = np.concatenate([st.champion.score(data, rows_of_day[x]) for x in window])
                value = score_drift(st.champion.reference, cur)
                if value > PSI_THRESHOLD:
                    st.pending = True
                    events.append({"policy": pol, "day": d, "delay": data.delay, "kind": "alarm",
                                   "detector": "PSI", "value": float(value),
                                   "champion": st.champion.model_id})  # fmt: skip
        if "P3" in state:  # labels arriving on day d, for transactions the champion served
            st = state["P3"]
            arriving = np.flatnonzero((data.arrival_day == d) & (data.day >= deploy_day)
                                      & (data.day <= d))  # fmt: skip
            if len(arriving):
                served_rows = arriving[served_model["P3"][arriving] == st.champion.model_id]
                if len(served_rows):
                    p = st.champion.score(data, served_rows)
                    alert = e12_actions(st.champion, p, data.amount[served_rows]) >= STEP_UP
                    errors = (alert.astype(int) != data.y[served_rows]).astype(int)
                    fired = any([st.adwin.update(e) for e in errors])
                    if fired and not st.pending:
                        st.pending = True
                        events.append({"policy": "P3", "day": d, "delay": data.delay,
                                       "kind": "alarm", "detector": "ADWIN",
                                       "champion": st.champion.model_id})  # fmt: skip
    out = pd.concat(served, ignore_index=True) if served else pd.DataFrame()
    return out, events


# ---- evaluation ---------------------------------------------------------------------------
def served_cost(data: Data, served: pd.DataFrame) -> float:
    rows = served["row"].to_numpy()
    return float(total_cost(served["action"].to_numpy(), data.y[rows], data.amount[rows]))


def evaluate(datas: dict[int, Data], served_by_seed: dict[int, pd.DataFrame], days: tuple[int, int],
             policies=POLICIES, n_boot: int = 1000, reference: str = "P0",
             boot_seed: int = 0) -> dict:  # fmt: skip
    """Per policy: PR-AUC (seed mean, per seed, 95% CI over rows), E12 cost (seed mean, per
    seed), and the paired comparison with the reference policy (PR-AUC and cost)."""
    from vaultic.eval.bootstrap import _iter_resample_indices, seed_mean_ci

    lo, hi = days
    seeds = sorted(served_by_seed)
    data = datas[seeds[0]]
    out = {}
    per = {}
    for pol in policies:
        scores, costs, rows_ref = [], [], None
        for s in seeds:
            f = served_by_seed[s]
            f = f[(f["policy"] == pol) & (f["day"] >= lo) & (f["day"] <= hi)].sort_values("row")
            rows = f["row"].to_numpy()
            if rows_ref is None:
                rows_ref = rows
            if not np.array_equal(rows, rows_ref):
                raise ValueError("every seed must serve the same rows")
            scores.append(f["p"].to_numpy())
            costs.append(served_cost(datas[s], f))
            per[(pol, s)] = f
        y = data.y[rows_ref]
        ci = seed_mean_ci(y, scores, pr_auc, n_boot=n_boot, seed=boot_seed)
        out[pol] = {"pr_auc": float(np.mean([pr_auc(y, x) for x in scores])),
                    "pr_auc_per_seed": [float(pr_auc(y, x)) for x in scores],
                    "pr_auc_ci": [float(ci[0]), float(ci[1])], "cost": float(np.mean(costs)),
                    "cost_per_seed": costs, "rows": int(len(rows_ref)), "frauds": int(y.sum())}  # fmt: skip
        out[pol]["_scores"], out[pol]["_rows"] = scores, rows_ref
    ref = out.get(reference)
    if ref is not None:
        y = data.y[ref["_rows"]]
        for pol in policies:
            if pol == reference:
                continue
            o = out[pol]
            o["vs_" + reference] = paired_bootstrap(y, o["_scores"], ref["_scores"], pr_auc,
                                                    n_boot=n_boot, seed=boot_seed)  # fmt: skip
            # cost difference (policy - reference) on the same resamples, seed-averaged
            per_row = {}
            for name in (pol, reference):
                per_row[name] = []
                for s in seeds:
                    f = per[(name, s)]
                    r = f["row"].to_numpy()
                    from vaultic.trust.decision import DEFAULT_COST

                    c = DEFAULT_COST.per_action(data.y[r], data.amount[r])
                    per_row[name].append(c[np.arange(len(r)), f["action"].to_numpy()])
            a, b = np.mean(per_row[pol], 0), np.mean(per_row[reference], 0)
            diffs = [float((a[idx] - b[idx]).sum()) for idx in
                     _iter_resample_indices(len(y), n_boot, boot_seed, y)]  # fmt: skip
            q = np.quantile(diffs, [0.025, 0.975])
            p = 2 * min(np.mean(np.array(diffs) <= 0), np.mean(np.array(diffs) >= 0))
            o["cost_vs_" + reference] = {"diff": float((a - b).sum()), "ci_low": float(q[0]),
                                         "ci_high": float(q[1]), "p_value": float(min(p, 1.0))}  # fmt: skip
    for pol in out:
        out[pol].pop("_scores")
        out[pol].pop("_rows")
    return out
