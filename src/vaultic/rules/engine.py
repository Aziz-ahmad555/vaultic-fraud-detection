"""Rules that run before the model (Phase B, D195).

Each rule reads point-in-time inputs that the feature pipeline already computes for every
transaction (strictly before t; labels known by t - 30 days), so the same rule gives the same
answer on validation (offline features) and in a live scenario (online features):

  R1 customer velocity, 1 h     uid_n_1h >= theta              -> step-up
  R2 customer velocity, 24 h    uid_n_24h >= theta             -> step-up
  R3 shared device              ent_uids_7d_device >= theta    -> hold
       (distinct customers on the device in the 7 days before t)
  R4 blocklist                  max(g_fraud_rate_card, g_fraud_rate_device) >= theta -> hold
       (share of the card's / device's earlier transactions with a known fraud label)
  R5 new-customer amount cap    hist_n_past == 0 and TransactionAmt >= theta         -> step-up
  R6 new device + high amount   new_device == 1 and amt_ratio_median >= theta        -> hold

A rule fires when its value is at least its threshold (and its condition holds). Thresholds are
chosen on validation only (rules/calibrate.py): D196 (primary, rules/thresholds_v2.json) uses one
per-rule FPR cap for all six rules, the largest meeting D191's budgets on validation; D195
(rules/thresholds_v1.json, cap 0.5% per D192) is reported next to it as a sensitivity row. A
missing input never fires a rule (rule 11: missing stays missing).

Decisions: "rules only" takes the most severe action of the rules that fired (allow if none);
"rules + model" takes the more severe of that and the model's E12 action; "model only" is the E12
action. Degraded mode (model off) = rules only. Every decision carries the rule trace.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ACTIONS = ("allow", "monitor", "step_up", "hold", "block")
SEVERITY = {a: i for i, a in enumerate(ACTIONS)}
THRESHOLDS_FILE = Path(__file__).with_name("thresholds_v2.json")  # D196 primary
SENSITIVITY_FILE = Path(__file__).with_name("thresholds_v1.json")  # D195, sensitivity row


@dataclass(frozen=True)
class Rule:
    rule_id: str
    name: str
    inputs: tuple[str, ...]
    action: str
    condition: str = ""  # human-readable condition (shown in the trace)

    def value(self, row: dict) -> float:
        """The rule's score for one transaction (NaN = cannot fire)."""
        g = {k: _num(row.get(k)) for k in self.inputs}
        if self.rule_id == "R4":
            vals = [g["g_fraud_rate_card"], g["g_fraud_rate_device"]]
            vals = [v for v in vals if not np.isnan(v)]
            return max(vals) if vals else np.nan
        if self.rule_id == "R5":
            return g["TransactionAmt"] if g["hist_n_past"] == 0 else np.nan
        if self.rule_id == "R6":
            return g["amt_ratio_median"] if g["new_device"] == 1 else np.nan
        return g[self.inputs[0]]


def _num(x) -> float:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return np.nan
    return x


RULES = (
    Rule("R1", "customer velocity, 1 hour", ("uid_n_1h",), "step_up",
         "customer's transactions in the previous hour"),
    Rule("R2", "customer velocity, 24 hours", ("uid_n_24h",), "step_up",
         "customer's transactions in the previous 24 hours"),
    Rule("R3", "shared device", ("ent_uids_7d_device",), "hold",
         "distinct customers on this device in the previous 7 days"),
    Rule("R4", "blocklist (card or device with known fraud)",
         ("g_fraud_rate_card", "g_fraud_rate_device"), "hold",
         "known-fraud share of the card's or device's earlier transactions (labels known by t - 30 days)"),
    Rule("R5", "new-customer amount cap", ("hist_n_past", "TransactionAmt"), "step_up",
         "amount, for a customer with no earlier transaction"),
    Rule("R6", "new device + high amount", ("new_device", "amt_ratio_median"), "hold",
         "amount / customer's median amount, on a device new to the customer"),
)  # fmt: skip
BY_ID = {r.rule_id: r for r in RULES}


def load_thresholds(path: Path = THRESHOLDS_FILE) -> dict[str, float]:
    data = json.loads(path.read_text("utf-8"))
    return {k: float(v["threshold"]) for k, v in data["rules"].items()}


def evaluate(row: dict, thresholds: dict[str, float]) -> dict:
    """Rule decision for one transaction: action and the full trace (every rule, fired or not)."""
    trace, action = [], "allow"
    for rule in RULES:
        v = rule.value(row)
        th = thresholds[rule.rule_id]
        fired = bool(np.isfinite(v) and v >= th)
        trace.append({"rule": rule.rule_id, "name": rule.name, "value": None if np.isnan(v) else v,
                      "threshold": th, "fired": fired, "action": rule.action if fired else None,
                      "condition": f"{rule.condition} >= {th:g}"})  # fmt: skip
        if fired and SEVERITY[rule.action] > SEVERITY[action]:
            action = rule.action
    return {"action": action, "alert": SEVERITY[action] >= SEVERITY["step_up"], "trace": trace}


def combine(rule_action: str, model_action: str | None) -> dict:
    """Rules + model: the more severe action. model_action None = degraded mode (rules only)."""
    if model_action is None:
        return {"action": rule_action, "mode": "degraded (rules only, model off)"}
    action = rule_action if SEVERITY[rule_action] > SEVERITY[model_action] else model_action
    return {"action": action, "mode": "rules + model"}


def rule_values(frame, rule: Rule) -> np.ndarray:
    """A rule's value for every row of a feature frame (vectorised; NaN = cannot fire)."""

    def col(c):
        return frame[c].to_numpy(dtype=float) if c in frame else np.full(len(frame), np.nan)

    if rule.rule_id == "R4":
        return np.fmax(col("g_fraud_rate_card"), col("g_fraud_rate_device"))
    if rule.rule_id == "R5":
        return np.where(col("hist_n_past") == 0, col("TransactionAmt"), np.nan)
    if rule.rule_id == "R6":
        return np.where(col("new_device") == 1, col("amt_ratio_median"), np.nan)
    return col(rule.inputs[0])
