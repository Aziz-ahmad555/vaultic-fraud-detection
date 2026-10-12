"""Phase 9.2 reason codes: every model feature mapped to a plain-language template.

A reason is (feature, sentence, signed points). Points come from the feature's SHAP value in
log-odds, on the credit-scoring "points to double the odds" scale: points = phi * PDO / ln 2
with PDO = 20, so +20 points means the feature doubles the fraud odds (research/decisions.md
D38). Points are additive like SHAP values.

Wording rules (CLAUDE.md):
- Masked raw features keep honest wording ("Unusual value in masked counter C13"); no meaning
  is invented for them. Only what Vesta's data description states is used (card = payment
  card fields, addr1 / addr2 = billing region / country, all still masked values).
- Time is TransactionDT, a relative clock: hours are "relative hour", never local time.
- A missing value is reported as missing, never as a guess (rule 11).
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass

PDO = 20.0  # points to double the odds
POINTS_PER_LOGODDS = PDO / math.log(2)

WINDOW_TEXT = {"1h": "hour", "24h": "24 hours", "7d": "7 days", "30d": "30 days"}
ENTITY_TEXT = {
    "card": "card",
    "email": "email domain",
    "address": "billing address",
    "device": "device",
    "addr2": "billing country code",
    "product": "product code",
}


def duration(seconds: float) -> str:
    s = abs(float(seconds))
    if s < 3600:
        return f"{s / 60:.0f} min"
    if s < 2 * 86_400:
        return f"{s / 3600:.1f} h"
    return f"{s / 86_400:.1f} days"


@dataclass(frozen=True)
class Template:
    """`text` is a str.format template over: v (the value), top (1 - v), dur (v as a
    duration), plus any named groups of the feature pattern (w = window, e = entity).
    `up` / `down` replace `text` when the feature raises / lowers risk; `when` maps exact
    values (binary flags) to sentences; `missing` is used for NaN."""

    text: str = ""
    up: str | None = None
    down: str | None = None
    when: Mapping[float, str] | None = None
    missing: str = "No value for {feature} (not available for this transaction)"


def _masked(kind: str) -> Template:
    return Template(
        up=f"Unusual value in masked {kind} {{feature}}",
        down=f"Value in masked {kind} {{feature}} lowers the risk score",
        missing=f"No value in masked {kind} {{feature}}",
    )


_W = r"(?P<w>1h|24h|7d|30d)"
_E = r"(?P<e>card|email|address|device)"

# (pattern, template); the first full match wins, so exact names come before families.
LIBRARY: list[tuple[str, Template]] = [
    # --- raw columns with a documented meaning --------------------------------------------
    ("TransactionAmt", Template("Amount ${v:,.2f}")),
    ("ProductCD", Template("Product code {raw}", missing="No product code")),
    ("P_emaildomain", Template("Purchaser email domain {raw}", missing="No purchaser email")),
    ("R_emaildomain", Template("Recipient email domain {raw}", missing="No recipient email")),
    ("DeviceType", Template("Device type {raw}", missing="No device information")),
    ("DeviceInfo", Template("Device {raw}", missing="No device information")),
    ("has_identity", Template(when={1: "Identity record present", 0: "No identity record"})),
    ("addr1", _masked("billing-region code")),
    ("addr2", _masked("billing-country code")),
    (r"card[1-6]", _masked("payment-card field")),
    (r"dist[12]", _masked("distance field")),
    # --- masked raw families --------------------------------------------------------------
    (r"C\d+", _masked("counter")),
    (r"D\d+", _masked("time-delta field")),
    (r"M\d", _masked("match flag")),
    (r"V\d+", _masked("Vesta feature")),
    (r"id_\d+", _masked("identity field")),
    # --- point-in-time base features (features/pipeline.py) -------------------------------
    ("uid_n_past", Template("Customer has {v:.0f} earlier transactions")),
    ("uid_amt_mean_past", Template("Customer's earlier average amount is ${v:,.2f}",
                                   missing="No earlier amounts for this customer")),  # fmt: skip
    ("uid_amt_std_past", Template("Spread of the customer's earlier amounts is ${v:,.2f}",
                                  missing="Too few earlier amounts to measure spread")),  # fmt: skip
    ("uid_amt_vs_mean", Template("Amount is {v:.1f}x this customer's earlier average",
                                 missing="No earlier amounts for this customer")),  # fmt: skip
    ("uid_secs_since_prev", Template("{dur} since this customer's previous transaction",
                                     missing="First transaction seen for this customer")),  # fmt: skip
    (rf"uid_n_{_W}", Template("{v:.0f} earlier transactions by this customer in the last {w}")),
    (rf"uid_amt_sum_{_W}", Template("Customer spent ${v:,.2f} in the last {w}")),
    ("uid_n_labels_known", Template("{v:.0f} of the customer's earlier transactions have "
                                    "confirmed labels (older than the label delay)")),  # fmt: skip
    ("uid_fraud_known", Template("{v:.0f} of the customer's earlier transactions were "
                                 "confirmed fraud")),  # fmt: skip
    ("uid_fraud_rate_known", Template("{v:.0%} of the customer's labelled earlier transactions "
                                      "were fraud", missing="No confirmed labels yet for this "
                                      "customer")),  # fmt: skip
    (r"freq_(?P<col>card1|addr1|P_emaildomain|uid)",
     Template("This {col_text} appeared {v:.0f} times in the training period")),  # fmt: skip
    # --- behavioral view (features/behavioral.py) -----------------------------------------
    ("hist_n_past", Template("Customer has {v:.0f} earlier transactions",
                             when={0: "No earlier transactions for this customer (cold start)"})),  # fmt: skip
    ("hist_days_since_first", Template("Customer first seen {v:.0f} days ago",
                                       when={0: "Customer seen for the first time"})),  # fmt: skip
    ("amt_z", Template("Amount is {v:+.1f} standard deviations from this customer's usual",
                       missing="Too few earlier transactions to compare the amount")),  # fmt: skip
    ("amt_robust_z", Template("Amount is {v:+.1f} robust deviations (median/MAD) from this "
                              "customer's usual", missing="Too few earlier transactions to "
                              "compare the amount")),  # fmt: skip
    ("amt_ratio_median", Template("Amount is {v:.1f}x this customer's usual (median) amount",
                                  missing="No earlier amounts for this customer")),  # fmt: skip
    ("amt_percentile", Template("Amount is above {v:.0%} of this customer's earlier amounts",
                                missing="No earlier amounts for this customer")),  # fmt: skip
    (rf"vel_n_{_W}", Template("{v:.0f} earlier transactions by this customer in the last {w}")),
    (rf"vel_amt_{_W}", Template("Customer spent ${v:,.2f} in the last {w}")),
    ("secs_since_prev", Template("{dur} since this customer's previous transaction",
                                 missing="First transaction seen for this customer")),  # fmt: skip
    ("mean_gap", Template("Customer usually transacts every {dur}",
                          missing="Too few earlier transactions to measure a usual gap")),  # fmt: skip
    ("gap_ratio", Template("Time since the previous transaction is {v:.2f}x the customer's "
                           "usual gap", missing="Too few earlier transactions to measure a "
                           "usual gap")),  # fmt: skip
    ("amt_slope_5", Template("Over the last 5 transactions the amount changes by ${v:+,.2f} "
                             "per step", missing="Too few earlier transactions for a trend")),  # fmt: skip
    (r"new_(?P<e>email|device|addr2|product)",
     Template(when={1: "First time this customer uses this {e_text}",
                    0: "This customer has used this {e_text} before"},
              missing="No {e_text} on this transaction")),  # fmt: skip
    ("n_new_attributes", Template("{v:.0f} of email domain, device, billing country and "
                                  "product are new for this customer")),  # fmt: skip
    (r"ent_uids_7d_(?P<e>card|email|device)",
     Template("{v:.0f} different customers used this {e_text} in the last 7 days")),  # fmt: skip
    ("hour", Template("Transaction at relative hour {v:.0f} (dataset clock, not local time)")),
    ("hour_deviation", Template("{v:.1f} hours from this customer's usual relative hour",
                                missing="No earlier transactions to compare the hour")),  # fmt: skip
    # --- graph view (features/graph.py) ---------------------------------------------------
    (rf"g_deg_tx_{_E}", Template("This {e_text} appeared in {v:.0f} earlier transactions",
                                 missing="No {e_text} on this transaction")),  # fmt: skip
    (rf"g_deg_uid_{_E}", Template("{v:.0f} customers used this {e_text} before",
                                  missing="No {e_text} on this transaction")),  # fmt: skip
    (rf"g_shared_uids_{_E}", Template("{v:.0f} other customers share this {e_text}",
                                      missing="No {e_text} on this transaction")),  # fmt: skip
    (rf"g_fraud_rate_{_E}", Template("{v:.0%} of earlier labelled transactions on this "
                                     "{e_text} were confirmed fraud",
                                     missing="No confirmed labels yet on this {e_text}")),  # fmt: skip
    ("g_shared_nonhub", Template("{v:.0f} other customers share this card or device "
                                 "(shared terminals and other hubs excluded)",
                                 when={0: "No other customer shares this card or device"},
                                 missing="No card or device on this transaction")),  # fmt: skip
    ("g_twohop_fraud", Template("Linked through shared entities to {v:.0f} confirmed-fraud "
                                "transactions (counted once per shared entity)")),  # fmt: skip
    ("g_comp_tx", Template("Its linked group (shared customer, card, device) has {v:.0f} "
                           "recent transactions")),  # fmt: skip
    ("g_comp_uids", Template("Its linked group has {v:.0f} customers")),
    ("g_comp_fraud_rate", Template("{v:.0%} of labelled transactions in its linked group were "
                                   "confirmed fraud", missing="No confirmed labels yet in its "
                                   "linked group")),  # fmt: skip
    ("g_comm_tx", Template("Its community of linked entities has {v:.0f} recent transactions")),
    ("g_comm_fraud_rate", Template("{v:.0%} of labelled transactions in its community were "
                                   "confirmed fraud", missing="No confirmed labels yet in its "
                                   "community")),  # fmt: skip
    # --- anomaly view (views/anomaly.py), scores rank-normalised on training --------------
    ("anomaly_if", Template("More unusual than {v:.0%} of training transactions (isolation "
                            "forest on legitimate patterns)")),  # fmt: skip
    ("anomaly_ae", Template("Harder to reconstruct than {v:.0%} of training transactions "
                            "(autoencoder on legitimate patterns)")),  # fmt: skip
    ("anomaly_uid_if", Template("More unusual for this customer than {v:.0%} of training "
                                "scores (per-customer isolation forest)",
                                missing="Too few earlier transactions for a per-customer "
                                "anomaly score")),  # fmt: skip
]

_COMPILED = [(re.compile(pattern), template) for pattern, template in LIBRARY]
_COL_TEXT = {"card1": "card number field", "addr1": "billing region", "uid": "customer",
             "P_emaildomain": "email domain"}  # fmt: skip


def lookup(feature: str) -> tuple[Template, dict[str, str]]:
    """Template for a feature and its pattern groups; KeyError if the library has none."""
    for pattern, template in _COMPILED:
        m = pattern.fullmatch(feature)
        if m:
            groups = {k: v for k, v in m.groupdict().items() if v is not None}
            if "w" in groups:
                groups["w"] = WINDOW_TEXT[groups["w"]]
            if "e" in groups:
                groups["e_text"] = ENTITY_TEXT[groups["e"]]
            if "col" in groups:
                groups["col_text"] = _COL_TEXT[groups["col"]]
            return template, groups
    raise KeyError(f"no reason-code template for feature {feature!r}")


def _is_missing(value) -> bool:
    if value is None:
        return True
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return False


def describe(feature: str, value, points: float = 0.0) -> str:
    """The sentence for one feature value. `points` only selects up/down wording."""
    template, groups = lookup(feature)
    fields: dict = {"feature": feature, **groups}
    if _is_missing(value):
        return template.missing.format(**fields)
    fields["raw"] = value
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = math.nan
    fields.update(v=v, top=1 - v, dur=duration(v) if not math.isnan(v) else "")
    if template.when is not None and v in template.when:
        return template.when[v].format(**fields)
    if points > 0 and template.up:
        return template.up.format(**fields)
    if points <= 0 and template.down:
        return template.down.format(**fields)
    return (template.text or template.up or "").format(**fields)


@dataclass(frozen=True)
class Reason:
    feature: str
    text: str
    points: float  # signed; + raises risk

    def render(self) -> str:
        return f"{self.text} ({self.points:+.0f} pts)"


def to_points(shap_logodds: float) -> float:
    return float(shap_logodds) * POINTS_PER_LOGODDS


def top_reasons(
    shap_values: Mapping[str, float],
    values: Mapping[str, object],
    k: int = 5,
    direction: str = "both",
    describe_fn: Callable[[str, object, float], str] = describe,
) -> list[Reason]:
    """The k features with the largest |SHAP| (direction 'up': only risk-raising ones)."""
    if direction not in ("both", "up"):
        raise ValueError("direction must be 'both' or 'up'")
    items = [(f, to_points(phi)) for f, phi in shap_values.items() if phi == phi]
    if direction == "up":
        items = [(f, p) for f, p in items if p > 0]
    items.sort(key=lambda fp: (-abs(fp[1]), fp[0]))
    return [Reason(f, describe_fn(f, values.get(f), p), p) for f, p in items[:k]]
