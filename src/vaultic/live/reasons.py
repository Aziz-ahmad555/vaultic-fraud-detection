"""Reason sentences for the Test Lab (D186): explain/reason_codes.describe, plus honest sentences
for the temporal view's sequence inputs, which the reason-code library has no template for.
Masked IEEE-CIS fields keep their masked wording (CLAUDE.md wording rules)."""

from __future__ import annotations

import re

from vaultic.explain.reason_codes import describe as library_describe

_CHANNEL = {
    "log_amount": "log amount",
    "log_gap_next": "log time gap to the following transaction",
    "product_code": "product code",
    "C1": "masked counter C1",
    "C13": "masked counter C13",
    "D1": "masked time-delta field D1",
    "dist1": "masked distance field dist1",
}
_STEP = re.compile(r"^seq(\d{2})_(.+)$")
_CUR = re.compile(r"^seq_cur_(.+)$")


def _effect(points: float) -> str:
    return "raises the risk score" if points > 0 else "lowers the risk score"


def describe(feature: str, value, points: float = 0.0) -> str:
    try:
        return library_describe(feature, value, points)
    except KeyError:
        pass
    if feature == "seq_n_steps":
        return f"Number of earlier transactions in the temporal view's window {_effect(points)}"
    m = _STEP.match(feature)
    if m:
        step, ch = int(m.group(1)), m.group(2)
        return (f"Temporal view: {_CHANNEL.get(ch, ch)} of an earlier transaction (window step "
                f"{step + 1} of 20) {_effect(points)}")  # fmt: skip
    m = _CUR.match(feature)
    if m:
        ch = m.group(1)
        return f"Temporal view: {_CHANNEL.get(ch, ch)} of this transaction {_effect(points)}"
    return f"Model input {feature} {_effect(points)}"
