"""Phase 9.2 case narrative: one paragraph combining the decision, top reasons, view-level
contributions, graph evidence and a counterfactual.

Template only (no LLM): every sentence comes from a reason-code template or from numbers
passed in, so the narrative cannot state anything that is not in the evidence. A part with no
evidence says so ("No graph evidence ...") instead of being left to the reader's imagination.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from vaultic.explain.reason_codes import Reason, describe

# graph features worth surfacing as relational evidence, in priority order; each is shown only
# when it carries evidence of risk (a positive count or rate)
GRAPH_EVIDENCE = (
    "g_twohop_fraud",
    "g_fraud_rate_device",
    "g_fraud_rate_card",
    "g_fraud_rate_email",
    "g_fraud_rate_address",
    "g_comm_fraud_rate",
    "g_comp_fraud_rate",
    "g_shared_uids_device",
    "g_shared_uids_card",
)


def graph_evidence(features: Mapping[str, object], limit: int = 3) -> list[str]:
    """Sentences for the graph features that point at risk; NaN or 0 is not evidence."""
    out = []
    for name in GRAPH_EVIDENCE:
        v = features.get(name)
        if v is None or (isinstance(v, float) and math.isnan(v)) or float(v) <= 0:
            continue
        out.append(describe(name, v, points=1.0))
        if len(out) == limit:
            break
    return out


@dataclass
class Case:
    transaction_id: int | str
    risk: float  # fused probability in [0, 1]
    decision: str  # e.g. "send to review", "approve", "block"
    reasons: Sequence[Reason] = ()
    views: str | None = None  # view_contrib.describe_row(...)
    graph: Sequence[str] = ()
    counterfactual: str | None = None  # counterfactual.describe(...)
    uncertain: bool | None = None  # conformal set {legit, fraud}


def narrative(case: Case) -> str:
    parts = [f"Transaction {case.transaction_id}: risk {100 * case.risk:.0f}/100, decision: "
             f"{case.decision}."]  # fmt: skip
    if case.uncertain:
        parts.append("The model is uncertain about this case (its conformal set contains both "
                     "classes).")  # fmt: skip
    if case.reasons:
        parts.append("Main reasons: " + "; ".join(r.render() for r in case.reasons) + ".")
    else:
        parts.append("No feature-level reasons are available.")
    if case.views:
        parts.append(f"Evidence by view: {case.views}.")
    if case.graph:
        parts.append("Graph evidence: " + "; ".join(case.graph) + ".")
    else:
        parts.append("No graph evidence of risk (no shared entities with confirmed fraud).")
    if case.counterfactual:
        parts.append(case.counterfactual)
    else:
        parts.append("No valid counterfactual was found with the locked features unchanged.")
    return " ".join(parts)
