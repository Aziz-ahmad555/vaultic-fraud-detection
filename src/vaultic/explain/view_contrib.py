"""Phase 9.1 view-level explanation: how much each evidence view drove the fused score.

MVAF fuses in logit space: logit(p) = sum_v w_v * logit(p_v) + b(x). The contribution of
view v is c_v = w_v * logit(p_v) (signed: + pushes towards fraud), exactly additive with the
gate's context bias b(x). The share shown to the analyst is |c_v| / sum_u |c_u| over the
available views; a missing view has no contribution and no share (rule 11), never 0%.
A view with high weight but a score near 0.5 adds little evidence, and its share says so.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vaultic.fusion.mvaf import _logit

VIEW_TEXT = {
    "tabular": "tabular",
    "behavioral": "behavior",
    "temporal": "sequence",
    "graph": "graph evidence",
    "anomaly": "anomaly",
}


@dataclass
class ViewContributions:
    names: list[str]
    contribution: np.ndarray  # (n, V) signed logit contribution, NaN where the view is missing
    share: np.ndarray  # (n, V) |c| / sum |c| over available views, NaN where missing
    bias: np.ndarray | None  # (n,) gate context bias, when known


def view_contributions(weights, views, names, fused=None) -> ViewContributions:
    """weights: (n, V) gate weights; views: (n, V) view probabilities with NaN where missing;
    fused: optional (n,) fused probabilities, used to recover the gate's bias term."""
    weights = np.asarray(weights, dtype=float)
    views = np.asarray(views, dtype=float)
    if weights.shape != views.shape or views.shape[1] != len(names):
        raise ValueError("weights, views and names must agree in shape")
    mask = ~np.isnan(views)
    c = np.where(mask, weights * _logit(np.where(mask, views, 0.5)), np.nan)
    total = np.nansum(np.abs(c), axis=1, keepdims=True)
    count = np.maximum(mask.sum(axis=1, keepdims=True), 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        # all available views exactly at 0.5: no view drove the score, shares are equal
        share = np.where(mask, np.where(total > 0, np.abs(c) / total, 1.0 / count), np.nan)
    bias = None
    if fused is not None:
        bias = _logit(np.asarray(fused, dtype=float)) - np.nansum(c, axis=1)
    return ViewContributions(list(names), c, share, bias)


def from_mvaf(model, views, names, context=None) -> ViewContributions:
    """Contributions for an MVAF-style model (anything with view_weights / predict_proba)."""
    weights = model.view_weights(views, context)
    return view_contributions(weights, views, names, model.predict_proba(views, context))


def describe_row(vc: ViewContributions, i: int) -> str:
    """'Graph evidence 45% (raises risk), behavior 30% (raises risk), ...; not available: ...'"""
    parts, missing = [], []
    order = np.argsort(-np.nan_to_num(vc.share[i], nan=-1.0), kind="stable")
    for j in order:
        name = VIEW_TEXT.get(vc.names[j], vc.names[j])
        if np.isnan(vc.share[i, j]):
            missing.append(name)
            continue
        effect = "raises risk" if vc.contribution[i, j] > 0 else "lowers risk"
        parts.append(f"{name} {vc.share[i, j]:.0%} ({effect})")
    text = ", ".join(parts) if parts else "no view could score this transaction"
    if missing:
        text += "; not available: " + ", ".join(missing)
    return text[0].upper() + text[1:]
