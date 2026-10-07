"""Phase 9.3 faithfulness and stability of explanations (research side; no explainer runs here).

Every function takes attributions from any explainer (SHAP, gate weights, ...) as an
(n_rows, n_features) array and a `predict` function mapping an (n, d) array to fraud
probabilities.

- deletion: replace each row's top-k features (by |attribution|) with baseline values
  (training medians) for k = 0..K and record the mean risk; a faithful explanation makes the
  risk drop fast, so a LOWER area under the deletion curve is better.
- insertion: start from the baseline and put the top-k real values back; HIGHER area is better.
- baselines: the same curves with a random feature order (averaged over repeats) and with
  one global order from permutation importance.
- stability: top-k Jaccard overlap between explanations of near-identical transactions;
  seed consistency: mean pairwise top-k Jaccard across models trained with different seeds.
- counterfactuals: validity (risk really falls below the threshold), proximity (L1 distance
  in units of each feature's scale), sparsity (features changed), actionability (only allowed
  features changed).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from itertools import combinations

import numpy as np

Predict = Callable[[np.ndarray], np.ndarray]


def _ranking(attributions: np.ndarray) -> np.ndarray:
    """Per row, feature indices from most to least important (|attribution|, stable on ties)."""
    return np.argsort(-np.abs(np.asarray(attributions, dtype=float)), axis=1, kind="stable")


def _curve(predict: Predict, X, attributions, baseline, max_k: int, delete: bool) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    baseline = np.broadcast_to(np.asarray(baseline, dtype=float), X.shape)
    order = _ranking(attributions)
    rows = np.arange(len(X))[:, None]
    points = []
    for k in range(max_k + 1):
        top = order[:, :k]
        if delete:
            current = X.copy()
            current[rows, top] = baseline[rows, top]
        else:
            current = baseline.copy()
            current[rows, top] = X[rows, top]
        points.append(float(np.mean(predict(current))))
    return np.array(points)


def deletion_curve(predict: Predict, X, attributions, baseline, max_k: int) -> np.ndarray:
    return _curve(predict, X, attributions, baseline, max_k, delete=True)


def insertion_curve(predict: Predict, X, attributions, baseline, max_k: int) -> np.ndarray:
    return _curve(predict, X, attributions, baseline, max_k, delete=False)


def curve_auc(curve: np.ndarray) -> float:
    """Area under a curve over k = 0..K, with k rescaled to [0, 1]."""
    curve = np.asarray(curve, dtype=float)
    if len(curve) < 2:
        return float(curve[0]) if len(curve) else float("nan")
    dx = 1.0 / (len(curve) - 1)
    return float(np.sum((curve[1:] + curve[:-1]) * dx / 2))


def random_attributions(shape: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    return rng.random(shape)


def faithfulness_report(
    predict: Predict,
    X,
    attributions,
    baseline,
    global_importance,
    max_k: int,
    repeats: int = 10,
    seed: int = 0,
) -> dict[str, float]:
    """Deletion and insertion AUCs for the explanation, random order and permutation order."""
    X = np.asarray(X, dtype=float)
    rng = np.random.default_rng(seed)
    perm = np.tile(np.asarray(global_importance, dtype=float), (len(X), 1))
    rand_del, rand_ins = [], []
    for _ in range(repeats):
        r = random_attributions(X.shape, rng)
        rand_del.append(curve_auc(deletion_curve(predict, X, r, baseline, max_k)))
        rand_ins.append(curve_auc(insertion_curve(predict, X, r, baseline, max_k)))
    return {
        "deletion_auc": curve_auc(deletion_curve(predict, X, attributions, baseline, max_k)),
        "deletion_auc_random": float(np.mean(rand_del)),
        "deletion_auc_permutation": curve_auc(deletion_curve(predict, X, perm, baseline, max_k)),
        "insertion_auc": curve_auc(insertion_curve(predict, X, attributions, baseline, max_k)),
        "insertion_auc_random": float(np.mean(rand_ins)),
        "insertion_auc_permutation": curve_auc(insertion_curve(predict, X, perm, baseline, max_k)),
    }


def jaccard_top_k(attr_a, attr_b, k: int = 3) -> np.ndarray:
    """Per row, |top-k(a) & top-k(b)| / |top-k(a) | top-k(b)|."""
    a, b = _ranking(attr_a)[:, :k], _ranking(attr_b)[:, :k]
    out = np.empty(len(a))
    for i, (x, y) in enumerate(zip(a, b, strict=True)):
        sx, sy = set(x.tolist()), set(y.tolist())
        out[i] = len(sx & sy) / len(sx | sy)
    return out


def seed_consistency(attributions_per_seed: Sequence[np.ndarray], k: int = 5) -> float:
    """Mean over rows and seed pairs of the top-k Jaccard overlap (1 = identical top-k)."""
    pairs = list(combinations(range(len(attributions_per_seed)), 2))
    if not pairs:
        return float("nan")
    scores = [
        jaccard_top_k(attributions_per_seed[i], attributions_per_seed[j], k).mean()
        for i, j in pairs
    ]
    return float(np.mean(scores))


def counterfactual_quality(
    predict: Predict,
    X,
    X_cf,
    threshold: float,
    scale,
    actionable,
    atol: float = 1e-9,
) -> dict[str, float]:
    """Quality of counterfactuals X_cf proposed for rows X (same shape)."""
    X, X_cf = np.asarray(X, dtype=float), np.asarray(X_cf, dtype=float)
    changed = ~np.isclose(X, X_cf, atol=atol, rtol=0)
    actionable = np.asarray(actionable, dtype=bool)
    distance = np.abs(X - X_cf) / np.asarray(scale, dtype=float)
    return {
        "validity": float(np.mean(predict(X_cf) < threshold)),
        "proximity": float(distance.sum(axis=1).mean()),
        "sparsity": float(changed.sum(axis=1).mean()),
        "actionable_share": float(np.mean(~(changed & ~actionable).any(axis=1))),
    }
