"""E15: explanation faithfulness, stability, seed consistency (D121 section 1; D158-D163).

Explained systems (the frozen final-run models, refitted by stage2/system.py):
  MVAF  feature level: a_f = sum_v w_v x s_{v,f}, TreeSHAP log-odds of each XGBoost view model
        (xgboost's exact pred_contribs) weighted by the 5-seed mean gate weight w_v; risk =
        MVAF's 5-seed mean score, rescored end to end (views, calibrators, gate) with every
        view keeping the original row's availability (D160)
  B5    TreeSHAP averaged over its 5 seeds; risk = the 5-seed mean score

  1.1 deletion   top-k features (k = 1..10, by |attribution|, D121) set to their training-period
                 medians; AOPC per alert = mean over k of risk(original) - risk(top-k replaced);
                 comparators: random-k (the first k of a random feature order, 20 orders per
                 alert, seed 0) and the global top-k by permutation importance (validation,
                 5 repeats); H6a / H6b: paired bootstrap over alerts, Holm within each system
  1.2 views      MVAF: mask the top-1 / top-2 views by |c_v| (c_v = w_v x logit p_v) vs one
                 random available view; drop in risk (alerts with enough views, D160)
  1.3 insertion  from all features at their medians, put the top-k real values back;
                 mean over k of risk(top-k inserted) - risk(all medians)
  1.4 stability  amount x0.99 / x1.01 / x0.95 / x1.05 with every amount-derived input recomputed
                 (stage2/amount.py); top-3 Jaccard and share of alerts whose top-1 changes
  1.5 seeds      top-5 Jaccard over the 10 seed pairs (MVAF: gate seeds; B5: model seeds)

A secondary deletion curve ranks by positive attribution (D59's order), labelled secondary.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from vaultic.fusion.final_eval import _ci, _p_two_sided, holm_nan

KS = tuple(range(1, 11))
RANDOM_ORDERS = 20
PERM_REPEATS = 5
PERTURBATIONS = (0.99, 1.01, 0.95, 1.05)
N_BOOT = 1000


def tree_shap(model, frame: pd.DataFrame) -> np.ndarray:
    """Exact TreeSHAP log-odds contributions of an XGBoost classifier (no bias column)."""
    from xgboost import DMatrix

    booster = model.get_booster()
    contrib = booster.predict(DMatrix(frame, missing=np.nan), pred_contribs=True)
    return np.asarray(contrib)[:, :-1]


class MvafExplained:
    """MVAF explained at feature level; `columns` is the union of the five views' inputs."""

    name = "MVAF"

    def __init__(self, system):
        from vaultic.views.orchestrate import VIEWS

        self.system, self.views = system, VIEWS
        cols = []
        for v in VIEWS:
            cols += [c for c in system.views[v].columns if c not in cols]
        self.columns = cols

    def shap_by_view(self, frame, avail) -> dict[str, np.ndarray]:
        """view -> (n, len(view columns)) contributions; 0 where the view is missing."""
        out = {}
        for j, v in enumerate(self.views):
            view = self.system.views[v]
            s = np.zeros((len(frame), len(view.columns)))
            rows = np.asarray(avail)[:, j]
            if view.model_ is not None and rows.any():
                s[rows] = tree_shap(view.model_, frame.loc[rows, view.columns])
            out[v] = s
        return out

    def combine(self, shap: dict, weights: np.ndarray) -> np.ndarray:
        a = np.zeros((weights.shape[0], len(self.columns)))
        pos = {c: i for i, c in enumerate(self.columns)}
        for j, v in enumerate(self.views):
            idx = [pos[c] for c in self.system.views[v].columns]
            a[:, idx] += np.nan_to_num(weights[:, j])[:, None] * shap[v]
        return a

    def attributions(self, frame, context, avail, per_seed: bool = False):
        shap = self.shap_by_view(frame, avail)
        p = self.system.calibrated(self.system.raw(frame, avail))
        if per_seed:
            return [self.combine(shap, w) for w in self.system.gate_weights(p, context, True)]
        return self.combine(shap, self.system.gate_weights(p, context))

    def risk(self, frame, context, avail) -> np.ndarray:
        return self.system.score(frame, context, avail)["risk"]

    def raw_cache(self, frame, avail) -> np.ndarray:
        return self.system.raw(frame, avail)

    def risk_changed(self, frame, context, avail, raw, column: str) -> np.ndarray:
        """Risk after `column` changed in `frame`, rescoring only the views that use it."""
        new = raw.copy()
        for j, v in enumerate(self.views):
            view = self.system.views[v]
            rows = np.asarray(avail)[:, j]
            if column in view.columns and view.model_ is not None and rows.any():
                new[rows, j] = view.model_.predict_proba(frame.loc[rows, view.columns])[:, 1]
        return self.system.fused(self.system.calibrated(new), context)[0]


class B5Explained:
    name = "B5"

    def __init__(self, system):
        self.system, self.columns = system, list(system.b5_columns)

    def attributions(self, frame, context=None, avail=None, per_seed: bool = False):
        per = [tree_shap(m, frame[self.columns]) for m in self.system.b5_models]
        return per if per_seed else np.mean(per, axis=0)

    def risk(self, frame, context=None, avail=None) -> np.ndarray:
        return self.system.b5(frame)[0]


# ---- replacement curves ------------------------------------------------------------------


def order_by(attr: np.ndarray, rank_by: str = "absolute") -> np.ndarray:
    a = np.asarray(attr, dtype=float)
    key = -np.abs(a) if rank_by == "absolute" else -a
    return np.argsort(key, axis=1, kind="stable")


def _replace(frame: pd.DataFrame, columns, order, k, source: pd.DataFrame) -> pd.DataFrame:
    """Copy of frame with each row's first k features of `order` taken from `source`."""
    out = frame.copy()
    X, S = out[columns].to_numpy(dtype=float, copy=True), source[columns].to_numpy(dtype=float)
    rows = np.arange(len(out))[:, None]
    top = order[:, :k]
    X[rows, top] = S[rows, top]
    out[list(columns)] = X.astype(np.float32)
    return out


def median_frame(frame: pd.DataFrame, columns, medians: pd.Series) -> pd.DataFrame:
    out = frame.copy()
    med = medians.reindex(list(columns)).to_numpy(dtype=float)
    out[list(columns)] = np.broadcast_to(med, (len(out), len(columns))).astype(np.float32)
    return out


def deletion_aopc(
    expl, frame, context, avail, medians, order, ks=KS
) -> tuple[np.ndarray, np.ndarray]:
    """Per alert: AOPC = mean over k of risk(original) - risk(top-k at medians); also the
    (len(ks), n) risk curve."""
    base = median_frame(frame, expl.columns, medians)
    r0 = expl.risk(frame, context, avail)
    curve = np.array([expl.risk(_replace(frame, expl.columns, order, k, base), context, avail)
                      for k in ks])  # fmt: skip
    return (r0[None, :] - curve).mean(axis=0), curve


def insertion_gain(expl, frame, context, avail, medians, order, ks=KS) -> np.ndarray:
    """Per alert: mean over k of risk(top-k real values inserted) - risk(all at medians)."""
    base = median_frame(frame, expl.columns, medians)
    r_med = expl.risk(base, context, avail)
    curve = np.array([expl.risk(_replace(base, expl.columns, order, k, frame), context, avail)
                      for k in ks])  # fmt: skip
    return (curve - r_med[None, :]).mean(axis=0)


def random_orders(n: int, d: int, draws: int = RANDOM_ORDERS, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [np.argsort(rng.random((n, d)), axis=1) for _ in range(draws)]


def global_order(importance, n: int) -> np.ndarray:
    order = np.argsort(-np.asarray(importance, dtype=float), kind="stable")
    return np.tile(order, (n, 1))


def paired_mean_test(a, b, n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """Mean of a - b over alerts with a bootstrap 95% CI and two-sided p (resampling alerts)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    lo, hi = _ci(boots)
    return {"diff": float(d.mean()), "ci_low": lo, "ci_high": hi, "p_value": _p_two_sided(boots),
            "n": int(len(d))}  # fmt: skip


def mean_ci(x, n_boot: int = N_BOOT, seed: int = 0) -> dict:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"mean": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n": 0}
    rng = np.random.default_rng(seed)
    lo, hi = _ci([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n_boot)])
    return {"mean": float(x.mean()), "ci_low": lo, "ci_high": hi, "n": int(len(x))}


def faithfulness(expl, frame, context, avail, medians, importance, n_boot: int = N_BOOT,
                 seed: int = 0, log=print) -> dict:  # fmt: skip
    """1.1 deletion with H6a / H6b, 1.3 insertion, and the secondary positive-order deletion."""
    attr = expl.attributions(frame, context, avail)
    n, d = attr.shape
    order = order_by(attr, "absolute")
    a_attr, curve = deletion_aopc(expl, frame, context, avail, medians, order)
    log(f"{expl.name}: deletion done")
    rand = random_orders(n, d, seed=seed)
    a_rand = np.mean([deletion_aopc(expl, frame, context, avail, medians, o)[0] for o in rand], 0)
    log(f"{expl.name}: random deletion done")
    perm = global_order(importance, n)
    a_perm = deletion_aopc(expl, frame, context, avail, medians, perm)[0]
    h = {
        "H6a": paired_mean_test(a_attr, a_rand, n_boot, seed),
        "H6b": paired_mean_test(a_attr, a_perm, n_boot, seed),
    }
    for key, p in zip(h, holm_nan([h[k]["p_value"] for k in h]), strict=True):
        h[key]["p_holm"] = p
        h[key]["supported"] = bool(h[key]["ci_low"] > 0 and p < 0.05)
    ins = {
        "attribution": insertion_gain(expl, frame, context, avail, medians, order),
        "permutation": insertion_gain(expl, frame, context, avail, medians, perm),
    }
    ins["random"] = np.mean([insertion_gain(expl, frame, context, avail, medians, o)
                             for o in rand], axis=0)  # fmt: skip
    log(f"{expl.name}: insertion done")
    a_pos = deletion_aopc(expl, frame, context, avail, medians, order_by(attr, "positive"))[0]
    return {"aopc": {"attribution": mean_ci(a_attr), "random": mean_ci(a_rand),
                     "permutation": mean_ci(a_perm),
                     "attribution_positive_order (secondary)": mean_ci(a_pos)},
            "hypotheses": h,
            "deletion_curve": curve.mean(axis=1).tolist(),
            "insertion": {k: mean_ci(v) for k, v in ins.items()},
            "insertion_tests": {f"attribution - {k}": paired_mean_test(ins["attribution"], ins[k],
                                                                       n_boot, seed)
                                for k in ("random", "permutation")}}  # fmt: skip


# ---- views, stability, seeds ---------------------------------------------------------------


def view_deletion(system, frame, context, avail, seed: int = 0) -> dict:
    """1.2: drop in MVAF risk when the top-1 / top-2 views by |c_v| or one random available
    view is masked. Top-1 and random need >= 2 available views, top-2 needs >= 3 (D160)."""
    from vaultic.explain.view_contrib import view_contributions

    s = system.score(frame, context, avail)
    p, risk = s["p"], s["risk"]
    vc = view_contributions(s["weights"], p, list(range(p.shape[1])), risk).contribution
    n_av = (~np.isnan(p)).sum(axis=1)
    order = np.argsort(-np.nan_to_num(np.abs(vc), nan=-1.0), axis=1, kind="stable")
    rng = np.random.default_rng(seed)

    def drop(mask_cols, rows):
        q = p[rows].copy()
        for i, cols in enumerate(mask_cols):
            q[i, cols] = np.nan
        return risk[rows] - system.fused(q, context[rows])[0]

    out = {}
    for name, need, cols_of in (
        ("top-1 view", 2, lambda i: order[i, :1]),
        ("top-2 views", 3, lambda i: order[i, :2]),
        ("random view", 2, lambda i: [rng.choice(np.flatnonzero(~np.isnan(p[i])))]),
    ):
        rows = np.flatnonzero(n_av >= need)
        out[name] = drop([cols_of(i) for i in rows], rows) if len(rows) else np.array([])
    res = {k: mean_ci(v) for k, v in out.items()}
    for k in ("top-1 view", "top-2 views"):
        rows1 = np.flatnonzero(n_av >= (2 if k == "top-1 view" else 3))
        rows_r = np.flatnonzero(n_av >= 2)
        common = np.isin(rows_r, rows1)
        if len(out[k]) and common.any():
            res[f"{k} - random view"] = paired_mean_test(out[k], out["random view"][common])
    return res


def top_jaccard(a, b, k: int) -> np.ndarray:
    from vaultic.explain.faithfulness import jaccard_top_k

    return jaccard_top_k(a, b, k)


def stability(expl, attr0, perturbed: dict) -> dict:
    """1.4: perturbed = factor -> attributions of the perturbed rows."""
    top1 = order_by(attr0)[:, 0]
    out = {}
    for f, attr in perturbed.items():
        out[f"x{f}"] = {"top3_jaccard": mean_ci(top_jaccard(attr0, attr, 3)),
                        "top1_changed": mean_ci((order_by(attr)[:, 0] != top1).astype(float))}  # fmt: skip
    return out


def seed_consistency(per_seed: list[np.ndarray], k: int = 5) -> dict:
    """1.5: per alert, the mean top-k Jaccard over the seed pairs; mean with CI over alerts."""
    pairs = list(combinations(range(len(per_seed)), 2))
    per_alert = np.mean([top_jaccard(per_seed[i], per_seed[j], k) for i, j in pairs], axis=0)
    return {"pairs": len(pairs), **mean_ci(per_alert)}


def permutation_importance(expl, frame, context, avail, y, repeats: int = PERM_REPEATS,
                           seed: int = 0, log=None) -> np.ndarray:  # fmt: skip
    """Mean drop in PR-AUC over `repeats` permutations of each feature column (validation)."""
    from vaultic.eval.metrics import pr_auc

    rng = np.random.default_rng(seed)
    base = pr_auc(y, expl.risk(frame, context, avail))
    raw = expl.raw_cache(frame, avail) if hasattr(expl, "raw_cache") else None
    imp = np.zeros(len(expl.columns))
    for j, c in enumerate(expl.columns):
        drops = []
        original = frame[c].to_numpy()
        for _ in range(repeats):
            f = frame.copy()
            f[c] = original[rng.permutation(len(f))]
            r = (expl.risk_changed(f, context, avail, raw, c) if raw is not None
                 else expl.risk(f, context, avail))  # fmt: skip
            drops.append(base - pr_auc(y, r))
        imp[j] = float(np.mean(drops))
        if log and j % 50 == 0:
            log(f"{expl.name}: permutation importance {j + 1}/{len(expl.columns)}")
    return imp
