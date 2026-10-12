"""E15 1.6: counterfactual validity (D121; D158 (2), D159, D160, D163).

For each alert, DiCE's `random` method (dice-ml 0.12, .venv-dice, D41) proposes changes to the
actionable inputs it may vary: TransactionAmt and ProductCD (D163: the D38 actionable features
whose every dependent input can be recomputed exactly; the email and device fields feed the
graph, entity-count and frequency features, which are not recomputed here). Every other input
is locked; the dependents of a change are recomputed (`apply_change`):

  amount   stage2/amount.py (D158 (1))
  product  the design-matrix code, new_product and n_new_attributes (from the customer's
           earlier products), seq_cur_product_code (temporal current step) and ctx_product (gate
           context), and the anomaly scores (rescored)

DiCE sees a model that returns 1 while the transaction is still an alert and 0 once it is not,
so its stopping threshold is exactly the validity boundary: MVAF, the E12 decision is allow or
monitor (D159, recomputing fused calibration, conformal set and disagreement); B5, the score
falls below B5's frozen cost threshold. Validity is re-checked end to end on every candidate.

Metrics per alert: validity = valid candidates / requested candidates (none returned = 0);
any-valid; for MVAF also the looser D121 reading (the action falls below the alert's own
action); proximity = mean over candidates of the L1 change of the amount / training MAD plus 1
for a changed product; sparsity = inputs changed; actionable share (100% by construction).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from vaultic.stage2.amount import AmountHistory, with_amount

VARY = ("TransactionAmt", "ProductCD")
GRID = {"total_cfs": (3, 5), "sample_size": (1000, 5000), "amount_range": ("below", "p1-p99")}
SELECTION_TIE = 0.01  # D158: validity ties within 0.01 -> proximity, sparsity, cheaper setting


class ProductCodes:
    """ProductCD string <-> its codes: the design matrix's and the training-period encoder's."""

    def __init__(self, df: pd.DataFrame, features: pd.DataFrame, encoder):
        s = df["ProductCD"].astype(str)
        pairs = pd.DataFrame({"p": s, "code": features["ProductCD"].to_numpy()}).dropna()
        self.design = pairs.drop_duplicates("p").set_index("p")["code"].to_dict()
        self.values = sorted(self.design)
        self.encoded = (
            {p: float(encoder.encode(pd.Series([p]), "ProductCD")[0]) for p in self.values}
            if encoder is not None
            else {p: float(self.design[p]) for p in self.values}
        )


class ProductHistory:
    """Each requested row's earlier ProductCD values of the same customer (time < t)."""

    def __init__(self, df: pd.DataFrame, uid, rows):
        time = df["TransactionDT"].to_numpy(dtype=np.int64)
        prod = df["ProductCD"].astype(str).to_numpy()
        uid = np.asarray(uid).astype(str)
        self.seen = []
        frame = pd.DataFrame({"u": uid, "t": time, "p": prod})
        groups = {u: g for u, g in frame.groupby("u", sort=False)}
        for r in np.asarray(rows):
            g = groups[uid[r]]
            self.seen.append(set(g.loc[g["t"] < time[r], "p"]))


def apply_change(system, frame: pd.DataFrame, context: np.ndarray, amounts: AmountHistory,
                 products: ProductHistory, codes: ProductCodes, new_amount, new_product,
                 row_index: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:  # fmt: skip
    """Candidate rows (frame rows `row_index` with amount / product changed) with every
    dependent input recomputed. amounts / products hold the histories of the frame's rows."""
    from vaultic.views.orchestrate import CONTEXT

    sub = frame.iloc[row_index].reset_index(drop=True)
    ctx = np.array(context[row_index], dtype=float, copy=True)
    hist = AmountHistory.__new__(AmountHistory)
    hist.rows = amounts.rows[row_index]
    hist.past = [amounts.past[i] for i in row_index]
    new_product = np.asarray(new_product, dtype=object)
    if "ProductCD" in sub:
        sub["ProductCD"] = np.array([codes.design[p] for p in new_product], dtype=np.float32)
    if "new_product" in sub:
        flag = np.array([float(p not in products.seen[i]) for p, i in zip(new_product, row_index,
                                                                           strict=True)])  # fmt: skip
        sub["new_product"] = flag.astype(np.float32)
        if "n_new_attributes" in sub:
            others = [c for c in ("new_email", "new_device", "new_addr2") if c in sub]
            stacked = sub[others + ["new_product"]].to_numpy(dtype=float)
            total = np.where(np.isnan(stacked).all(axis=1), np.nan, np.nansum(stacked, axis=1))
            sub["n_new_attributes"] = total.astype(np.float32)
    if "seq_cur_product_code" in sub:
        keep = sub["seq_cur_product_code"].notna().to_numpy()
        enc = np.array([codes.encoded[p] for p in new_product], dtype=np.float32)
        sub.loc[keep, "seq_cur_product_code"] = enc[keep]
    ctx[:, list(CONTEXT).index("ctx_product")] = [codes.encoded[p] for p in new_product]
    return with_amount(system, sub, ctx, hist, new_amount)  # also rescores the anomaly scores


class AlertModel:
    """The explained system as DiCE's classifier for ONE alert: predict_proba of a frame of
    (TransactionAmt, ProductCD) candidates is [[1 - alert, alert], ...]."""

    def __init__(self, problem, i: int):
        self.problem, self.i = problem, i

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        a = self.problem.still_alert(self.i, X["TransactionAmt"].to_numpy(float),
                                     X["ProductCD"].astype(str).to_numpy())  # fmt: skip
        return np.column_stack([1 - a, a]).astype(float)


class Problem:
    """Alerts of one explained system ("MVAF" or "B5") with everything needed to rescore them."""

    def __init__(self, system, kind: str, df_rows: pd.DataFrame, frame: pd.DataFrame,
                 context: np.ndarray, avail: np.ndarray, amounts: AmountHistory,
                 products: ProductHistory, codes: ProductCodes):  # fmt: skip
        self.system, self.kind, self.frame = system, kind, frame
        self.context, self.avail, self.amounts, self.products = context, avail, amounts, products
        self.codes = codes
        self.amount = df_rows["TransactionAmt"].to_numpy(float)
        self.product = df_rows["ProductCD"].astype(str).to_numpy()
        self.action = self.actions(np.arange(len(frame)), self.amount, self.product)

    def actions(self, idx, amount, product) -> np.ndarray:
        """MVAF: E12 action codes; B5: 1 if flagged else 0."""
        idx = np.asarray(idx)
        f, ctx = apply_change(self.system, self.frame, self.context, self.amounts,
                              self.products, self.codes, amount, product, idx)  # fmt: skip
        if self.kind == "B5":
            return (self.system.b5(f)[0] >= self.system.b5_threshold).astype(int)
        s = self.system.score(f, ctx, self.avail[idx])
        return self.system.decide(s["risk"], s["p"], np.asarray(amount, float))

    def still_alert(self, i: int, amount, product) -> np.ndarray:
        from vaultic.trust.decision import STEP_UP

        idx = np.full(len(amount), i)
        a = self.actions(idx, amount, product)
        return (a >= 1).astype(int) if self.kind == "B5" else (a >= STEP_UP).astype(int)


def dice_candidates(problem: Problem, i: int, train: pd.DataFrame, setting: dict, seed: int = 0,
                    amount_range=None) -> pd.DataFrame:  # fmt: skip
    """DiCE random candidates for alert i (columns TransactionAmt, ProductCD)."""
    import dice_ml

    data = dice_ml.Data(dataframe=train, continuous_features=["TransactionAmt"],
                        outcome_name="alert")  # fmt: skip
    model = dice_ml.Model(model=AlertModel(problem, i), backend="sklearn")
    explainer = dice_ml.Dice(data, model, method="random")
    query = pd.DataFrame({"TransactionAmt": [problem.amount[i]],
                          "ProductCD": [problem.product[i]]})  # fmt: skip
    try:
        res = explainer.generate_counterfactuals(
            query, total_CFs=setting["total_cfs"], desired_class=0, features_to_vary=list(VARY),
            permitted_range={"TransactionAmt": list(amount_range)} if amount_range else None,
            stopping_threshold=0.5, sample_size=setting["sample_size"], random_seed=seed,
            posthoc_sparsity_algorithm=setting.get("posthoc", "binary"),
        )  # fmt: skip
        cfs = res.cf_examples_list[0].final_cfs_df
    except Exception as e:  # DiCE raises when it finds nothing
        if "No counterfactuals found" not in str(e):
            raise
        cfs = None
    if cfs is None or not len(cfs):
        return pd.DataFrame(columns=list(VARY))
    return cfs[list(VARY)].reset_index(drop=True)


def amount_range_for(setting: dict, amount: float, train_amounts: np.ndarray):
    if setting["amount_range"] == "below":
        return (0.01, max(float(amount), 0.01))
    lo, hi = np.quantile(train_amounts, [0.01, 0.99])
    return (float(lo), float(hi))


def evaluate_alert(problem: Problem, i: int, cands: pd.DataFrame, total: int, mad: float) -> dict:
    """Validity and quality of alert i's candidates, re-checked end to end."""
    from vaultic.trust.decision import STEP_UP

    if not len(cands):
        return {"validity": 0.0, "any_valid": 0.0, "loose_validity": 0.0 if problem.kind == "MVAF"
                else np.nan, "returned": 0, "proximity": np.nan, "sparsity": np.nan,
                "actionable_share": np.nan}  # fmt: skip
    amt = cands["TransactionAmt"].to_numpy(float)
    prod = cands["ProductCD"].astype(str).to_numpy()
    act = problem.actions(np.full(len(cands), i), amt, prod)
    valid = act < (1 if problem.kind == "B5" else STEP_UP)
    loose = act < problem.action[i] if problem.kind == "MVAF" else np.full(len(act), np.nan)
    d_amt = np.abs(amt - problem.amount[i])
    ch_amt, ch_prod = d_amt > 1e-9, prod != problem.product[i]
    return {"validity": float(valid.sum() / total), "any_valid": float(valid.any()),
            "loose_validity": float(np.nansum(loose) / total) if problem.kind == "MVAF" else np.nan,
            "returned": int(len(cands)),
            "proximity": float(np.mean(np.where(ch_amt, d_amt / mad, 0) + ch_prod)),
            "sparsity": float(np.mean(ch_amt.astype(int) + ch_prod.astype(int))),
            "actionable_share": 1.0}  # fmt: skip


def run_setting(problem: Problem, train: pd.DataFrame, setting: dict, mad: float,
                train_amounts: np.ndarray, seed: int = 0, log=None) -> pd.DataFrame:  # fmt: skip
    rows = []
    for i in range(len(problem.frame)):
        rng = amount_range_for(setting, problem.amount[i], train_amounts)
        cands = dice_candidates(problem, i, train, setting, seed, rng)
        rows.append({"alert": i, **evaluate_alert(problem, i, cands, setting["total_cfs"], mad)})
        if log and (i + 1) % 25 == 0:
            log(f"{problem.kind} {setting}: {i + 1}/{len(problem.frame)} alerts")
    return pd.DataFrame(rows)


def grid_settings() -> list[dict]:
    return [{"total_cfs": t, "sample_size": s, "amount_range": r} for t in GRID["total_cfs"]
            for s in GRID["sample_size"] for r in GRID["amount_range"]]  # fmt: skip


def cost_rank(setting: dict) -> tuple:
    """Cheaper first: fewer candidates, smaller sample (the range costs the same)."""
    return (setting["total_cfs"], setting["sample_size"])


def choose_setting(results: list[tuple[dict, pd.DataFrame]]) -> dict:
    """D158's rule: highest mean validity; within SELECTION_TIE of it, lower proximity, then
    fewer changed inputs, then the cheaper setting."""
    summ = [(s, r["validity"].mean(), np.nanmean(r["proximity"]) if r["proximity"].notna().any()
             else np.inf, np.nanmean(r["sparsity"]) if r["sparsity"].notna().any() else np.inf)
            for s, r in results]  # fmt: skip
    best = max(v for _, v, _, _ in summ)
    tied = [x for x in summ if x[1] >= best - SELECTION_TIE]
    tied.sort(key=lambda x: (x[2], x[3], cost_rank(x[0])))
    return tied[0][0]
