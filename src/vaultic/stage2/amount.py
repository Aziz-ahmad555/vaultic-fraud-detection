"""The inputs that change with a transaction's own amount (D158 (1)).

When the amount of ONE transaction changes (E15 stability, 1.4; counterfactuals, 1.6), these
inputs of that transaction are recomputed, with the same formulas as the feature pipelines:

  tabular / B5   TransactionAmt, uid_amt_vs_mean                    (features/pipeline.py)
  behavioral     amt_z, amt_robust_z, amt_ratio_median, amt_percentile,
                 amt_slope_5 (the slope includes the current amount) (features/behavioral.py)
  temporal       seq_cur_log_amount                                 (views/temporal_diag.py)
  anomaly        anomaly_if, anomaly_ae, anomaly_uid_if: rescored by the anomaly models that
                 scored the period after training (stage2/system.py); NaN stays NaN
  gate context   ctx_log_amount                                     (views/orchestrate.py)

Everything built from earlier transactions only (uid_amt_mean_past, uid_amt_std_past,
uid_amt_sum_*, vel_amt_*, the history steps seq00-seq09, every graph feature) is unchanged, and
the change does not reach later transactions of the same customer.

tests/test_stage2_amount.py checks that, on a synthetic data set, these values equal the
feature pipelines rebuilt on the data with the amount changed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from vaultic.features.behavioral import MAD_EPS, MAD_SCALE, SLOPE_POINTS

TABULAR = ("TransactionAmt", "uid_amt_vs_mean")
BEHAVIORAL = ("amt_z", "amt_robust_z", "amt_ratio_median", "amt_percentile", "amt_slope_5")
TEMPORAL = ("seq_cur_log_amount",)
ANOMALY = ("anomaly_if", "anomaly_ae", "anomaly_uid_if")
CONTEXT = ("ctx_log_amount",)
FEATURES = TABULAR + BEHAVIORAL + TEMPORAL  # recomputed here; ANOMALY needs the anomaly models


class AmountHistory:
    """Each requested row's earlier amounts of the same customer (time < t, in time order)."""

    def __init__(self, df: pd.DataFrame, uid, rows):
        time = df["TransactionDT"].to_numpy(dtype=np.int64)
        amount = df["TransactionAmt"].to_numpy(dtype=np.float64)
        codes = pd.factorize(np.asarray(uid), use_na_sentinel=False)[0]
        order = np.lexsort((time, codes))  # by customer, then time, ties in row order
        bounds = np.flatnonzero(np.diff(codes[order])) + 1
        groups = {int(codes[g[0]]): g for g in np.split(order, bounds)}
        self.rows = np.asarray(rows)
        self.past = []
        for r in self.rows:
            g = groups[int(codes[r])]
            k = int(np.searchsorted(time[g], time[r], side="left"))
            self.past.append(amount[g[:k]])

    def features(self, new_amount) -> pd.DataFrame:
        """The FEATURES of every requested row, with its amount set to new_amount."""
        new_amount = np.broadcast_to(np.asarray(new_amount, dtype=np.float64), self.rows.shape)
        out = {c: np.full(len(self.rows), np.nan) for c in FEATURES}
        for i, (past, a) in enumerate(zip(self.past, new_amount, strict=True)):
            n = len(past)
            out["TransactionAmt"][i] = a
            out["seq_cur_log_amount"][i] = np.log1p(max(a, 0.0)) if n else np.nan
            if n:
                total = np.cumsum(past)[-1]
                squares = np.cumsum(past**2)[-1]
                mean = total / n
                out["uid_amt_vs_mean"][i] = a / mean
                srt = np.sort(past)
                med = float(np.median(srt))
                mad = float(np.median(np.abs(srt - med)))
                out["amt_ratio_median"][i] = a / med
                below = np.searchsorted(srt, a, side="left")
                equal = np.searchsorted(srt, a, side="right") - below
                out["amt_percentile"][i] = (below + 0.5 * equal) / n
                if n > 1:
                    var = (squares - n * mean**2) / (n - 1)
                    std = np.sqrt(max(var, 0.0))
                    out["amt_z"][i] = (a - mean) / (std if std > 0 else MAD_EPS)
                    out["amt_robust_z"][i] = (a - med) / (MAD_SCALE * mad + MAD_EPS)
            recent = np.append(past[max(0, n - (SLOPE_POINTS - 1)) :], a)
            if len(recent) >= 2:
                x = np.arange(len(recent), dtype=float)
                x -= x.mean()
                out["amt_slope_5"][i] = float((x * recent).sum() / (x * x).sum())
        return pd.DataFrame({c: v.astype(np.float32) for c, v in out.items()})


def with_amount(system, frame: pd.DataFrame, context: np.ndarray, history: AmountHistory,
                new_amount) -> tuple[pd.DataFrame, np.ndarray]:  # fmt: skip
    """frame (the requested rows' features) and gate context with the amount changed and every
    amount-derived input recomputed (D158 (1)); the anomaly scores are rescored where the
    original row had them."""
    from vaultic.views.orchestrate import CONTEXT as CTX

    out = frame.copy()
    new = history.features(new_amount)
    for c in FEATURES:
        if c in out:
            if c == "seq_cur_log_amount":  # the temporal view is missing without history
                keep = out[c].notna().to_numpy()
                out.loc[keep, c] = new.loc[keep, c].to_numpy()
            else:
                out[c] = new[c].to_numpy()
    if system is not None:
        scores = system.anomaly_scores(out)
        for c in ANOMALY:
            if c in out:
                had = out[c].notna().to_numpy()
                out.loc[had, c] = scores.loc[had, c].to_numpy(dtype=np.float32)
    ctx = np.array(context, dtype=float, copy=True)
    ctx[:, list(CTX).index("ctx_log_amount")] = np.log1p(new["TransactionAmt"].to_numpy(float))
    return out, ctx
