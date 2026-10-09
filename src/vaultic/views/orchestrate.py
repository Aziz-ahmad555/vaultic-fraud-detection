"""Out-of-sample view predictions for the fusion gate (roadmap 7.3).

For every fold of a plan (views/plan.py) each view is trained on the fold's training days and
predicts the fold's later block; the result is one table, one row per predicted transaction:

  TransactionID, TransactionDT, day, fold, role, split, label
  raw_<view>   the view model's own fraud probability, NaN where the view is missing (rule 11)
  m_<view>     availability mask (1 / 0)
  ctx_*        context for the gate: log amount, history length, has_identity, ProductCD code
               (shared training-period encoder, D37), relative hour

for the five views tabular, behavioral, temporal, graph, anomaly. `calibrate_views` then fits
one calibrator per view on the calibrate_views rows only (the earlier slice of the calibrate
tail, D53/D76; the later calibrate_fused slice is kept for fused-score calibration, conformal,
thresholds and routing lambdas) and adds, for every row,
  p_<view>     calibrated probability (NaN where missing)
  c_<view>     confidence |2p - 1| from the CALIBRATED p (0 where missing)
  disagreement std of the available CALIBRATED probabilities (0 with fewer than two)
using exactly MVAF's definitions (fusion.mvaf.view_inputs). Confidence and disagreement are
never computed from raw scores, whose scales differ between views (D56).

Views that need another environment (the GRU in .venv-torch) are external: they run as a
separate step (views/temporal_step.py) that writes a parquet file of TransactionID, fold,
p_<view>, which `view_table` joins. A view that is neither given nor external is missing for
every row and says so (all NaN, mask 0).

"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from vaultic.features.categories import CategoryEncoder
from vaultic.fusion.mvaf import view_inputs
from vaultic.trust.calibration import choose_calibrator
from vaultic.views.plan import Fold

VIEWS = ("tabular", "behavioral", "temporal", "graph", "anomaly")
CONTEXT = ("ctx_log_amount", "ctx_hist_n_past", "ctx_has_identity", "ctx_product", "ctx_hour")


# ---- availability rules ----------------------------------------------------------------------


def has_history(frame: pd.DataFrame) -> np.ndarray:
    """Behavioral view: needs at least one earlier transaction of the uid."""
    return frame["hist_n_past"].to_numpy(dtype=float) > 0


def has_graph_evidence(frame: pd.DataFrame) -> np.ndarray:
    """Graph view: a non-hub card or device of the transaction was used by another uid before t
    (g_shared_nonhub > 0; hub threshold fitted on the training period, D52). An entity seen
    before only by the same customer, or only hubs, is not relational evidence."""
    if "g_shared_nonhub" not in frame:
        raise ValueError(
            "graph availability needs g_shared_nonhub: build graph features with hub_thresholds"
        )
    return frame["g_shared_nonhub"].fillna(0).to_numpy(dtype=float) > 0


# ---- in-process views ------------------------------------------------------------------------


class SupervisedView:
    """A classifier (views.tabular.make_model) on a fixed set of feature columns, trained only
    on rows where the view is available and scoring only those."""

    def __init__(
        self,
        columns: list[str],
        model: str = "xgboost",
        params: dict | None = None,
        available: Callable[[pd.DataFrame], np.ndarray] | None = None,
        seed: int = 0,
    ):
        self.columns, self.model, self.params = list(columns), model, dict(params or {})
        self.available, self.seed = available, seed

    def _mask(self, frame: pd.DataFrame) -> np.ndarray:
        return np.ones(len(frame), bool) if self.available is None else self.available(frame)

    def fit(self, frame: pd.DataFrame, y: np.ndarray) -> SupervisedView:
        from vaultic.views.tabular import make_model

        rows = self._mask(frame)
        if len(np.unique(y[rows])) < 2:
            raise ValueError("the view's training rows need both classes")
        self.model_ = make_model(self.model, self.params, self.seed)
        self.model_.fit(frame.loc[rows, self.columns], y[rows])
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        out = np.full(len(frame), np.nan)
        rows = self._mask(frame)
        if rows.any():
            out[rows] = self.model_.predict_proba(frame.loc[rows, self.columns])[:, 1]
        return out


class AnomalyScoreView:
    """Phase 6 anomaly view as a probability (D57). The fold's training rows (time-ordered) are
    split by time: the anomaly models (views/anomaly.py, legit rows only) are fitted on the
    earlier `1 - link_fraction`, and the logistic link from their scores to a fraud probability
    is fitted on the later `link_fraction`, where those scores are out-of-sample. The per-uid
    score enters with a missing indicator, so rows without enough history still get a score
    from the global models."""

    def __init__(self, columns: list[str], uid_col: str = "uid", n_past_col: str = "hist_n_past",
                 encoder: CategoryEncoder | None = None, seed: int = 0, link_fraction: float = 0.2,
                 **ae_kwargs):  # fmt: skip
        if not 0 < link_fraction < 1:
            raise ValueError("link_fraction must be in (0, 1)")
        self.columns, self.uid_col, self.n_past_col = list(columns), uid_col, n_past_col
        self.encoder, self.seed, self.ae_kwargs = encoder, seed, ae_kwargs
        self.link_fraction = link_fraction

    def _scores(self, frame: pd.DataFrame) -> np.ndarray:
        s = self.view_.transform(frame[self.columns], frame[self.uid_col].to_numpy(),
                                 frame[self.n_past_col].to_numpy())  # fmt: skip
        uid = s["anomaly_uid_if"].to_numpy()
        return np.column_stack([s["anomaly_if"], s["anomaly_ae"], np.nan_to_num(uid, nan=0.5),
                                np.isnan(uid)])  # fmt: skip

    def fit(self, frame: pd.DataFrame, y: np.ndarray) -> AnomalyScoreView:
        from vaultic.views.anomaly import AnomalyView

        if "TransactionDT" in frame and np.any(np.diff(frame["TransactionDT"].to_numpy()) < 0):
            raise ValueError("training rows must be in time order")
        y = np.asarray(y)
        cut = int(round(len(frame) * (1 - self.link_fraction)))
        early, tail = frame.iloc[:cut], frame.iloc[cut:]
        if len(np.unique(y[cut:])) < 2:
            raise ValueError(
                "the link tail needs both classes: use a longer fold or a larger link_fraction"
            )
        self.view_ = AnomalyView(seed=self.seed, encoder=self.encoder, **self.ae_kwargs)
        self.view_.fit(early[self.columns], y[:cut], early[self.uid_col].to_numpy())
        self.link_ = LogisticRegression(class_weight="balanced").fit(self._scores(tail), y[cut:])
        self.n_anomaly_rows_, self.n_link_rows_ = len(early), len(tail)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self.link_.predict_proba(self._scores(frame))[:, 1]


# ---- the table -------------------------------------------------------------------------------


def context(df: pd.DataFrame, features: pd.DataFrame, encoder: CategoryEncoder) -> pd.DataFrame:
    has_id = df["has_identity"] if "has_identity" in df else pd.Series(0, index=df.index)
    return pd.DataFrame(
        {
            "ctx_log_amount": np.log1p(df["TransactionAmt"].to_numpy(dtype=float)),
            "ctx_hist_n_past": features["hist_n_past"].to_numpy(dtype=float),
            "ctx_has_identity": has_id.to_numpy(dtype=float),
            "ctx_product": encoder.encode(df["ProductCD"], "ProductCD").astype(float),
            "ctx_hour": ((df["TransactionDT"].to_numpy() // 3600) % 24).astype(float),
        },
        index=df.index,
    )


def _check_external(name: str, ext: pd.DataFrame, plan: list[Fold], wanted: pd.DataFrame) -> None:
    col = f"p_{name}"
    missing_cols = {"TransactionID", "fold", col} - set(ext.columns)
    if missing_cols:
        raise ValueError(f"external view {name} lacks columns {sorted(missing_cols)}")
    unknown = set(ext["fold"]) - {f.name for f in plan}
    if unknown:
        raise ValueError(f"external view {name} has folds not in the plan: {sorted(unknown)}")
    got = ext.set_index(["TransactionID", "fold"]).index
    need = wanted.set_index(["TransactionID", "fold"]).index
    if not need.isin(got).all():
        raise ValueError(f"external view {name} is missing predictions for some planned rows")


def view_table(
    df: pd.DataFrame,
    features: pd.DataFrame,
    views: Mapping[str, object],
    plan: list[Fold],
    encoder: CategoryEncoder,
    splits=None,
    external: Mapping[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """df: the transactions (sorted by time, with day, isFraud); features: one row per df row
    with every view's feature columns (+ uid, hist_n_past); views: name -> object with
    fit(frame, y) / predict(frame) for the in-process views."""
    external = dict(external or {})
    unknown = (set(views) | set(external)) - set(VIEWS)
    if unknown or set(views) & set(external):
        raise ValueError(f"views must be distinct names among {VIEWS}")
    day = df["day"].to_numpy()
    time = df["TransactionDT"].to_numpy()
    y = df["isFraud"].to_numpy()
    ctx = context(df, features, encoder)

    parts = []
    for fold in plan:
        tr, pr = fold.train_rows(day, time), fold.predict_rows(day)
        part = pd.DataFrame(
            {
                "TransactionID": df["TransactionID"].to_numpy()[pr],
                "TransactionDT": time[pr],
                "day": day[pr],
                "fold": fold.name,
                "role": fold.row_roles(day[pr]),
                "label": y[pr],
            }
        )
        if splits is not None:
            part.insert(5, "split", splits.assign(day[pr]))
        for name in VIEWS:
            part[f"raw_{name}"] = np.nan
        for name, view in views.items():
            view.fit(features.loc[tr].reset_index(drop=True), y[tr])
            part[f"raw_{name}"] = view.predict(features.loc[pr].reset_index(drop=True))
        parts.append(pd.concat([part, ctx.loc[pr].reset_index(drop=True)], axis=1))
    table = pd.concat(parts, ignore_index=True)

    for name, ext in external.items():
        _check_external(name, ext, plan, table[["TransactionID", "fold"]])
        col = f"p_{name}"
        joined = table[["TransactionID", "fold"]].merge(
            ext[["TransactionID", "fold", col]], on=["TransactionID", "fold"], how="left"
        )
        table[f"raw_{name}"] = joined[col].to_numpy()

    for name in VIEWS:
        table[f"m_{name}"] = table[f"raw_{name}"].notna().astype(np.int8)
    return table


def calibrate_views(table: pd.DataFrame, folds: int = 5) -> tuple[pd.DataFrame, dict]:
    """Calibrated p_<view>, then c_<view> and disagreement from them (D56).

    One calibrator per view (Platt or isotonic, chosen by out-of-fold ECE as in D29) fitted ONLY
    on calibrate_views rows (D76) where the view is available, applied to every row. A view
    missing on every row stays missing; a view with too few such rows or one class raises."""
    from vaultic.views.plan import check_calibration_scope

    out = table.copy()
    cal = (out["role"] == "calibrate_views").to_numpy()
    if not cal.any():
        raise ValueError(
            "no calibrate_views rows: the plan needs splits.yaml's calibrate slices (D53, D76)"
        )
    fold = out["fold"].to_numpy()
    info = {}
    p_all = {name: np.full(len(out), np.nan) for name in VIEWS}
    # one calibrator per (fold, view): a fold is one set of view models (D77)
    for f in pd.unique(fold):
        mine = fold == f
        if not (cal & mine).any():
            raise ValueError(
                f"fold {f} has no calibrate_views rows: its views cannot be calibrated"
            )
        for name in VIEWS:
            raw = out[f"raw_{name}"].to_numpy(dtype=float)
            avail = ~np.isnan(raw) & mine
            if not avail.any():
                continue
            fit_rows = cal & avail
            y = out["label"].to_numpy()[fit_rows]
            if fit_rows.sum() < 2 * folds or len(np.unique(y)) < 2:
                raise ValueError(
                    f"cannot calibrate view {name} of fold {f}: needs both classes among enough "
                    "calibrate_views rows"
                )
            check_calibration_scope(fold[fit_rows], fold[avail], f"calibrator of view {name}")
            calibrator, report = choose_calibrator(y, raw[fit_rows], folds=folds)
            p_all[name][avail] = calibrator.predict(raw[avail])
            info[(f, name)] = report
    for name in VIEWS:
        out[f"p_{name}"] = p_all[name]
    mask, _, confidence, disagreement = view_inputs(
        out[[f"p_{v}" for v in VIEWS]].to_numpy(dtype=float)
    )
    for j, name in enumerate(VIEWS):
        out[f"c_{name}"] = confidence[:, j]
    out["disagreement"] = disagreement
    out.attrs["calibrated"] = True
    return out, info


def require_calibrated(table: pd.DataFrame) -> None:
    if not table.attrs.get("calibrated", False):
        raise ValueError("view table is not calibrated: run calibrate_views first (D56)")


def mvaf_inputs(table: pd.DataFrame, role: str | None = None) -> tuple:
    """(views, context, y) arrays for MVAF / the fusion baselines, optionally for one role."""
    require_calibrated(table)
    t = table if role is None else table[table["role"] == role]
    views = t[[f"p_{v}" for v in VIEWS]].to_numpy(dtype=float)
    return views, t[list(CONTEXT)].to_numpy(dtype=float), t["label"].to_numpy()
