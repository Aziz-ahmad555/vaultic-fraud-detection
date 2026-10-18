"""Batch scoring of an uploaded CSV in IEEE-CIS columns (D185).

Each row is scored INDEPENDENTLY against the stored history before its own TransactionDT: rows of
the file are not added to the history, so one row cannot change another's score. The template
holds every IEEE-CIS transaction and identity column (isFraud optional). Required: TransactionID,
TransactionDT, TransactionAmt, ProductCD, card1. Refused, with the reason: unknown columns, a
missing required column, a TransactionDT outside the test period, a non-numeric value in a numeric
column, a category never seen in the data, an isFraud column with an empty cell, more than
MAX_ROWS rows. Missing optional columns are
treated as missing values. The customer (uid) is reconstructed from card1 + addr1 + day - D1 as
in the research pipeline; an unknown key is a new customer.

With an isFraud column: PR-AUC (if both classes occur), recall at the E12 alert (primary) and at
MVAF's E10 cost threshold (0.881), and the confusion matrix of the E12 alert.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd

from vaultic.data.load import SECONDS_PER_DAY

REQUIRED = ("TransactionID", "TransactionDT", "TransactionAmt", "ProductCD", "card1")
DERIVED = ("day", "has_identity")  # computed here, not uploaded
MAX_ROWS = 1000


class BatchError(ValueError):
    """An uploaded file the Test Lab refuses (message shown to the user)."""


def template_columns(store) -> list[str]:
    return [c for c in store.df.columns if c not in DERIVED]


def template_csv(store) -> str:
    return ",".join(template_columns(store)) + "\n"


def parse(store, text: str) -> pd.DataFrame:
    """The uploaded rows as transactions with the stored dtypes; raises BatchError."""
    from vaultic.live.custom import IDENTITY

    try:
        up = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=True)
    except Exception as e:  # malformed CSV
        raise BatchError(f"the file is not a readable CSV: {e}") from e
    if up.empty:
        raise BatchError("the file has no rows")
    if len(up) > MAX_ROWS:
        raise BatchError(f"at most {MAX_ROWS:,} rows per file (got {len(up):,})")
    allowed = set(template_columns(store))
    unknown = [c for c in up.columns if c not in allowed]
    if unknown:
        raise BatchError(f"unknown columns: {unknown[:10]} (use the template's IEEE-CIS columns)")
    missing = [c for c in REQUIRED if c not in up.columns]
    if missing:
        raise BatchError(f"missing required columns: {missing}")
    cols: dict = {}
    bad = []
    for c, dtype in store.df.dtypes.items():
        if c in DERIVED:
            continue
        col = up[c] if c in up else pd.Series([np.nan] * len(up), index=up.index, dtype=object)
        if isinstance(dtype, pd.CategoricalDtype):
            vals = col.where(col.notna(), None)
            unseen = sorted({v for v in vals.dropna() if v not in dtype.categories})
            if unseen:
                bad.append(f"{c}: values never seen in the data {unseen[:5]}")
            cols[c] = pd.Categorical(vals, categories=dtype.categories)
        else:
            num = pd.to_numeric(col, errors="coerce")
            if (col.notna() & num.isna()).any():
                bad.append(
                    f"{c}: non-numeric values in rows {list(np.flatnonzero(col.notna() & num.isna())[:5] + 2)}"
                )
            cols[c] = num
    if bad:
        raise BatchError("; ".join(bad[:8]))
    rows = pd.DataFrame(cols, index=up.index)
    for c in REQUIRED:
        if rows[c].isna().any():
            raise BatchError(f"{c} is empty in rows {list(np.flatnonzero(rows[c].isna())[:5] + 2)}")
    lo, hi = store.splits.test.first, store.splits.test.last
    day = (rows["TransactionDT"] // SECONDS_PER_DAY).astype(int)
    outside = np.flatnonzero((day < lo) | (day > hi))
    if len(outside):
        raise BatchError(f"TransactionDT outside the test period (days {lo}-{hi}) in rows "
                         f"{list(outside[:5] + 2)}")  # fmt: skip
    ident = [c for c in rows.columns if IDENTITY.match(c)]
    rows = rows.assign(day=day, has_identity=rows[ident].notna().any(axis=1).astype("int8"))
    has_label = "isFraud" in up
    if has_label:  # labels are all or nothing: an empty cell is refused, never silently ignored
        empty = np.flatnonzero(rows["isFraud"].isna())
        if len(empty):
            raise BatchError(f"isFraud is empty in rows {list(empty[:5] + 2)}: give every row a "
                             "label (0 or 1) or remove the isFraud column")  # fmt: skip
        if not set(rows["isFraud"].unique()) <= {0, 1}:
            raise BatchError("isFraud must be 0 or 1")
    rows["isFraud"] = rows["isFraud"].fillna(0) if "isFraud" in rows else 0
    rows = rows[store.df.columns].astype(store.df.dtypes.to_dict())
    rows.attrs["has_label"] = bool(has_label)
    rows.attrs["uploaded_columns"] = [c for c in up.columns if c in rows.columns]
    return rows


def resolve_uids(store, rows: pd.DataFrame) -> np.ndarray:
    """Existing uid where the key is known, else a new uid (one per new key in the file)."""
    from vaultic.live.store import uid_key_strings

    keys = uid_key_strings(rows)
    out, new = np.empty(len(rows), dtype=np.int64), {}
    for i, k in enumerate(keys):
        if k in store.key_to_uid:
            out[i] = int(store.key_to_uid[k])
        else:
            if k not in new:
                new[k] = store.new_uid()
            out[i] = new[k]
    return out


STORED, EDITED, NEW = "stored transaction", "edited - not the stored transaction", "new transaction"


def provenance(store, rows: pd.DataFrame) -> list[str]:
    """Per uploaded row: a stored transaction (its TransactionID exists and every uploaded field
    equals the stored value), an edited one (the id exists but any uploaded field differs, the
    label included), or a new one. Only the columns present in the file are compared, so leaving
    an optional column out is not an edit (D186 review #2)."""
    ids = store.df["TransactionID"].to_numpy()
    pos = pd.Series(np.arange(len(ids)), index=ids)
    cols = rows.attrs.get("uploaded_columns", list(rows.columns))
    out = []
    for i in range(len(rows)):
        tid = int(rows["TransactionID"].iat[i])
        if tid not in pos.index:
            out.append(NEW)
            continue
        ref = store.df.iloc[int(pos[tid])]
        same = True
        for c in cols:
            a, b = rows[c].iat[i], ref[c]
            if pd.isna(a) and pd.isna(b):
                continue
            if pd.isna(a) or pd.isna(b):
                same = False
            elif isinstance(rows[c].dtype, pd.CategoricalDtype):
                same = str(a) == str(b)
            else:
                same = np.float32(a) == np.float32(b)
            if not same:
                break
        out.append(STORED if same else EDITED)
    return out


def score_batch(engine, rows: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Every row scored independently; results table and (with labels) metrics."""
    from vaultic.eval.metrics import pr_auc

    uids = resolve_uids(engine.store, rows)
    results = []
    for i in range(len(rows)):
        r = engine.score(rows.iloc[[i]], int(uids[i]), explain=False)
        results.append({"TransactionID": int(rows["TransactionID"].iloc[i]), "uid": int(uids[i]),
                        "risk": r["risk"], "decision": r["decision"], "alert": r["alert"],
                        "flagged_e10": r["flagged_e10"], "conformal_set": r["conformal_set"],
                        "latency_ms": r["latency_ms"]})  # fmt: skip
    table = pd.DataFrame(results)
    table["provenance"] = provenance(engine.store, rows)
    metrics = {"rows": int(len(table)), "has_label": bool(rows.attrs.get("has_label")),
               "edited_rows": int((table["provenance"] == EDITED).sum())}  # fmt: skip
    if metrics["has_label"]:
        y = rows["isFraud"].to_numpy().astype(int)
        table["isFraud"] = y
        alert, flag = table["alert"].to_numpy(), table["flagged_e10"].to_numpy()
        metrics.update({
            "frauds": int(y.sum()),
            "pr_auc": float(pr_auc(y, table["risk"])) if 0 < y.sum() < len(y) else None,
            "recall_e12_alert": float(alert[y == 1].mean()) if y.sum() else None,
            "recall_e10_threshold": float(flag[y == 1].mean()) if y.sum() else None,
            "confusion_e12_alert": {"tp": int((alert & (y == 1)).sum()),
                                    "fp": int((alert & (y == 0)).sum()),
                                    "fn": int((~alert & (y == 1)).sum()),
                                    "tn": int((~alert & (y == 0)).sum())},
        })  # fmt: skip
    return table, metrics
