"""
global_model.py

Loads the XGBoost model trained on the IEEE-CIS dataset (train_model.py)
and scores live transactions from the Flask app.

IMPORTANT LIMITATION (be upfront about this in your report):
The live app only captures amount, tx_type, timestamp, and device_id per
transaction. The XGBoost model was trained on ~85 IEEE-CIS features
(card type, product code, address, email domain, etc.) that the live app
does not collect. For every feature we can't observe, we substitute the
training-set default (median for numeric columns, mode for one-hot dummy
columns) computed by export_feature_metadata.py.

In practice this means the "global" model's live score is driven almost
entirely by TransactionAmt, since that's the only feature we can actually
fill in with a real, transaction-specific value. This is a legitimate,
documented approximation -- not full utilization of the trained model --
and should be described that way in your report rather than as a full
integration.
"""
import json
import os

import numpy as np
import pandas as pd

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models_store")
MODEL_PATH = os.path.join(MODEL_DIR, "xgboost_baseline.json")
FEATURE_COLUMNS_PATH = os.path.join(MODEL_DIR, "feature_columns.json")
FEATURE_DEFAULTS_PATH = os.path.join(MODEL_DIR, "feature_defaults.json")

_model = None
_feature_columns = None
_feature_defaults = None
_load_error = None


def _load():
    global _model, _feature_columns, _feature_defaults, _load_error

    if _model is not None or _load_error is not None:
        return

    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"{MODEL_PATH} not found")
        if not os.path.exists(FEATURE_COLUMNS_PATH):
            raise FileNotFoundError(
                f"{FEATURE_COLUMNS_PATH} not found -- run export_feature_metadata.py first"
            )
        if not os.path.exists(FEATURE_DEFAULTS_PATH):
            raise FileNotFoundError(
                f"{FEATURE_DEFAULTS_PATH} not found -- run export_feature_metadata.py first"
            )

        from xgboost import XGBClassifier

        model = XGBClassifier()
        model.load_model(MODEL_PATH)

        with open(FEATURE_COLUMNS_PATH) as f:
            feature_columns = json.load(f)
        with open(FEATURE_DEFAULTS_PATH) as f:
            feature_defaults = json.load(f)

        _model = model
        _feature_columns = feature_columns
        _feature_defaults = feature_defaults
    except Exception as e:
        _load_error = str(e)


def is_available():
    _load()
    return _model is not None


def build_feature_vector(txn_dict):
    _load()
    if _model is None:
        raise RuntimeError(f"global_model not available: {_load_error}")

    row = dict(_feature_defaults)

    amount = float(txn_dict.get("amount", 0.0))
    if "TransactionAmt" in row:
        row["TransactionAmt"] = amount

    df = pd.DataFrame([row])
    df = df.reindex(columns=_feature_columns, fill_value=0)
    return df


def score_global(txn_dict):
    _load()
    if _model is None:
        return None

    try:
        X = build_feature_vector(txn_dict)
        proba = _model.predict_proba(X)[0][1]
        return float(proba)
    except Exception as e:
        print(f"[global_model] scoring error: {e}")
        return None
