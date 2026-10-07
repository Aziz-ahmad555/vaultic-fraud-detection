# ml_model.py
import os
import uuid

import numpy as np
import pandas as pd
from joblib import dump, load
from datetime import datetime, timezone
from sklearn.ensemble import IsolationForest

from collections import Counter
from models import Transaction
import global_model

MODEL_DIR = "models_store"
os.makedirs(MODEL_DIR, exist_ok=True)

MIN_HISTORY = 30
FEATURE_COLS = [
    "amount", "log_amount", "hour", "dow", "delta_prev_seconds",
    "rolling_mean_10", "rolling_count_10", "device_change"
]

def model_path(user_id):
    return os.path.join(MODEL_DIR, f"user_model_{user_id}.joblib")

def save_model(model, user_id):
    path = model_path(user_id)
    dump(model, path)
    return path

def load_model(user_id):
    path = model_path(user_id)
    if os.path.exists(path):
        return load(path)
    return None

def build_user_df(user_id, exclude_transaction_id=None):
    """
    Return DataFrame with user's transactions sorted asc and timestamp UTC-aware.

    exclude_transaction_id: if given, skips that transaction. Callers scoring
    a transaction that's already been committed to the DB must pass its id
    here -- otherwise the transaction being scored counts toward its own
    historical baseline (e.g. its own amount would inflate "historical max"
    to itself, making the extreme-amount rule below unable to ever trigger).
    """
    query = Transaction.query.filter_by(user_id=user_id)
    if exclude_transaction_id:
        query = query.filter(Transaction.transaction_id != exclude_transaction_id)
    txns = query.order_by(Transaction.timestamp).all()
    rows = []
    for t in txns:
        if t.timestamp is None:
            continue
        ts = t.timestamp
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        rows.append({
            "transaction_id": t.transaction_id,
            "amount": float(t.amount),
            "timestamp": ts,
            "device_id": t.device_id or ""
        })
    if not rows:
        df = pd.DataFrame(columns=["transaction_id", "amount", "timestamp", "device_id"])
        return df
    df = pd.DataFrame(rows)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df

def featurize_df(df):
    """Generates feature matrix for historical transactions (no leakage)."""
    if df.empty:
        return np.empty((0, len(FEATURE_COLS)))
    df = df.copy()
    df['log_amount'] = np.log1p(df['amount'].clip(lower=0))
    df['hour'] = df['timestamp'].dt.hour
    df['dow'] = df['timestamp'].dt.weekday
    df['delta_prev_seconds'] = df['timestamp'].diff().dt.total_seconds().fillna(0)
    df['rolling_mean_10'] = df['amount'].rolling(window=10, min_periods=1).mean().shift(1).fillna(0)
    df['rolling_count_10'] = df['amount'].rolling(window=10, min_periods=1).count().shift(1).fillna(0)
    df['device_change'] = (df['device_id'] != df['device_id'].shift(1)).astype(int).fillna(0)
    X = df[["amount", "log_amount", "hour", "dow", "delta_prev_seconds", "rolling_mean_10", "rolling_count_10", "device_change"]].values
    return X

def featurize_for_scoring(user_df, new_txn):
    """Create feature vector for a new transaction using user's historical df."""
    ts = new_txn.get("timestamp")
    if isinstance(ts, str):
        ts = pd.to_datetime(ts, utc=True)
    elif isinstance(ts, datetime):
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        ts = pd.to_datetime(ts).tz_convert('UTC') if getattr(ts, "tzinfo", None) else pd.to_datetime(ts).tz_localize('UTC')
    else:
        ts = pd.to_datetime(datetime.now(timezone.utc), utc=True)

    amount = float(new_txn.get("amount", 0.0))
    log_amount = np.log1p(max(0.0, amount))
    hour = int(ts.hour)
    dow = int(ts.weekday())

    if user_df is None or user_df.empty:
        delta_prev_seconds = 0.0
        rolling_mean_10 = 0.0
        rolling_count_10 = 0.0
        last_device = ""
    else:
        last_ts = user_df['timestamp'].max()
        delta_prev_seconds = (ts - last_ts).total_seconds()
        last_n = user_df.tail(10)
        rolling_mean_10 = float(last_n['amount'].mean()) if not last_n.empty else 0.0
        rolling_count_10 = float(len(last_n))
        last_device = last_n['device_id'].iloc[-1] if not last_n.empty else ""

    device_change = 0
    new_device = new_txn.get("device_id") or ""
    if new_device:
        device_change = 1 if new_device != last_device else 0

    x = np.array([amount, log_amount, hour, dow, delta_prev_seconds, rolling_mean_10, rolling_count_10, device_change], dtype=float)
    return x.reshape(1, -1)

def adaptive_contamination(user_id, default=0.01):
    """
    Continuous-learning hook: use analyst-confirmed labels (Label table) to
    tune how aggressively the model flags anomalies for this user.
    """
    from models import Label, Alert
    labels = (
        Label.query.join(Alert, Label.alert_id == Alert.alert_id)
        .filter(Alert.user_id == user_id)
        .all()
    )
    if len(labels) < 5:
        return default
    fraud_rate = sum(1 for l in labels if l.is_fraud) / len(labels)
    return float(min(max(fraud_rate, 0.005), 0.15))


def train_user_model(user_id):
    """Train IsolationForest for a user and save to disk."""
    df = build_user_df(user_id)
    n = len(df)
    if n < MIN_HISTORY:
        return {"status": "not_enough_data", "n": n}
    X = featurize_df(df)
    contamination = adaptive_contamination(user_id)
    model = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
    model.fit(X)
    path = save_model(model, user_id)
    return {"status": "trained", "n": n, "path": path, "contamination": contamination}

def severity_band(score):
    """Map a continuous 0..1 fraud score to a Low/Medium/High band."""
    if score >= 0.8:
        return "High"
    elif score >= 0.5:
        return "Medium"
    else:
        return "Low"


# --------------------------------------------------------------------------
# Rule-based extreme-amount override
# --------------------------------------------------------------------------
# IsolationForest saturates once a value exceeds every split threshold the
# trees ever learned: two amounts that are both "bigger than anything seen
# in training" take an identical path through every tree and get an
# identical anomaly score, even if one is 10x the other and one is 10,000x.
# This means a user whose training history happens to include a few very
# large amounts (e.g. from generated test data) can end up with a model
# that literally cannot distinguish "somewhat large" from "catastrophically
# large." A hard, explainable rule on top of the ML score closes that gap
# and is standard practice in real fraud systems (ML catches subtle
# patterns; simple thresholds catch extreme statistical outliers).
EXTREME_MULTIPLIER = 5.0   # new amount > this many times the user's historical max
MIN_HISTORY_FOR_RULE = 3   # need at least this many past transactions to compare against
# NOTE: lower thresholds (like 3) make the rule usable for newer accounts/small
# test datasets, but a "historical max" computed from only 3 transactions is
# noisier and more prone to false positives than one computed from 30+. For
# your final report, treat this as a documented, tunable tradeoff -- not a
# fixed constant -- and mention that a production system would likely raise
# this threshold as real user history accumulates.


def check_extreme_amount(user_df, amount):
    """
    Returns (triggered: bool, reason: str | None) if `amount` is a large
    multiple of this user's historical maximum transaction amount.
    """
    if user_df is None or user_df.empty or len(user_df) < MIN_HISTORY_FOR_RULE:
        return False, None

    historical_max = float(user_df["amount"].max())
    if historical_max <= 0:
        return False, None

    if amount >= EXTREME_MULTIPLIER * historical_max:
        reason = (
            f"Amount is {amount / historical_max:.1f}x this user's historical "
            f"maximum transaction (Pkr.{historical_max:,.2f})"
        )
        return True, reason

    return False, None


def score_transaction(user_id, txn_dict):
    """
    Score a new transaction for a user, blending two models:
      - Personal model: Isolation Forest trained on this user's own history.
      - Global model: XGBoost trained on IEEE-CIS (see global_model.py).
        NOTE: live schema only supplies TransactionAmt directly; other
        features fall back to training-set defaults -- documented limitation.
      - Rule-based override: catches amounts so far outside the user's own
        history that the Isolation Forest has saturated and can no longer
        tell them apart (see check_extreme_amount above).
    """
    model = load_model(user_id)
    current_txn_id = txn_dict.get("transaction_id")
    user_df = build_user_df(user_id, exclude_transaction_id=current_txn_id)
    amount = float(txn_dict.get("amount", 0.0))

    global_score = global_model.score_global(txn_dict)
    rule_triggered, rule_reason = check_extreme_amount(user_df, amount)

    if model is None:
        if global_score is not None:
            score = global_score
            is_anomaly = global_score >= 0.5
        else:
            score = None
            is_anomaly = False

        if rule_triggered:
            score = max(score or 0.0, 0.95)
            is_anomaly = True

        return {
            "has_model": model is not None,
            "score": score,
            "is_anomaly": is_anomaly,
            "raw": None,
            "personal_score": None,
            "global_score": global_score,
            "rule_triggered": rule_triggered,
            "rule_reason": rule_reason,
        }

    x = featurize_for_scoring(user_df, txn_dict)

    raw = float(model.decision_function(x)[0])
    pred = int(model.predict(x)[0])

    try:
        personal_score = float(1.0 / (1.0 + np.exp(raw)))
    except OverflowError:
        personal_score = 1.0 if raw < 0 else 0.0

    is_anomaly = (pred == -1)

    if global_score is not None:
        blended_score = 0.5 * personal_score + 0.5 * global_score
    else:
        blended_score = personal_score

    # Rule-based override: force a high score/anomaly flag if the amount is
    # an extreme multiple of this user's own history, regardless of what
    # the (potentially saturated) model score says.
    if rule_triggered:
        blended_score = max(blended_score, 0.95)
        is_anomaly = True

    return {
        "has_model": True,
        "score": blended_score,
        "is_anomaly": bool(is_anomaly),
        "raw": raw,
        "personal_score": personal_score,
        "global_score": global_score,
        "rule_triggered": rule_triggered,
        "rule_reason": rule_reason,
    }


def analyze_daily_behavior(user_id):
    """Detect abnormal number of transactions in a single day for a given user."""
    from app import Transaction, Alert, db

    txns = Transaction.query.filter_by(user_id=user_id).order_by(Transaction.timestamp.asc()).all()
    if not txns or len(txns) < 10:
        return {"status": "insufficient_data"}

    daily_counts = Counter([t.timestamp.date() for t in txns])
    daily_values = np.array(list(daily_counts.values()))

    mean_txn = np.mean(daily_values)
    std_txn = np.std(daily_values) if np.std(daily_values) > 0 else 1.0

    today = datetime.now(timezone.utc).date()
    today_count = daily_counts.get(today, 0)

    z_score = (today_count - mean_txn) / std_txn

    if z_score >= 2:
        existing = Alert.query.filter_by(
            user_id=user_id,
            message=f"Unusual transaction frequency detected for {today}"
        ).first()

        if not existing:
            alert = Alert(
                alert_id=str(uuid.uuid4()),
                user_id=user_id,
                transaction_id=None,
                score=float(z_score),
                severity="High" if z_score >= 3 else "Medium",
                status="Pending",
                message=f"Unusual transaction frequency detected for {today}"
            )
            db.session.add(alert)
            db.session.commit()
        return {"status": "anomaly", "z_score": z_score, "count": today_count}
    else:
        return {"status": "normal", "z_score": z_score, "count": today_count}
