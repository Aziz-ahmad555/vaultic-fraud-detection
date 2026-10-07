"""
Run this once (after preprocess_data.py + train_model.py have already run).

It reads data/X_train.csv, which has the exact column layout the XGBoost
model was trained on, and saves two small files into models_store/:

  - feature_columns.json : ordered list of column names the model expects
  - feature_defaults.json: a sensible default value for each column, used
                            to fill in features we can't observe from a
                            live transaction (e.g. card type, email domain)
"""
import json
import os
import pandas as pd
from paths import DATA_DIR, MODEL_DIR

X_train = pd.read_csv(DATA_DIR / "X_train.csv")

feature_columns = X_train.columns.tolist()

defaults = {}
for col in feature_columns:
    series = X_train[col]
    if set(series.unique()) <= {0, 1}:
        defaults[col] = int(series.mode()[0])
    else:
        defaults[col] = float(series.median())

os.makedirs(MODEL_DIR, exist_ok=True)

with open(MODEL_DIR / "feature_columns.json", "w") as f:
    json.dump(feature_columns, f, indent=2)

with open(MODEL_DIR / "feature_defaults.json", "w") as f:
    json.dump(defaults, f, indent=2)

print(f"Saved {len(feature_columns)} feature columns and defaults to models_store/")
