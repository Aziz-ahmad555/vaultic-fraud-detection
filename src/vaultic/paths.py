"""Repository and data paths, independent of the working directory.

Environment overrides (e.g. on Kaggle, where the competition data is read-only):
  VAULTIC_DATA_DIR  data root holding raw/, interim/ and features/ (default: <repo>/data)
  VAULTIC_RAW_DIR   folder with the raw IEEE-CIS CSVs (default: <data root>/raw)
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("VAULTIC_DATA_DIR") or REPO_ROOT / "data")
RAW_DIR = Path(os.environ.get("VAULTIC_RAW_DIR") or DATA_DIR / "raw")  # read-only CSVs
INTERIM_DIR = DATA_DIR / "interim"
FEATURES_DIR = DATA_DIR / "features"
RESEARCH_DIR = REPO_ROOT / "research"
CONFIG_DIR = REPO_ROOT / "experiments" / "configs"
RUNS_DIR = REPO_ROOT / "experiments" / "runs"

MERGED_PATH = INTERIM_DIR / "merged.parquet"
