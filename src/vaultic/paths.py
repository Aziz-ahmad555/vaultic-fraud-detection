"""Repository paths, independent of the working directory."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"  # read-only IEEE-CIS CSVs
INTERIM_DIR = DATA_DIR / "interim"
FEATURES_DIR = DATA_DIR / "features"
RESEARCH_DIR = REPO_ROOT / "research"
CONFIG_DIR = REPO_ROOT / "experiments" / "configs"
RUNS_DIR = REPO_ROOT / "experiments" / "runs"

MERGED_PATH = INTERIM_DIR / "merged.parquet"
