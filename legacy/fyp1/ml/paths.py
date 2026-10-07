"""Paths for the FYP-1 offline pipeline, independent of the working directory."""
from pathlib import Path

FYP1_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = FYP1_DIR.parents[1]

RAW_DIR = REPO_ROOT / "data" / "raw"  # read-only IEEE-CIS CSVs, shared with the research code
DATA_DIR = FYP1_DIR / "data"  # FYP-1's random 80/20 split (not used by Vaultic research)
MODEL_DIR = FYP1_DIR / "models_store"
