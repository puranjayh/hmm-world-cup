"""Project-wide constants. Keep this file the single source of truth for paths and labels."""
from pathlib import Path
import pandas as pd

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
DATA_CSV      = PROJECT_ROOT / "data" / "raw" / "filtered_matches.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "model" / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

STATE_LABELS   = ["Very Poor", "Poor", "Below Neutral", "Neutral", "Above Neutral", "Peak", "Dominant"]
OUTCOME_LABELS = ["Loss", "Draw", "Win"]
N_STATES   = 7
N_OUTCOMES = 3

TRAIN_END_DATE = pd.Timestamp("2022-11-19")
RANDOM_SEED    = 42
MIN_MATCHES    = 62