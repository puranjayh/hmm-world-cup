"""Project-wide constants. Keep this file the single source of truth for paths and labels."""
from pathlib import Path
import pandas as pd

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
DATA_CSV      = PROJECT_ROOT / "data" / "raw" / "filtered_matches.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "model" / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

STATE_LABELS   = ["Very Poor", "Poor", "Neutral", "Peak", "Dominant"]   # state index 0,1,2,3,4
OUTCOME_LABELS = ["Loss", "Draw", "Win"]                       # outcome index 0,1,2
N_STATES   = 5
N_OUTCOMES = 3

TRAIN_END_DATE = pd.Timestamp("2022-01-01")  # exclusive: train = date < this; test = date >= this
RANDOM_SEED    = 42
MIN_MATCHES    = 40    # minimum training matches per team to fit an HMM
