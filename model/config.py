"""Project-wide paths — the single source of truth for where data and artifacts live.

Model hyperparameters (state count, seed) are defined next to the code that uses
them: N_STATES and RANDOM_SEED live in model/gaussian_hmm/hmm_global.py, and the
per-tournament train/test cutoffs are the EVAL_RUNS table in evaluate_global.py.
"""
from pathlib import Path

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
DATA_CSV      = PROJECT_ROOT / "data" / "raw" / "filtered_matches.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "model" / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
