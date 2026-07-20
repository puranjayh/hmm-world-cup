"""
Data loading for the global Gaussian HMM benchmark.

Reads the filtered match CSV and attaches an integer `outcome` column
(0=Loss, 1=Draw, 2=Win) from the perspective of `team`. The train/test
split is not done here — evaluate_global.py slices each held-out tournament
by date from its EVAL_RUNS table.
"""
from __future__ import annotations

import pandas as pd

from model.config import DATA_CSV

# Map raw `result` column (-1 loss, 0 draw, +1 win for `team`) -> outcome index.
RESULT_TO_OUTCOME = {-1: 0, 0: 1, 1: 2}


def load_matches() -> pd.DataFrame:
    """Load the filtered matches CSV and attach an integer `outcome` column.

    Returns a DataFrame sorted ascending by `date` with a clean RangeIndex.
    Rows with an unmappable / missing outcome are dropped.
    """
    df = pd.read_csv(DATA_CSV)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Map result -> outcome (0/1/2). Anything unrecognised becomes NaN and is dropped.
    df["outcome"] = df["result"].map(RESULT_TO_OUTCOME)
    df = df.dropna(subset=["outcome", "date"]).copy()
    df["outcome"] = df["outcome"].astype(int)

    df = df.sort_values("date").reset_index(drop=True)
    return df
