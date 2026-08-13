"""
Data loading utilities for the HMM football predictor.

Reads the filtered match CSV, converts the textual/integer result column into
a 3-way outcome code (0=Loss, 1=Draw, 2=Win) from the perspective of `team`,
splits chronologically into train/test, and produces per-team observation
sequences that the per-team HMMs can consume directly.

Plain-English summary: this module turns the raw CSV into NumPy integer
sequences, one per national team, ordered by date. Each integer is what that
team's HMM sees as an "emission" (Loss/Draw/Win).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from model.config import DATA_CSV, TRAIN_END_DATE

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


def split_train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split chronologically on TRAIN_END_DATE (exclusive for train).

    train = rows strictly before TRAIN_END_DATE
    test  = rows on/after TRAIN_END_DATE  (this is our holdout, e.g. 2022 WC).
    """
    mask = df["date"] < TRAIN_END_DATE
    train_df = df.loc[mask].copy().reset_index(drop=True)
    test_df  = df.loc[~mask].copy().reset_index(drop=True)
    return train_df, test_df


def team_sequences(train_df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Return {team_name: np.ndarray[int]} of date-sorted outcome sequences.

    Each array is the chronological list of Loss/Draw/Win codes for that
    team — i.e. the observation sequence we feed to that team's HMM during
    `.fit(...)`.
    """
    sequences: dict[str, np.ndarray] = {}
    # `train_df` is already sorted by date; groupby preserves within-group order.
    for team, group in train_df.groupby("team", sort=False):
        seq = group.sort_values("date")["outcome"].to_numpy(dtype=int)
        sequences[team] = seq
    return sequences
