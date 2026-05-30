"""
joint_emission.py — Estimate the joint emission tensor P(outcome | state_team, state_opp).

T[i, j, o] = P(outcome = o | team_state = i, opp_state = j)

All HMMs use 5 states. Tensor shape is always (5, 5, 3).
"""

import numpy as np
import pandas as pd


def build_joint_tensor(train_df: pd.DataFrame,
                       team_hmms: dict,
                       smoothing: float = 1.0):
    """Estimate the (5, 5, 3) joint emission tensor from training matches.

    Returns
    -------
    tensor : np.ndarray shape (5, 5, 3) — T[i, j, o] = P(o | team=i, opp=j)
    diagnostics : dict
    """
    n_states = 7  # all HMMs are 5-state

    sorted_df = train_df.sort_values("date").reset_index(drop=True)
    per_team = {}
    for team, grp in sorted_df.groupby("team", sort=False):
        per_team[team] = {
            "dates":    grp["date"].to_numpy(),
            "outcomes": grp["outcome"].to_numpy(dtype=int),
        }

    def prior_outcomes(team: str, date) -> np.ndarray:
        rec = per_team.get(team)
        if rec is None:
            return np.empty(0, dtype=int)
        idx = np.searchsorted(
            rec["dates"], np.datetime64(pd.Timestamp(date)), side="left"
        )
        return rec["outcomes"][:idx]

    counts = np.zeros((n_states, n_states, 3), dtype=float)
    matches_used = 0
    matches_skipped = 0

    for _, row in sorted_df.iterrows():
        team    = row["team"]
        opp     = row["opponent"]
        date    = row["date"]
        outcome = int(row["outcome"])

        if team not in team_hmms or opp not in team_hmms:
            matches_skipped += 1
            continue

        p_team = team_hmms[team].predictive_state_dist(prior_outcomes(team, date))
        p_opp  = team_hmms[opp].predictive_state_dist(prior_outcomes(opp, date))

        counts[:, :, outcome] += np.outer(p_team, p_opp)
        matches_used += 1

    counts = counts + smoothing
    tensor = counts / counts.sum(axis=-1, keepdims=True)

    diagnostics = {
        "matches_used":    matches_used,
        "matches_skipped": matches_skipped,
        "n_states":        n_states,
    }
    return tensor, diagnostics