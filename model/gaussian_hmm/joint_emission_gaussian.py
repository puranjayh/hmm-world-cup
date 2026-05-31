"""
joint_emission_gaussian.py — Joint emission tensor for the Gaussian HMM.

T[i, j, o] = P(outcome = o | team_state = i, opp_state = j)
Tensor shape: (n_states, n_states, 3)
"""

import numpy as np
import pandas as pd

from model.gaussian_hmm.hmm_team_gaussian import FEATURE_NAMES


def build_joint_tensor_gaussian(
    train_df: pd.DataFrame,
    team_hmms: dict,
    smoothing: float = 1.0,
):
    n_states = next(iter(team_hmms.values())).n_states

    sorted_df = train_df.sort_values("date").reset_index(drop=True)

    # Pre-build per-team feature history for fast date-gated slicing
    per_team: dict[str, dict] = {}
    for team, grp in sorted_df.groupby("team", sort=False):
        per_team[team] = {
            "dates":    grp["date"].to_numpy(),
            "features": grp[FEATURE_NAMES].fillna(0).to_numpy(dtype=float),
        }

    def prior_features(team: str, date) -> np.ndarray:
        rec = per_team.get(team)
        if rec is None:
            return np.empty((0, len(FEATURE_NAMES)), dtype=float)
        idx = np.searchsorted(
            rec["dates"], np.datetime64(pd.Timestamp(date)), side="left"
        )
        return rec["features"][:idx]

    counts = np.zeros((n_states, n_states, 3), dtype=float)
    matches_used    = 0
    matches_skipped = 0

    for _, row in sorted_df.iterrows():
        team    = row["team"]
        opp     = row["opponent"]
        date    = row["date"]
        outcome = int(row["outcome"])

        if team not in team_hmms or opp not in team_hmms:
            matches_skipped += 1
            continue

        p_team = team_hmms[team].predictive_state_dist(prior_features(team, date))
        p_opp  = team_hmms[opp].predictive_state_dist(prior_features(opp, date))

        counts[:, :, outcome] += np.outer(p_team, p_opp)
        matches_used += 1

    counts = counts + smoothing
    tensor = counts / counts.sum(axis=-1, keepdims=True)

    return tensor, {
        "matches_used":    matches_used,
        "matches_skipped": matches_skipped,
        "n_states":        n_states,
    }