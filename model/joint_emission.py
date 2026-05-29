"""
joint_emission.py — Estimate the joint emission tensor P(outcome | state_team, state_opp).

Each team has its own HMM telling us the probability distribution over
its hidden form (Poor/Neutral/Peak) right before a match. But a match
outcome depends on BOTH teams' states. We learn a tensor

    T[i, j, o] = P(outcome = o | team_state = i, opp_state = j)

by accumulating soft counts over historical matches: for every match we
take the predictive state distributions for both teams (using ONLY
prior outcomes — no leakage) and add their outer product into the slice
of T at the observed outcome. Laplace smoothing then a normalize over
the outcome axis gives proper probabilities.
"""

import numpy as np
import pandas as pd


def build_joint_tensor(train_df: pd.DataFrame,
                       team_hmms: dict,
                       smoothing: float = 1.0):
    """Estimate the (3,3,3) joint emission tensor from training matches.

    Returns
    -------
    tensor : np.ndarray shape (3, 3, 3) — T[i, j, o] = P(o | team=i, opp=j)
    diagnostics : dict with counts of matches used / skipped
    """
    # --- Pre-build, per team, a sorted (date, outcome) view ----------------
    # This lets us look up "all outcomes strictly before date D" in O(log n)
    # via searchsorted.
    sorted_df = train_df.sort_values("date").reset_index(drop=True)
    per_team = {}
    for team, grp in sorted_df.groupby("team", sort=False):
        per_team[team] = {
            "dates": grp["date"].to_numpy(),
            "outcomes": grp["outcome"].to_numpy(dtype=int),
        }

    def prior_outcomes(team: str, date) -> np.ndarray:
        rec = per_team.get(team)
        if rec is None:
            return np.empty(0, dtype=int)
        # strictly less than -> 'left' side of searchsorted
        idx = np.searchsorted(rec["dates"], np.datetime64(pd.Timestamp(date)), side="left")
        return rec["outcomes"][:idx]

    counts = np.zeros((3, 3, 3), dtype=float)
    matches_used = 0
    matches_skipped = 0

    for _, row in sorted_df.iterrows():
        team = row["team"]
        opp = row["opponent"]
        date = row["date"]
        outcome = int(row["outcome"])

        if team not in team_hmms or opp not in team_hmms:
            matches_skipped += 1
            continue

        p_team = team_hmms[team].predictive_state_dist(prior_outcomes(team, date))
        p_opp = team_hmms[opp].predictive_state_dist(prior_outcomes(opp, date))

        # Outer product of the two predictive distributions, added into
        # the slice of counts at the observed outcome.
        counts[:, :, outcome] += np.outer(p_team, p_opp)
        matches_used += 1

    # Laplace smoothing then normalize along the outcome axis.
    counts = counts + smoothing
    tensor = counts / counts.sum(axis=-1, keepdims=True)

    diagnostics = {
        "matches_used": matches_used,
        "matches_skipped": matches_skipped,
    }
    return tensor, diagnostics
