"""
predictor.py — Match outcome predictor that combines per-team HMMs with the joint tensor.

For a hypothetical match (team vs opponent on some date):
  1. Ask each team's HMM for its predictive hidden-state distribution,
     using only outcomes strictly before the match date (no leakage).
  2. Combine via the joint emission tensor learned in joint_emission.py:
         P(outcome) = sum_{i,j} p_team[i] * p_opp[j] * T[i, j, outcome]
This is just marginalizing out the two unknown hidden form states.
"""

import numpy as np
import pandas as pd


class Predictor:
    """Predict {Loss, Draw, Win} probabilities for a given (team, opp, date)."""

    def __init__(self,
                 team_hmms: dict,
                 joint_tensor: np.ndarray,
                 history_df: pd.DataFrame):
        self.team_hmms = team_hmms
        self.joint_tensor = np.asarray(joint_tensor, dtype=float)
        # Pre-sort once so we can take fast slices per team.
        hist = history_df.sort_values("date").reset_index(drop=True)
        self._per_team = {}
        for team, grp in hist.groupby("team", sort=False):
            self._per_team[team] = {
                "dates": grp["date"].to_numpy(),
                "outcomes": grp["outcome"].to_numpy(dtype=int),
            }

    def _team_prior_outcomes(self, team: str, as_of_date) -> np.ndarray:
        """Return outcomes for `team` strictly before `as_of_date`."""
        rec = self._per_team.get(team)
        if rec is None:
            return np.empty(0, dtype=int)
        idx = np.searchsorted(rec["dates"], np.datetime64(pd.Timestamp(as_of_date)), side="left")
        return rec["outcomes"][:idx]

    def state_dist_as_of(self, team: str, as_of_date) -> np.ndarray:
        """Predictive hidden state dist for `team` just before `as_of_date`."""
        hmm = self.team_hmms.get(team)
        if hmm is None:
            # Unknown team -> uniform prior over hidden form.
            return np.full(3, 1.0 / 3.0)
        prior = self._team_prior_outcomes(team, as_of_date)
        return hmm.predictive_state_dist(prior)

    def predict(self, team: str, opponent: str, as_of_date) -> dict:
        """Return outcome probability dict + the two state distributions."""
        p_team = self.state_dist_as_of(team, as_of_date)
        p_opp = self.state_dist_as_of(opponent, as_of_date)

        # P(o) = sum_{i,j} p_team[i] * p_opp[j] * T[i, j, o]
        probs = np.einsum("i,j,ijo->o", p_team, p_opp, self.joint_tensor)
        # Defensive normalization in case of tiny numerical drift.
        total = probs.sum()
        if total <= 0:
            probs = np.full(3, 1.0 / 3.0)
        else:
            probs = probs / total

        return {
            "Loss": float(probs[0]),
            "Draw": float(probs[1]),
            "Win": float(probs[2]),
            "state_team": p_team.tolist(),
            "state_opp": p_opp.tolist(),
        }
