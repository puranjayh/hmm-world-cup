"""
predictor.py — Match outcome predictor combining per-team HMMs with the joint tensor.

For a hypothetical match (team vs opponent on some date):
  1. Ask each team's HMM for its predictive hidden-state distribution.
  2. Combine via the joint emission tensor:
         P(outcome) = sum_{i,j} p_team[i] * p_opp[j] * T[i, j, outcome]
"""

import numpy as np
import pandas as pd


class Predictor:
    """Predict {Loss, Draw, Win} probabilities for a given (team, opp, date)."""

    def __init__(self,
                 team_hmms: dict,
                 joint_tensor: np.ndarray,
                 history_df: pd.DataFrame,
                 elo_ratings: dict | None = None):
        self.team_hmms    = team_hmms
        self.joint_tensor = np.asarray(joint_tensor, dtype=float)
        self.elo_ratings  = elo_ratings or {}
        self._n_states    = self.joint_tensor.shape[0]  # always 5

        hist = history_df.sort_values("date").reset_index(drop=True)
        self._per_team = {}
        for team, grp in hist.groupby("team", sort=False):
            self._per_team[team] = {
                "dates":    grp["date"].to_numpy(),
                "outcomes": grp["outcome"].to_numpy(dtype=int),
            }

    def _team_prior_outcomes(self, team: str, as_of_date) -> np.ndarray:
        rec = self._per_team.get(team)
        if rec is None:
            return np.empty(0, dtype=int)
        idx = np.searchsorted(
            rec["dates"], np.datetime64(pd.Timestamp(as_of_date)), side="left"
        )
        return rec["outcomes"][:idx]

    def state_dist_as_of(self, team: str, as_of_date,
                         elo_advantage: float = 0.0) -> np.ndarray:
        hmm = self.team_hmms.get(team)
        if hmm is None:
            # Unknown team — uniform over 5 states
            return np.full(self._n_states, 1.0 / self._n_states)
        prior = self._team_prior_outcomes(team, as_of_date)
        return hmm.predictive_state_dist(prior, elo_advantage=elo_advantage)

    def predict(self, team: str, opponent: str, as_of_date) -> dict:
        elo_team = self.elo_ratings.get(team, 0.0)
        elo_opp  = self.elo_ratings.get(opponent, 0.0)
        elo_diff = elo_team - elo_opp

        p_team = self.state_dist_as_of(team,     as_of_date, elo_advantage=elo_diff)
        p_opp  = self.state_dist_as_of(opponent, as_of_date, elo_advantage=-elo_diff)

        probs = np.einsum("i,j,ijo->o", p_team, p_opp, self.joint_tensor)
        total = probs.sum()
        probs = probs / total if total > 0 else np.full(3, 1.0 / 3.0)

        return {
            "Loss":       float(probs[0]),
            "Draw":       float(probs[1]),
            "Win":        float(probs[2]),
            "state_team": p_team.tolist(),
            "state_opp":  p_opp.tolist(),
            "elo_diff":   float(elo_diff),
        }