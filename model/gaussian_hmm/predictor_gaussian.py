"""
predictor_gaussian.py — Match outcome predictor for the Gaussian HMM.

Same interface as predictor.py but uses continuous feature histories
instead of discrete outcome sequences for state inference.
"""

import numpy as np
import pandas as pd

from model.gaussian_hmm.hmm_team_gaussian import FEATURE_NAMES


class GaussianPredictor:
    """Predict {Loss, Draw, Win} probabilities using Gaussian HMMs."""

    def __init__(self,
                 team_hmms: dict,
                 joint_tensor: np.ndarray,
                 history_df: pd.DataFrame,
                 elo_ratings: dict | None = None):
        self.team_hmms    = team_hmms
        self.joint_tensor = np.asarray(joint_tensor, dtype=float)
        self.elo_ratings  = elo_ratings or {}
        self._n_states    = self.joint_tensor.shape[0]

        hist = history_df.sort_values("date").reset_index(drop=True)
        self._per_team: dict[str, dict] = {}
        for team, grp in hist.groupby("team", sort=False):
            self._per_team[team] = {
                "dates":    grp["date"].to_numpy(),
                "features": grp[FEATURE_NAMES].fillna(0).to_numpy(dtype=float),
            }

    def _prior_features(self, team: str, as_of_date) -> np.ndarray:
        """Return feature rows for `team` strictly before `as_of_date`."""
        rec = self._per_team.get(team)
        if rec is None:
            return np.empty((0, len(FEATURE_NAMES)), dtype=float)
        idx = np.searchsorted(
            rec["dates"], np.datetime64(pd.Timestamp(as_of_date)), side="left"
        )
        return rec["features"][:idx]

    def state_dist_as_of(self, team: str, as_of_date,
                         elo_advantage: float = 0.0) -> np.ndarray:
        hmm = self.team_hmms.get(team)
        if hmm is None:
            return np.full(self._n_states, 1.0 / self._n_states)
        prior = self._prior_features(team, as_of_date)
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