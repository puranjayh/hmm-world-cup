"""
predictor_global.py — Prediction head for the global Gaussian HMM.

For each match (team A vs B on date D):
1. Take team A's last WINDOW matches before D → forward_state_dist → p_A (n_states,)
2. Take team B's last WINDOW matches before D → forward_state_dist → p_B (n_states,)
3. Build feature vector:
      [outer(p_A, p_B).flatten(), elo_diff, elo_diff^2]
4. Feed into a trained LogisticRegression → P(Win/Draw/Loss)

This avoids the joint tensor entirely. The logistic head learns how to combine
two teams' state distributions directly from labelled match outcomes.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from model.gaussian_hmm.hmm_global import GlobalGaussianHMM, FEATURE_NAMES

WINDOW = 20   # last N matches used for state inference


class GlobalPredictor:

    def __init__(self,
                 global_hmm: GlobalGaussianHMM,
                 head: LogisticRegression,
                 history_df: pd.DataFrame):
        self.hmm     = global_hmm
        self.head    = head
        self._build_index(history_df)

    def _build_index(self, df: pd.DataFrame) -> None:
        df = df.sort_values("date").reset_index(drop=True)
        self._per_team = {}
        for team, grp in df.groupby("team", sort=False):
            self._per_team[team] = {
                "dates":    grp["date"].to_numpy(),
                "features": grp[FEATURE_NAMES].fillna(0).to_numpy(dtype=float),
            }

    def update_history(self, df: pd.DataFrame) -> None:
        self._build_index(df)

    def _state_dist(self, team: str, as_of_date) -> np.ndarray:
        rec = self._per_team.get(team)
        if rec is None:
            return np.full(self.hmm.n_states, 1.0 / self.hmm.n_states)

        idx = np.searchsorted(
            rec["dates"], np.datetime64(pd.Timestamp(as_of_date)), side="left"
        )
        # Take last WINDOW matches before date
        feats = rec["features"][max(0, idx - WINDOW): idx]
        return self.hmm.forward_state_dist(feats)

    def _build_feature_vec(self,
                           p_team: np.ndarray,
                           p_opp: np.ndarray,
                           elo_diff: float) -> np.ndarray:
        """Flatten outer product + elo features."""
        outer = np.outer(p_team, p_opp).ravel()   # (n_states^2,)
        return np.concatenate([outer, [elo_diff, elo_diff ** 2, abs(elo_diff)]])

    def predict(self, team: str, opponent: str, as_of_date,
                elo_ratings: dict | None = None,
                elo_diff: float | None = None) -> dict:
        elo = elo_ratings or {}
        if elo_diff is None:
            elo_diff = elo.get(team, 0.0) - elo.get(opponent, 0.0)
        else:
            elo_diff = float(elo_diff)

        p_team = self._state_dist(team, as_of_date)
        p_opp  = self._state_dist(opponent, as_of_date)

        fv  = self._build_feature_vec(p_team, p_opp, elo_diff).reshape(1, -1)
        fv  = np.nan_to_num(fv, nan=0.0, posinf=0.0, neginf=0.0)
        raw = self.head.predict_proba(fv)[0]

        # align to [Loss=0, Draw=1, Win=2]
        probs = np.zeros(3, dtype=float)
        for k, cls in enumerate(self.head.classes_):
            probs[int(cls)] = raw[k]

        return {
            "Loss":       float(probs[0]),
            "Draw":       float(probs[1]),
            "Win":        float(probs[2]),
            "state_team": p_team.tolist(),
            "state_opp":  p_opp.tolist(),
            "elo_diff":   float(elo_diff),
        }
