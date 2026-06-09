"""
predictor_global.py — Prediction head for the global Gaussian HMM.

For each match (team A vs B on date D):
1. Take team A's last WINDOW matches before D → posterior_features → pf_A (N+2,)
2. Take team B's last WINDOW matches before D → posterior_features → pf_B (N+2,)
3. Build feature vector:
      [outer(p_A, p_B).ravel(),          (N²)  joint regime interaction
       max_p_A, max_p_B,                 (2)   HMM confidence
       entropy_A, entropy_B,             (2)   HMM uncertainty
       elo_diff,                         (1)   rating difference
       elo_diff * max_p_A,               (1)   strength × confidence (team)
       elo_diff * max_p_B]               (1)   strength × confidence (opponent)
4. Feed into a trained LogisticRegression → P(Win/Draw/Loss)

Posterior summary layout (per team, from hmm_global.posterior_features):
    [0 : N]   p       — predictive state distribution
    [N]       max_p   — confidence (peak probability)
    [N+1]     entropy — uncertainty (Shannon entropy, nats)

Note: p_next and delta were removed — deterministic linear functions of p,
so they add collinearity without new information.
"""

import numpy as np
import pandas as pd

from model.gaussian_hmm.hmm_global import GlobalGaussianHMM, FEATURE_NAMES

WINDOW = 2   # last N matches used for state inference (~one competitive cycle)


class GlobalPredictor:

    def __init__(self,
                 global_hmm: GlobalGaussianHMM,
                 head,
                 history_df: pd.DataFrame):
        self.hmm  = global_hmm
        self.head = head
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

    def _posterior(self, team: str, as_of_date) -> np.ndarray:
        """
        Return the full posterior summary vector (N + 2,) for a team
        as of a given date, using only matches strictly before that date.
        Falls back to a uniform / max-entropy vector for unseen teams.
        """
        N   = self.hmm.n_states
        rec = self._per_team.get(team)

        if rec is None:
            unif = np.full(N, 1.0 / N)
            return np.concatenate([unif, [1.0 / N, np.log(N)]])

        idx   = np.searchsorted(
            rec["dates"], np.datetime64(pd.Timestamp(as_of_date)), side="left"
        )
        feats = rec["features"][max(0, idx - WINDOW): idx]
        return self.hmm.posterior_features(feats)

    def _build_feature_vec(self,
                           pf_team: np.ndarray,
                           pf_opp:  np.ndarray,
                           elo_diff: float) -> np.ndarray:
        """
        Construct head feature vector from two posterior summaries + elo_diff.
        Must stay in sync with evaluate_global._build_feature_vec.

        Layout:
            outer(p_A, p_B).ravel()   (N²)  joint regime interaction
            max_p_A, max_p_B          (2)   HMM confidence
            ent_A,   ent_B            (2)   HMM uncertainty
            elo_diff                  (1)
            elo_diff * max_p_A        (1)   interaction: strength × confidence
            elo_diff * max_p_B        (1)   interaction: strength × confidence
        """
        N = self.hmm.n_states

        p_A     = pf_team[:N]
        max_p_A = pf_team[N]
        ent_A   = pf_team[N + 1]

        p_B     = pf_opp[:N]
        max_p_B = pf_opp[N]
        ent_B   = pf_opp[N + 1]

        return np.concatenate([
            np.outer(p_A, p_B).ravel(),    # (N²,)
            [max_p_A, max_p_B],             # (2,)
            [ent_A,   ent_B],               # (2,)
            [elo_diff],                     # (1,)
            [elo_diff * max_p_A],           # (1,)
            [elo_diff * max_p_B],           # (1,)
        ])

    def predict(self, team: str, opponent: str, as_of_date,
                elo_ratings: dict | None = None,
                elo_diff: float | None = None) -> dict:
        elo = elo_ratings or {}
        if elo_diff is None:
            elo_diff = elo.get(team, 0.0) - elo.get(opponent, 0.0)
        else:
            elo_diff = float(elo_diff)

        pf_team = self._posterior(team,     as_of_date)
        pf_opp  = self._posterior(opponent, as_of_date)

        fv  = self._build_feature_vec(pf_team, pf_opp, elo_diff).reshape(1, -1)
        fv  = np.nan_to_num(fv, nan=0.0, posinf=0.0, neginf=0.0)
        raw = self.head.predict_proba(fv)[0]

        # align to [Loss=0, Draw=1, Win=2]
        probs = np.zeros(3, dtype=float)
        for k, cls in enumerate(self.head.classes_):
            probs[int(cls)] = raw[k]

        N = self.hmm.n_states
        return {
            "Loss":         float(probs[0]),
            "Draw":         float(probs[1]),
            "Win":          float(probs[2]),
            "state_team":   pf_team[:N].tolist(),    # predictive state distribution
            "state_opp":    pf_opp[:N].tolist(),
            "conf_team":    float(pf_team[N]),        # max_p
            "conf_opp":     float(pf_opp[N]),
            "entropy_team": float(pf_team[N + 1]),
            "entropy_opp":  float(pf_opp[N + 1]),
            "elo_diff":     float(elo_diff),
        }