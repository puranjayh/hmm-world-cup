"""
hmm_global.py — Single global Gaussian HMM over match feature vectors.

Architecture
------------
Instead of fitting one HMM per team, we fit ONE global HMM over all matches.
Each observation is a match represented as a pre-match feature vector.
Hidden states represent global "match regimes":
    e.g. "dominant favourite win", "competitive upset", "defensive draw", etc.

This sidesteps the per-team state comparability problem — all teams live in
the same state space, and the joint tensor is replaced by a direct logistic
regression from (state, elo_diff) → P(outcome).

Prediction pipeline
-------------------
1. For a match (team A vs team B on date D):
   a. Take the last K matches for team A before date D as observations.
   b. Run the forward algorithm to get P(state | team A's recent history).
   c. Do the same for team B.
2. Combine state distributions + elo_diff in a logistic regression head
   trained on held-out training data.

Features (all pre-match, no leakage)
-------------------------------------
Per match row (from team's perspective):
    - elo_diff           : team_elo - opponent_elo
    - rolling_win_rate_5 : win rate over last 5 matches
    - rolling_goal_diff_5: avg goal diff over last 5 matches
    - tournament_weight  : match importance (5=WC, 1=friendly)
"""

import pickle
import numpy as np
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp
from sklearn.preprocessing import StandardScaler

RANDOM_SEED     = 42
EPS             = 1e-12

FEATURE_NAMES = [
    "elo_diff",
    "tournament_weight",
    "ewa_win_rate",
    "ewa_goal_diff",
    "neutral",
    "opp_elo_strength_5",
    "rolling_win_vs_strong_5",
]
N_FEATURES = len(FEATURE_NAMES)
N_STATES   = 3   # global match regimes


class GlobalGaussianHMM:
    """Single HMM fitted on all matches — learns global match regimes."""

    def __init__(self, n_states: int = N_STATES):
        self.n_states = n_states
        self.scaler   = StandardScaler()
        self.model    = GaussianHMM(
            n_components=n_states,
            covariance_type="diag",  # diag: n_states*n_features params vs full: n_states*n_features^2
            n_iter=500,
            tol=1e-4,
            random_state=RANDOM_SEED,
            init_params="stmc",
            min_covar=1.0,           # strong floor on variance — prevents collapse with 9 features
        )

    def fit(self, X: np.ndarray, lengths: list) -> "GlobalGaussianHMM":
        """
        X       : (total_matches, N_FEATURES) — all training matches concatenated
        lengths : list of ints — number of matches per team sequence
        """
        X = np.asarray(X, dtype=float)
        X = self.scaler.fit_transform(X)

        self.model.fit(X, lengths=lengths)

        # Clamp diagonal variances — bypass hmmlearn's setter to avoid shape validation
        clamped = np.maximum(self.model.covars_, 0.5)
        object.__setattr__(self.model, '_covars_', clamped)

        # Fix degenerate transmat rows (zero sum = state never visited)
        # Replace with uniform distribution so forward algorithm doesn't produce NaN
        tm = self.model.transmat_.copy()
        zero_rows = tm.sum(axis=1) == 0
        tm[zero_rows] = 1.0 / self.n_states
        self.model.transmat_ = tm

        return self

    def state_sequence(self, X: np.ndarray) -> np.ndarray:
        """Viterbi decode — returns most likely state sequence."""
        X = self.scaler.transform(np.asarray(X, dtype=float))
        return self.model.predict(X)

    def forward_state_dist(self, X: np.ndarray) -> np.ndarray:
        """
        Run forward algorithm on sequence X, push one step ahead.
        Returns predictive state distribution P(next_state | X).
        """
        if len(X) == 0:
            return np.full(self.n_states, 1.0 / self.n_states)

        X = self.scaler.transform(np.asarray(X, dtype=float))
        log_lik   = self._log_likelihoods(X)
        log_trans = np.log(np.asarray(self.model.transmat_, dtype=float) + EPS)
        log_start = np.log(np.asarray(self.model.startprob_, dtype=float) + EPS)

        alpha = log_start + log_lik[0]
        for t in range(1, len(log_lik)):
            alpha = logsumexp(alpha[:, None] + log_trans, axis=0) + log_lik[t]

        log_pred = logsumexp(alpha[:, None] + log_trans, axis=0)
        log_pred -= log_pred.max()
        pred = np.exp(log_pred)
        return pred / (pred.sum() + EPS)

    def _log_likelihoods(self, X: np.ndarray) -> np.ndarray:
        """Diagonal covariance log-likelihood → (T, n_states).
        covars_ shape is (n_states, n_features) for diag.
        """
        means  = self.model.means_    # (n_states, n_features)
        covars = self.model.covars_   # (n_states, n_features)
        T, d   = X.shape
        log_lik = np.zeros((T, self.n_states), dtype=float)

        for s in range(self.n_states):
            mu  = means[s]
            var = np.maximum(covars[s], EPS)   # (n_features,)
            diff = X - mu
            log_lik[:, s] = -0.5 * np.sum(
                np.log(2 * np.pi * var) + (diff ** 2) / var, axis=1
            )
        return log_lik

    def save(self, path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path) -> "GlobalGaussianHMM":
        with open(path, "rb") as f:
            return pickle.load(f)