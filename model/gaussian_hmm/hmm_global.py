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
regression from (state_features, elo_diff) → P(outcome).

Prediction pipeline
-------------------
1. For a match (team A vs team B on date D):
   a. Take the last K matches for team A before date D as observations.
   b. Run the forward algorithm to get the full posterior summary for team A.
   c. Do the same for team B.
2. Combine posterior summaries + elo_diff in a logistic regression head
   trained on held-out training data.

Posterior summary vector (per team, produced by posterior_features())
----------------------------------------------------------------------
For an N-state HMM the summary has N + 2 dimensions:
    [0 : N]   p       — predictive state distribution
    [N]       max_p   — peak state probability (confidence)
    [N+1]     entropy — Shannon entropy of p (uncertainty)

Note: p_next and delta were removed — they are linear functions of p
given fixed T, so they add no information to the logistic head.

Features (all pre-match, no leakage)
-------------------------------------
Per match row (from team's perspective):
    - ewa_win_rate
    - ewa_goal_diff
    - rolling_win_vs_strong_5
    - rolling_goal_diff_std_5
    - rolling_win_rate_std_5
    - ewa_win_rate_momentum
    - ewa_goal_diff_momentum
"""

import pickle
import numpy as np
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp
from sklearn.preprocessing import StandardScaler

RANDOM_SEED     = 42
EPS             = 1e-12

FEATURE_NAMES = [
    'ewa_win_rate',
    'ewa_goal_diff',
    'rolling_win_vs_strong_5',
    'rolling_goal_diff_std_5',
    'rolling_win_rate_std_5',
    'ewa_win_rate_momentum',
    'ewa_goal_diff_momentum'
]
N_FEATURES = len(FEATURE_NAMES)
N_STATES   = 7   # global match regimes

# Tournament weights used during HMM fitting to up-weight competitive matches.
# Any tournament not listed defaults to 1.0 (friendly-level).
TOURNAMENT_WEIGHTS = {
    "FIFA World Cup":               5.0,
    "UEFA Euro":                    4.5,
    "Copa América":                 4.5,
    "Africa Cup of Nations":        4.0,
    "AFC Asian Cup":                4.0,
    "CONCACAF Gold Cup":            3.5,
    "FIFA World Cup qualification": 3.0,
    "UEFA Nations League":          2.5,
    "Friendly":                     0.5,   # actively down-weighted
}


def _tournament_sample_weight(tournament_col: np.ndarray) -> np.ndarray:
    """Convert a string array of tournament names to fitting sample weights."""
    return np.array(
        [TOURNAMENT_WEIGHTS.get(t, 1.0) for t in tournament_col],
        dtype=float,
    )


class GlobalGaussianHMM:
    """Single HMM fitted on all matches — learns global match regimes."""

    def __init__(self, n_states: int = N_STATES):
        self.n_states = n_states
        self.scaler   = StandardScaler()
        self.model    = GaussianHMM(
            n_components=n_states,
            covariance_type="diag",
            n_iter=500,
            tol=1e-4,
            random_state=RANDOM_SEED,
            init_params="stmc",
            min_covar=1.0,
        )

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        lengths: list,
        sample_weight: np.ndarray | None = None,
    ) -> "GlobalGaussianHMM":
        """
        X             : (total_matches, N_FEATURES) — all training matches concatenated
        lengths       : list of ints — number of matches per team sequence
        sample_weight : (total_matches,) optional per-observation weights.
                        Competitive matches should have higher weights so the
                        HMM learns regime structure from meaningful fixtures
                        rather than friendlies.  If None, all weights = 1.
        """
        X = np.asarray(X, dtype=float)
        X_scaled = self.scaler.fit_transform(X)

        if sample_weight is not None:
            # hmmlearn does not natively support per-observation weights, so we
            # approximate by repeating observations proportional to their weight.
            # We round weights to the nearest integer (min 1) and replicate rows
            # and update lengths accordingly.
            w = np.asarray(sample_weight, dtype=float)
            w = np.clip(np.round(w).astype(int), 1, None)

            X_rep, lengths_rep = [], []
            ptr = 0
            for seq_len in lengths:
                seq_w  = w[ptr: ptr + seq_len]
                seq_X  = X_scaled[ptr: ptr + seq_len]
                rows   = np.repeat(seq_X, seq_w, axis=0)
                X_rep.append(rows)
                lengths_rep.append(len(rows))
                ptr += seq_len

            X_fit      = np.vstack(X_rep)
            lengths_fit = lengths_rep
        else:
            X_fit      = X_scaled
            lengths_fit = lengths

        self.model.fit(X_fit, lengths=lengths_fit)

        # Clamp diagonal variances
        clamped = np.maximum(self.model.covars_, 0.5)
        object.__setattr__(self.model, '_covars_', clamped)

        # Fix degenerate transmat rows
        tm = self.model.transmat_.copy()
        zero_rows = tm.sum(axis=1) == 0
        tm[zero_rows] = 1.0 / self.n_states
        self.model.transmat_ = tm

        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def state_sequence(self, X: np.ndarray) -> np.ndarray:
        """Viterbi decode — returns most likely state sequence."""
        X = self.scaler.transform(np.asarray(X, dtype=float))
        return self.model.predict(X)

    def forward_state_dist(self, X: np.ndarray) -> np.ndarray:
        """
        Run forward algorithm on sequence X, push one step ahead.
        Returns predictive state distribution P(next_state | X).
        Shape: (n_states,)
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

    def posterior_features(self, X: np.ndarray) -> np.ndarray:
        """
        Posterior summary vector for use as head features.

        Returns a 1-D array of length N + 2:

            [0 : N]   p       — predictive state distribution (forward output)
            [N]       max_p   — peak probability (posterior confidence)
            [N+1]     entropy — Shannon entropy of p in nats (posterior uncertainty)

        p_next (= p @ T) and delta (= p_next - p) were removed because they are
        deterministic linear functions of p given the fixed transition matrix T,
        so they add no information beyond p while increasing dimensionality and
        collinearity in the logistic head.
        """
        p       = self.forward_state_dist(X)                  # (N,)
        max_p   = float(p.max())
        entropy = float(-np.sum(p * np.log(p + EPS)))         # nats

        return np.concatenate([p, [max_p, entropy]])

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _log_likelihoods(self, X: np.ndarray) -> np.ndarray:
        """Diagonal covariance log-likelihood → (T, n_states)."""
        means  = self.model.means_
        covars = self.model.covars_
        T, d   = X.shape
        log_lik = np.zeros((T, self.n_states), dtype=float)

        for s in range(self.n_states):
            mu  = means[s]
            var = np.maximum(covars[s], EPS)
            diff = X - mu
            log_lik[:, s] = -0.5 * np.sum(
                np.log(2 * np.pi * var) + (diff ** 2) / var, axis=1
            )
        return log_lik

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path) -> "GlobalGaussianHMM":
        with open(path, "rb") as f:
            return pickle.load(f)