"""
hmm_team_gaussian.py — Per-team 3-state Gaussian HMM using continuous match features.

Features used (all known BEFORE the match):
    [elo_diff, rolling_win_rate_5, rolling_goal_diff_5, tournament_weight]

Key design decisions:
- NO post-match features (goal_diff, goals_for, goals_against removed)
- Scaler is fitted on the GLOBAL training set (passed in), not per team —
  this preserves inter-team signal (France elo_diff=+300 vs Panama elo_diff=-300)
- Full covariance with regularisation to avoid hmmlearn's min_covar bug
"""

import pickle
import numpy as np
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp

RANDOM_SEED     = 42
EPS             = 1e-12
ELO_STATE_SCALE = 600.0

FEATURE_NAMES = [
    'goal_diff',
    'result',
    'elo_diff',
    'win_vs_strong',
    'tournament_weight',
    'days_since_last_match',
]
N_FEATURES = len(FEATURE_NAMES)


class TeamGaussianHMM:
    """A 3-state Gaussian HMM for one team's continuous feature sequences."""

    def __init__(self, n_states: int = 7, scaler=None):
        self.n_states = n_states
        self.scaler   = scaler   # global scaler fitted on full train set
        self.model    = GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=200,
            tol=1e-3,
            random_state=RANDOM_SEED,
            init_params="stmc",
            min_covar=1e-3,
        )

    # ---- training ---------------------------------------------------------
    def fit(self, feature_matrix: np.ndarray) -> "TeamGaussianHMM":
        X = np.asarray(feature_matrix, dtype=float)
        if self.scaler is not None:
            X = self.scaler.transform(X)   # global scale — preserves team differences

        self.model.fit(X, lengths=[len(X)])

        # Regularise covariances — hmmlearn doesn't always respect min_covar
        d   = X.shape[1]
        reg = 0.1 * np.eye(d)
        self.model.covars_ = np.array([c + reg for c in self.model.covars_])

        self._relabel_states()
        return self

    def _relabel_states(self) -> None:
        """Reorder states by mean elo_diff (feature 0) — lowest to highest."""
        order = np.argsort(self.model.means_[:, 0])
        self.model.startprob_ = self.model.startprob_[order]
        self.model.transmat_  = self.model.transmat_[order][:, order]
        self.model.means_     = self.model.means_[order]
        self.model.covars_    = self.model.covars_[order]

    # ---- predictive inference ---------------------------------------------
    def predictive_state_dist(self,
                              prior_features: np.ndarray,
                              elo_advantage: float = 0.0) -> np.ndarray:
        startprob = np.asarray(self.model.startprob_, dtype=float)
        transmat  = np.asarray(self.model.transmat_,  dtype=float)

        prior_features = np.asarray(prior_features, dtype=float)
        if prior_features.ndim == 1:
            prior_features = prior_features.reshape(1, -1)

        if prior_features.shape[0] == 0:
            pred = startprob.copy()
        else:
            if self.scaler is not None:
                X = self.scaler.transform(prior_features)
            else:
                X = prior_features

            log_lik   = self._log_likelihoods(X)
            log_start = np.log(startprob + EPS)
            log_trans = np.log(transmat  + EPS)

            alpha = log_start + log_lik[0]
            for t in range(1, len(log_lik)):
                alpha = logsumexp(alpha[:, None] + log_trans, axis=0) + log_lik[t]

            log_pred = logsumexp(alpha[:, None] + log_trans, axis=0)
            log_pred -= log_pred.max()
            pred = np.exp(log_pred)
            pred = pred / (pred.sum() + EPS)

        if elo_advantage != 0.0:
            elo_likelihood = np.exp(
                np.arange(self.n_states, dtype=float)
                * elo_advantage / ELO_STATE_SCALE
            )
            pred = pred * elo_likelihood
            pred = pred / (pred.sum() + EPS)

        return pred

    def _log_likelihoods(self, X: np.ndarray) -> np.ndarray:
        """Full covariance multivariate Gaussian log-likelihood → (T, n_states)."""
        means  = self.model.means_
        covars = self.model.covars_
        T, d   = X.shape
        log_lik = np.zeros((T, self.n_states), dtype=float)

        for s in range(self.n_states):
            mu  = means[s]
            cov = covars[s]
            try:
                cov_inv          = np.linalg.inv(cov)
                sign, log_det    = np.linalg.slogdet(cov)
                if sign <= 0:
                    raise np.linalg.LinAlgError
            except np.linalg.LinAlgError:
                cov_inv = np.eye(d)
                log_det = 0.0
            diff = X - mu
            maha = np.sum(diff @ cov_inv * diff, axis=1)
            log_lik[:, s] = -0.5 * (d * np.log(2 * np.pi) + log_det + maha)

        return log_lik

    # ---- persistence ------------------------------------------------------
    def save(self, path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path) -> "TeamGaussianHMM":
        with open(path, "rb") as f:
            return pickle.load(f)