"""
hmm_team.py — Per-team N-state Categorical HMM.

Wraps hmmlearn's CategoricalHMM to model a single team's hidden "form"
over time. Observations are match outcomes coded as {0=Loss, 1=Draw, 2=Win}.
Hidden states are relabeled after training so index 0 = worst form,
index n_states-1 = best form, ordered by P(Win).

n_states is configurable (default 5). More states give finer form
granularity but need more data to converge — 5 works well with ~50+
matches per team.
"""

import pickle
import numpy as np
from hmmlearn.hmm import CategoricalHMM
from scipy.special import logsumexp

RANDOM_SEED    = 42
EPS            = 1e-12
ELO_STATE_SCALE = 600.0


class TeamHMM:
    """An N-state Categorical HMM for one team's outcome sequence."""

    def __init__(self, n_states: int = 5):
        self.n_states = n_states
        self.model = CategoricalHMM(
            n_components=n_states,
            n_iter=200,
            tol=1e-3,
            random_state=RANDOM_SEED,
            init_params="ste",
        )
        self._train_outcomes = None

    # ---- training ---------------------------------------------------------
    def fit(self, outcomes: np.ndarray) -> "TeamHMM":
        """Fit the HMM on a 1D int array of outcomes in {0, 1, 2}."""
        outcomes = np.asarray(outcomes, dtype=int).ravel()
        X = outcomes.reshape(-1, 1)
        self.model.n_features = 3
        self.model.fit(X, lengths=[len(outcomes)])
        self._relabel_states()
        self._train_outcomes = outcomes.copy()
        return self

    def _relabel_states(self) -> None:
        """Reorder states so index 0..n-1 = increasing P(Win)."""
        emis     = self.model.emissionprob_       # (n_states, 3)
        win_prob = emis[:, 2]                     # P(Win | state)
        order    = np.argsort(win_prob)           # lowest -> highest
        self.model.startprob_    = self.model.startprob_[order]
        self.model.transmat_     = self.model.transmat_[order][:, order]
        self.model.emissionprob_ = self.model.emissionprob_[order]

    # ---- predictive inference ---------------------------------------------
    def predictive_state_dist(self,
                              prior_outcomes: np.ndarray,
                              elo_advantage: float = 0.0) -> np.ndarray:
        """P(state_t | outcomes_{1..t-1}, elo_advantage).

        Runs the forward algorithm in log-space then pushes one step forward
        through the transition matrix. Applies a Bayesian Elo nudge when
        elo_advantage != 0: higher-rated teams get mass shifted toward
        high-index (strong) states.
        """
        startprob = np.asarray(self.model.startprob_,    dtype=float)
        transmat  = np.asarray(self.model.transmat_,     dtype=float)
        emis      = np.asarray(self.model.emissionprob_, dtype=float)

        prior_outcomes = np.asarray(prior_outcomes, dtype=int).ravel()

        if prior_outcomes.size == 0:
            pred = startprob.copy()
        else:
            log_start = np.log(startprob + EPS)
            log_trans = np.log(transmat  + EPS)
            log_emis  = np.log(emis      + EPS)   # (n_states, 3)

            alpha = log_start + log_emis[:, prior_outcomes[0]]
            for o in prior_outcomes[1:]:
                alpha = (
                    logsumexp(alpha[:, None] + log_trans, axis=0)
                    + log_emis[:, o]
                )

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

    # ---- persistence ------------------------------------------------------
    def save(self, path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path) -> "TeamHMM":
        with open(path, "rb") as f:
            return pickle.load(f)