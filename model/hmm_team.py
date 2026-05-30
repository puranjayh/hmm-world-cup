"""
hmm_team.py — Per-team 3-state Categorical HMM.

This module wraps hmmlearn's CategoricalHMM to model a single team's
hidden "form" over time. Observations are match outcomes coded as
{0=Loss, 1=Draw, 2=Win}. The three hidden states are relabeled after
training so index 0 = Poor Form, 1 = Neutral Form, 2 = Peak Form,
ordered by P(Win).

Baum-Welch (the EM algorithm inside hmmlearn) in 4 lines:
  1. E-step: given current params, compute posterior over hidden states
     for every time step using forward-backward.
  2. M-step: re-estimate start, transition, and emission probabilities
     by counting expected transitions and emissions weighted by the
     posteriors.
  3. Iterate E and M until the data log-likelihood stops improving
     (controlled by `tol` and `n_iter`).
  4. The result is a local maximum of the likelihood — random init
     matters, so we fix random_state for reproducibility.
"""

import pickle
import numpy as np
from hmmlearn.hmm import CategoricalHMM
from scipy.special import logsumexp

RANDOM_SEED = 42
EPS = 1e-12
# Governs how strongly an ELO advantage shifts the predictive form-state
# distribution.  The effective Peak/Poor likelihood ratio is exp(2*delta/S).
# At S=600: a 300-pt gap → ~2.7x, keeping ELO secondary to the HMM history.
# Increase toward 1000+ for a subtler nudge; decrease toward 400 for stronger.
ELO_STATE_SCALE = 300.0


class TeamHMM:
    """A 3-state Categorical HMM for one team's outcome sequence."""

    def __init__(self):
        # n_features=3 because outcomes are {0,1,2}; init_params="ste"
        # asks hmmlearn to initialize start, transitions, emissions.
        self.model = CategoricalHMM(
            n_components=3,
            n_iter=200,
            tol=1e-3,
            random_state=RANDOM_SEED,
            init_params="ste",
        )
        self._train_outcomes = None

    # ---- training -----------------------------------------------------
    def fit(self, outcomes: np.ndarray) -> "TeamHMM":
        """Fit the HMM on a 1D int array of outcomes in {0,1,2}."""
        outcomes = np.asarray(outcomes, dtype=int).ravel()
        # hmmlearn requires a 2D column vector X plus per-sequence lengths.
        X = outcomes.reshape(-1, 1)
        self.model.n_features = 3  # explicit so emissionprob has 3 cols
        self.model.fit(X, lengths=[len(outcomes)])
        self._relabel_states()
        self._train_outcomes = outcomes.copy()
        return self

    def _relabel_states(self) -> None:
        """Reorder states so index 0..2 = increasing P(Win).

        We permute startprob_, transmat_, and emissionprob_ together so
        the model stays mathematically identical, only renamed.
        """
        emis = self.model.emissionprob_           # shape (3, 3) [state, obs]
        win_probs = emis[:, 2]                    # P(Win | state)
        order = np.argsort(win_probs)             # lowest -> highest P(Win)
        self.model.startprob_ = self.model.startprob_[order]
        self.model.transmat_ = self.model.transmat_[order][:, order]
        self.model.emissionprob_ = self.model.emissionprob_[order]

    # ---- predictive inference -----------------------------------------
    def predictive_state_dist(self,
                              prior_outcomes: np.ndarray,
                              elo_advantage: float = 0.0) -> np.ndarray:
        """P(state_t | outcomes_{1..t-1}, elo_advantage) — the predictive state dist.

        Uses the forward algorithm in log-space (with log-sum-exp) for
        numerical stability, then pushes one step forward through the
        transition matrix to get the distribution over the *next*
        (unobserved) match's hidden state.

        After the forward pass a Bayesian update is applied when
        elo_advantage != 0:

            P(state | history, elo) ∝ P(state | history) × exp(state_index
                                        × elo_advantage / ELO_STATE_SCALE)

        Positive elo_advantage (team rated higher) shifts mass toward Peak
        Form (state 2); negative shifts mass toward Poor Form (state 0).
        ELO_STATE_SCALE controls the magnitude — the HMM history remains
        the primary driver.
        """
        startprob = np.asarray(self.model.startprob_, dtype=float)
        transmat = np.asarray(self.model.transmat_, dtype=float)
        emis = np.asarray(self.model.emissionprob_, dtype=float)

        prior_outcomes = np.asarray(prior_outcomes, dtype=int).ravel()

        # No history -> the predictive dist for t=1 is just startprob.
        if prior_outcomes.size == 0:
            pred = startprob.copy()
        else:
            log_start = np.log(startprob + EPS)
            log_trans = np.log(transmat + EPS)
            log_emis = np.log(emis + EPS)              # shape (3 states, 3 obs)

            # Forward pass: alpha_t(i) = log P(o_{1..t}, state_t=i)
            # Step 1 (initial): alpha_1 = log_start + log_emis[:, o_1]
            alpha = log_start + log_emis[:, prior_outcomes[0]]

            # Steps 2..T: alpha_t = logsumexp_i' (alpha_{t-1} + log_trans[i', i])
            #                      + log_emis[:, o_t]
            for o in prior_outcomes[1:]:
                # alpha[:, None] has shape (3,1); log_trans shape (3,3).
                # Sum along axis=0 collapses the previous state i'.
                alpha = logsumexp(alpha[:, None] + log_trans, axis=0) + log_emis[:, o]

            # Push forward one step to get the predictive dist over state_{T+1}:
            # log P(state_{T+1}=j | o_{1..T}) ∝ logsumexp_i (alpha_T(i) + log_trans[i, j])
            log_pred = logsumexp(alpha[:, None] + log_trans, axis=0)
            # Softmax-normalize to a proper probability vector.
            log_pred -= log_pred.max()                  # numerical stability
            pred = np.exp(log_pred)
            pred = pred / (pred.sum() + EPS)

        if elo_advantage != 0.0:
            # Bayesian likelihood: exp(state_index × elo_advantage / scale).
            # For negative elo_advantage the exponents are negative, so mass
            # shifts toward state 0 (Poor Form) — handles weaker teams correctly.
            elo_likelihood = np.exp(
                np.arange(3, dtype=float) * elo_advantage / ELO_STATE_SCALE
            )
            pred = pred * elo_likelihood
            pred = pred / (pred.sum() + EPS)

        return pred

    # ---- persistence --------------------------------------------------
    def save(self, path) -> None:
        """Pickle the whole TeamHMM (model + cached train outcomes)."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path) -> "TeamHMM":
        """Load a previously pickled TeamHMM."""
        with open(path, "rb") as f:
            return pickle.load(f)
