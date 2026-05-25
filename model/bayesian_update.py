"""
bayesian_update.py — Tiny helpers for Bayesian belief updates over hidden form states.

After observing a new match outcome, we want to update our belief about
a team's current hidden form state. Two utilities:

  - marginal_emission: collapse the opponent's unknown state out of the
    joint emission tensor, giving an effective per-state likelihood
    P(outcome | team_state = i) marginalized over the opponent.
  - forward_update: one step of the HMM forward filter — push the prior
    belief through the transition matrix, multiply by the likelihood,
    and renormalize to obtain the posterior over the team's form state.
"""

import numpy as np

EPS = 1e-12


def marginal_emission(joint_tensor: np.ndarray,
                      opp_state_dist: np.ndarray,
                      outcome: int) -> np.ndarray:
    """v[i] = sum_j opp_state_dist[j] * joint_tensor[i, j, outcome].

    Marginalizes the opponent's hidden state out of the joint emission
    to produce a likelihood vector over the team's own hidden states.
    """
    T = np.asarray(joint_tensor, dtype=float)
    q = np.asarray(opp_state_dist, dtype=float)
    return T[:, :, int(outcome)] @ q


def forward_update(prior_state_dist: np.ndarray,
                   transmat: np.ndarray,
                   marginal_emission_col: np.ndarray) -> np.ndarray:
    """One step of HMM forward filtering with a marginalized emission.

      posterior_unnormalized = (prior @ transmat) * likelihood
      posterior = normalize(posterior_unnormalized)
    """
    prior = np.asarray(prior_state_dist, dtype=float)
    A = np.asarray(transmat, dtype=float)
    lik = np.asarray(marginal_emission_col, dtype=float)

    pushed = prior @ A                # predict step (propagate through dynamics)
    unnorm = pushed * lik             # update step (multiply by likelihood)
    total = unnorm.sum()
    if total <= EPS:
        # Degenerate case: fall back to a uniform belief rather than NaNs.
        return np.full_like(unnorm, 1.0 / len(unnorm))
    return unnorm / total
