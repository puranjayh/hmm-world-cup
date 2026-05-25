"""
Sanity tests for the HMM football predictor.

These are quick property-based checks that exercise the math without
needing the full CSV — they use small synthetic sequences so they run
in well under a second each.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from model.hmm_team import TeamHMM
from model.joint_emission import build_joint_tensor
from model.predictor import Predictor
from model.bayesian_update import marginal_emission, forward_update


# ----- TeamHMM ---------------------------------------------------------

def _synthetic_outcomes(n: int, win_rate: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Mix: with prob win_rate emit Win(2); else uniformly Loss(0) or Draw(1).
    out = rng.choice([0, 1, 2], size=n, p=[(1 - win_rate) / 2, (1 - win_rate) / 2, win_rate])
    return out.astype(int)


def test_teamhmm_fits_and_relabels_by_win_prob():
    seq = _synthetic_outcomes(120, win_rate=0.5, seed=0)
    hmm = TeamHMM().fit(seq)
    win_probs = hmm.model.emissionprob_[:, 2]
    # After relabel, P(Win) must be non-decreasing across state index 0..2.
    assert win_probs[0] <= win_probs[1] + 1e-9
    assert win_probs[1] <= win_probs[2] + 1e-9


def test_predictive_state_dist_normalizes():
    seq = _synthetic_outcomes(60, win_rate=0.4, seed=1)
    hmm = TeamHMM().fit(seq)

    # Empty prior -> equals startprob.
    p0 = hmm.predictive_state_dist(np.empty(0, dtype=int))
    assert p0.shape == (3,)
    assert np.isclose(p0.sum(), 1.0)
    assert np.allclose(p0, hmm.model.startprob_)

    # Non-empty prior still normalizes.
    p1 = hmm.predictive_state_dist(seq[:30])
    assert p1.shape == (3,)
    assert np.isclose(p1.sum(), 1.0)
    assert (p1 >= 0).all()


# ----- joint_emission --------------------------------------------------

def _toy_train_df() -> pd.DataFrame:
    """Two teams 'A' and 'B' playing alternating matches over 40 dates."""
    rng = np.random.default_rng(7)
    rows = []
    base = pd.Timestamp("2010-01-01")
    for i in range(40):
        d = base + pd.Timedelta(days=i * 7)
        # outcome from A's perspective
        oa = int(rng.choice([0, 1, 2], p=[0.3, 0.2, 0.5]))
        ob = {0: 2, 1: 1, 2: 0}[oa]   # mirror for opponent
        rows.append({"date": d, "team": "A", "opponent": "B", "outcome": oa})
        rows.append({"date": d, "team": "B", "opponent": "A", "outcome": ob})
    return pd.DataFrame(rows)


def test_joint_tensor_shape_and_normalization():
    df = _toy_train_df()
    seqs_a = df[df.team == "A"].sort_values("date")["outcome"].to_numpy(int)
    seqs_b = df[df.team == "B"].sort_values("date")["outcome"].to_numpy(int)
    hmms = {"A": TeamHMM().fit(seqs_a), "B": TeamHMM().fit(seqs_b)}

    tensor, diag = build_joint_tensor(df, hmms)
    assert tensor.shape == (3, 3, 3)
    # Each (team_state, opp_state) slice over outcomes sums to 1.
    sums = tensor.sum(axis=-1)
    assert np.allclose(sums, 1.0, atol=1e-9)
    assert diag["matches_used"] == 40
    assert diag["matches_skipped"] == 0


# ----- predictor -------------------------------------------------------

def test_predictor_no_leakage_and_sums_to_one():
    df = _toy_train_df()
    seqs_a = df[df.team == "A"].sort_values("date")["outcome"].to_numpy(int)
    seqs_b = df[df.team == "B"].sort_values("date")["outcome"].to_numpy(int)
    hmms = {"A": TeamHMM().fit(seqs_a), "B": TeamHMM().fit(seqs_b)}
    tensor, _ = build_joint_tensor(df, hmms)
    pred = Predictor(team_hmms=hmms, joint_tensor=tensor, history_df=df)

    # Prediction at first date -> no priors, so dist equals startprob for both.
    first_date = df["date"].min()
    out = pred.predict("A", "B", first_date)
    total = out["Loss"] + out["Draw"] + out["Win"]
    assert abs(total - 1.0) < 1e-9
    assert np.allclose(out["state_team"], hmms["A"].model.startprob_)
    assert np.allclose(out["state_opp"], hmms["B"].model.startprob_)

    # Prediction far in the future uses all observations.
    far = df["date"].max() + pd.Timedelta(days=1)
    out2 = pred.predict("A", "B", far)
    assert abs(out2["Loss"] + out2["Draw"] + out2["Win"] - 1.0) < 1e-9


def test_predictor_unknown_team_uniform():
    df = _toy_train_df()
    seqs_a = df[df.team == "A"].sort_values("date")["outcome"].to_numpy(int)
    hmms = {"A": TeamHMM().fit(seqs_a)}
    tensor = np.full((3, 3, 3), 1.0 / 3.0)
    pred = Predictor(team_hmms=hmms, joint_tensor=tensor, history_df=df)
    out = pred.predict("A", "UnknownTeam", df["date"].max())
    # Uniform tensor -> any state combo gives uniform output.
    assert abs(out["Win"] - 1.0 / 3.0) < 1e-9


# ----- bayesian_update -------------------------------------------------

def test_marginal_emission_and_forward_update():
    tensor = np.random.default_rng(0).dirichlet([1, 1, 1], size=(3, 3))  # shape (3,3,3)
    opp_dist = np.array([0.2, 0.5, 0.3])
    v = marginal_emission(tensor, opp_dist, outcome=2)
    assert v.shape == (3,)
    assert (v >= 0).all()

    prior = np.array([0.3, 0.4, 0.3])
    transmat = np.array([[0.7, 0.2, 0.1],
                         [0.2, 0.6, 0.2],
                         [0.1, 0.3, 0.6]])
    post = forward_update(prior, transmat, v)
    assert post.shape == (3,)
    assert abs(post.sum() - 1.0) < 1e-9
    assert (post >= 0).all()


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
