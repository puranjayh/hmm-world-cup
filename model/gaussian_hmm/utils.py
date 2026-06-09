"""
utils.py — Shared utilities for the global Gaussian HMM pipeline.

Imported by both evaluate_global.py and predictor_global.py to avoid
circular imports.
"""
from __future__ import annotations

import numpy as np
from model.gaussian_hmm.hmm_global import TOURNAMENT_WEIGHTS

# ---------------------------------------------------------------------------
# Elo constants + helpers
# ---------------------------------------------------------------------------
ELO_K     = 30
ELO_SCALE = 400

def _elo_expected(r_a: float, r_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / ELO_SCALE))

def _elo_update(r_a: float, r_b: float, score_a: float) -> tuple[float, float]:
    exp_a   = _elo_expected(r_a, r_b)
    new_r_a = r_a + ELO_K * (score_a - exp_a)
    new_r_b = r_b + ELO_K * ((1 - score_a) - (1 - exp_a))
    return float(new_r_a), float(new_r_b)

def _outcome_to_score(outcome: int) -> float:
    return {2: 1.0, 1: 0.5, 0: 0.0}[outcome]

# ---------------------------------------------------------------------------
# Tournament stage helpers
# ---------------------------------------------------------------------------
_KNOCKOUT_KEYWORDS = [
    "final", "semi", "quarter", "round of", "last 16",
    "knockout", "elimination", "third place",
]

def _is_knockout(tournament_str: str) -> int:
    if not isinstance(tournament_str, str):
        return 0
    t = tournament_str.lower()
    return int(any(k in t for k in _KNOCKOUT_KEYWORDS))

def _tournament_weight_val(tournament_str: str) -> float:
    if not isinstance(tournament_str, str):
        return 1.0
    for key, w in TOURNAMENT_WEIGHTS.items():
        if key.lower() in tournament_str.lower():
            return w
    return 1.0

# ---------------------------------------------------------------------------
# Draw propensity helpers
# ---------------------------------------------------------------------------

def _draw_features(probs_3way: np.ndarray,
                   elo_diffs:  np.ndarray,
                   entropy_a:  np.ndarray,
                   entropy_b:  np.ndarray,
                   is_knockout: np.ndarray) -> np.ndarray:
    elo_closeness = 1.0 / (1.0 + np.abs(elo_diffs) / 100.0)
    return np.column_stack([
        probs_3way[:, 1],
        elo_closeness,
        entropy_a,
        entropy_b,
        entropy_a + entropy_b,
        is_knockout.astype(float),
    ])

def _train_draw_model(X_draw_feats: np.ndarray,
                      outcomes:     np.ndarray,
                      random_seed:  int = 42):
    """Train a binary logistic classifier: draw (1) vs no-draw (0)."""
    from sklearn.linear_model import LogisticRegression
    y_draw = (outcomes == 1).astype(int)
    clf    = LogisticRegression(max_iter=1000, C=0.5, random_state=random_seed)
    clf.fit(X_draw_feats, y_draw)
    return clf


def _blend_draw_probs(probs_3way: np.ndarray,
                      draw_probs: np.ndarray,
                      alpha:      float = 0.3) -> np.ndarray:
    blended  = probs_3way.copy()
    new_draw = (1 - alpha) * probs_3way[:, 1] + alpha * draw_probs
    wl_mass  = probs_3way[:, 0] + probs_3way[:, 2]
    scale    = np.where(wl_mass > 1e-9, (1.0 - new_draw) / wl_mass, 0.5)
    blended[:, 0] = probs_3way[:, 0] * scale
    blended[:, 1] = new_draw
    blended[:, 2] = probs_3way[:, 2] * scale
    row_sums = blended.sum(axis=1, keepdims=True)
    return blended / np.where(row_sums > 0, row_sums, 1.0)