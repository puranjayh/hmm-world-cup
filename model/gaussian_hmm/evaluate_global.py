"""
evaluate_global.py — Benchmark the global Gaussian HMM against baselines.

Improvements over v1:
  1. Dynamic Elo updating  — after each test match, both teams' Elo ratings
     are updated using the standard K-factor formula before the next prediction.
  2. Tournament stage feature — is_knockout + tournament_weight passed to head.
  3. Draw propensity model — a secondary binary classifier (draw vs no-draw)
     trained on entropy/elo-closeness features; blended into final probs.
  4. Confidence gating — reported alongside accuracy at multiple thresholds.
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

# This script prints box-drawing and arrow characters. Windows consoles default
# to cp1252, and redirecting stdout to a file makes the encode error fatal —
# which killed the run *after* artifacts were written but *before* the metrics
# JSON, leaving the two out of sync.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

# ---------------------------------------------------------------------------
# Determinism — MUST run before numpy (and anything importing it) is loaded.
# ---------------------------------------------------------------------------
# BLAS sizes its thread pool at import time and parallel reductions sum partial
# results in nondeterministic order. That is enough to perturb the logistic
# head's coefficients in the last decimals, flip argmax on matches sitting near
# a decision boundary, and move reported accuracy by a full match between two
# runs of identical code (measured: copa_2024 accuracy 0.4923 vs 0.4769).
# Every model here already sets random_state, so this is the remaining source.
# Set REPRODUCIBLE=0 to opt out and use all cores.
if os.environ.get("REPRODUCIBLE", "1") != "0":
    for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

from model.config import ARTIFACTS_DIR
from model.data_loader import load_matches
from model.gaussian_hmm.hmm_global import (
    GlobalGaussianHMM,
    FEATURE_NAMES,
    N_STATES,
    TOURNAMENT_WEIGHTS,
    _tournament_sample_weight,
)
from model.gaussian_hmm.utils import (
    ELO_K,
    ELO_SCALE,
    FormTracker,
    _elo_update,
    _outcome_to_score,
    _is_knockout,
    _tournament_weight_val,
    _draw_features,
    _blend_draw_probs,
    _train_draw_model,
    seed_live_elo,
)

WINDOW = 3  # last N matches for state inference — empirically best

# Elo regimes scored at test time.
#   dynamic   — ratings are seeded from history and rolled forward with the
#               K-factor formula after every match in the window, so each
#               fixture is priced off ratings that already reflect the
#               tournament's earlier results. This is also what the 2026
#               simulator must do, since unplayed matches have no published
#               rating. Primary.
#   published — the static `elo_diff` column, i.e. the quantity the head was
#               trained on. Kept as a reference so the gap stays visible.
ELO_MODES        = ("dynamic", "published")
PRIMARY_ELO_MODE = "dynamic"

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*transmat_.*")

RANDOM_SEED = 42

# Test windows are selected by tournament name as well as by date. Filtering on
# dates alone let other competitions running in the same weeks leak into the
# window: the "Euro 2024" window was 85 fixtures, of which only 51 were Euro
# matches — the rest were Copa América, the COSAFA Cup, the Oceania Nations Cup
# and friendlies, i.e. the Copa run was scoring a large slice of the same
# matches as the Euro run.
def _tourn_window(names, start, end):
    names = tuple(names)
    return lambda df: df[
        (df["date"] >= start) & (df["date"] <= end)
        & (df["tournament"].isin(names))
    ]


EVAL_RUNS = [
    {
        "tag":           "wc_2018",
        "train_cutoff":  "2018-06-13",
        "test_filter":   _tourn_window(["World Cup"], "2018-06-14", "2018-07-15"),
        "label":         "2018 World Cup",
        "is_tournament": True,
        "save_artifacts": False,
    },
    {
        "tag":           "wc_2022",
        "train_cutoff":  "2022-11-19",
        "test_filter":   _tourn_window(["World Cup"], "2022-11-20", "2022-12-18"),
        "label":         "2022 World Cup",
        "is_tournament": True,
        "save_artifacts": False,
    },
    {
        "tag":          "euro_2024",
        "train_cutoff": "2024-06-14",
        "test_filter":  _tourn_window(["European Championship"],
                                      "2024-06-14", "2024-07-14"),
        "label": "UEFA Euro 2024",
        "is_tournament": True,
    },
    {
        "tag":          "copa_2024",
        "train_cutoff": "2024-06-20",
        "test_filter":  _tourn_window(["Copa América", "Copa America"],
                                      "2024-06-20", "2024-07-14"),
        "label": "Copa América 2024",
        "is_tournament": True,
    },
    # Production run. No test window — trains on the full corrected history and
    # writes global_hmm.pkl / head.pkl / draw_model.pkl, which are what
    # wc2026simulator.py loads. Without this entry nothing regenerates the
    # artifacts, so the deployed 2026 model silently stays on whatever data it
    # was last fit to (including the pre-fix, Elo-leaked dataset).
    # Keep this LAST so the benchmark windows above are unaffected by it.
    {
        "tag":            "production_2026",
        "train_cutoff":   "2099-01-01",
        "test_filter":    lambda df: df.iloc[0:0],
        "label":          "Production (WC 2026 artifacts)",
        "is_tournament":  True,
        "save_artifacts": True,
    },
]

TREE_FEATURES = [
    'ewa_win_rate',
    'ewa_goal_diff',
    'rolling_win_vs_strong_5',
    'rolling_goal_diff_std_5',
    'rolling_win_rate_std_5',
    'ewa_win_rate_momentum',
    'ewa_goal_diff_momentum'
]

# ---------------------------------------------------------------------------
# Elo helpers
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _metrics(probs, outcomes):
    eps     = 1e-12
    n       = len(outcomes)
    p_true  = probs[np.arange(n), outcomes]
    log_loss = float(-np.mean(np.log(np.clip(p_true, eps, 1.0))))
    one_hot  = np.zeros_like(probs)
    one_hot[np.arange(n), outcomes] = 1.0
    brier    = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))
    accuracy = float(np.mean(np.argmax(probs, axis=1) == outcomes))
    cum_p    = np.cumsum(probs,   axis=1)
    cum_a    = np.cumsum(one_hot, axis=1)
    rps      = float(np.mean(np.sum((cum_p - cum_a) ** 2, axis=1)
                             / (probs.shape[1] - 1)))
    return {"n": int(n), "log_loss": round(log_loss, 4), "brier": round(brier, 4),
            "accuracy": round(accuracy, 4), "rps": round(rps, 4)}


def _metrics_no_draw(probs, outcomes):
    """
    W/L accuracy, scored the way the Live Test sheet scores it.

    On the drawn matches the sheet leaves the "Accurate (W/L)" cell blank, and
    on the decided ones it asks only whether P(win) > P(loss) — the draw column
    is ignored, not treated as a competing prediction. Feeding the full 3-way
    vector to argmax instead (what this did before) marked a match wrong
    whenever draw was the modal class even if the win/loss call was right, which
    understated every model's W/L accuracy — most of all the draw-blended one.

    So: restrict to decided matches, drop the draw column, renormalise.
    """
    mask = outcomes != 1
    if mask.sum() == 0:
        return {}
    two = probs[mask][:, [0, 2]]
    two = two / np.clip(two.sum(axis=1, keepdims=True), 1e-12, None)
    y   = (outcomes[mask] == 2).astype(int)   # 1 = win, 0 = loss
    return _metrics(two, y)


def _metrics_at_thresholds(probs, outcomes, thresholds=(0.40, 0.45, 0.50, 0.55, 0.60)):
    """
    Accuracy and coverage at various confidence thresholds.
    Only predictions where max(prob) >= threshold are evaluated.
    """
    results = {}
    for t in thresholds:
        confident = np.max(probs, axis=1) >= t
        n_conf    = confident.sum()
        if n_conf == 0:
            results[f"thresh_{int(t*100)}"] = {"n": 0, "accuracy": None, "coverage": 0.0}
            continue
        acc = float(np.mean(np.argmax(probs[confident], axis=1) == outcomes[confident]))
        cov = float(n_conf / len(outcomes))
        results[f"thresh_{int(t*100)}"] = {
            "n": int(n_conf), "accuracy": round(acc, 4), "coverage": round(cov, 4)
        }
    return results


# ---------------------------------------------------------------------------
# Uncertainty — how much of a gap between two models is real?
# ---------------------------------------------------------------------------
# A single tournament is 30-64 matches. At n=44 one flipped match moves accuracy
# by 2.3 points, which is the same order as the entire spread between the models
# being compared. Reporting a bare accuracy invites reading a one-match swing as
# an improvement, so every accuracy now carries a bootstrap interval, and every
# model is tested against the reference model on the matches they BOTH scored.

N_BOOTSTRAP  = 20000
BOOTSTRAP_CI = 0.95
SIG_REFERENCE = "GlobalGHMM+Draw"


def _correct_flags(probs, outcomes):
    """Per-match 0/1 correctness under both scoring rules.

    Returns (wdl, wl) where `wdl` covers every match and `wl` covers only the
    decided ones, judged on P(win) vs P(loss) with the draw column ignored.
    """
    wdl  = (np.argmax(probs, axis=1) == outcomes).astype(int)
    mask = outcomes != 1
    wl   = ((probs[mask][:, 2] > probs[mask][:, 0]).astype(int)
            == (outcomes[mask] == 2).astype(int)).astype(int)
    return wdl, wl


def _bootstrap_ci(flags, seed=RANDOM_SEED):
    """Percentile bootstrap CI for the mean of a 0/1 vector."""
    flags = np.asarray(flags, dtype=float)
    n = len(flags)
    if n == 0:
        return None, None
    rng  = np.random.default_rng(seed)
    idx  = rng.integers(0, n, size=(N_BOOTSTRAP, n))
    means = flags[idx].mean(axis=1)
    lo = (1.0 - BOOTSTRAP_CI) / 2.0 * 100.0
    return (round(float(np.percentile(means, lo)), 4),
            round(float(np.percentile(means, 100.0 - lo)), 4))


def _mcnemar(flags_a, flags_b):
    """
    Exact McNemar test on paired predictions.

    Every model scores the identical fixture list, so the comparison should be
    paired: only the matches where exactly one of the two is right carry any
    information. `b` counts matches only A got right, `c` only B.
    """
    from scipy.stats import binomtest
    a = np.asarray(flags_a, dtype=int)
    b_only = int(((a == 1) & (np.asarray(flags_b, dtype=int) == 0)).sum())
    c_only = int(((a == 0) & (np.asarray(flags_b, dtype=int) == 1)).sum())
    n_disc = b_only + c_only
    p = (float(binomtest(b_only, n_disc, 0.5).pvalue) if n_disc > 0 else 1.0)
    return {"ref_only_correct": b_only, "other_only_correct": c_only,
            "n_discordant": n_disc, "p_value": round(p, 4)}


def _significance(model_probs, outcomes, reference=SIG_REFERENCE):
    """Accuracy + bootstrap CI per model, and a paired test against `reference`."""
    flags = {name: _correct_flags(p, outcomes) for name, p in model_probs.items()}
    out = {}
    for name, (wdl, wl) in flags.items():
        entry = {}
        for key, f in (("wdl", wdl), ("wl", wl)):
            lo, hi = _bootstrap_ci(f)
            entry[key] = {
                "n":        int(len(f)),
                "accuracy": round(float(np.mean(f)), 4) if len(f) else None,
                "ci_lo":    lo,
                "ci_hi":    hi,
            }
        if name != reference and reference in flags:
            entry["vs_reference"] = {
                "reference": reference,
                "wdl": _mcnemar(flags[reference][0], wdl),
                "wl":  _mcnemar(flags[reference][1], wl),
            }
        out[name] = entry
    return out


def _git_commit() -> str | None:
    """Short commit hash of the working tree, or None outside a repo."""
    import subprocess
    try:
        root = Path(__file__).resolve().parents[2]
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root, capture_output=True, text=True, timeout=10,
        )
        if out.returncode != 0:
            return None
        commit = out.stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root, capture_output=True, text=True, timeout=10,
        )
        if dirty.returncode == 0 and dirty.stdout.strip():
            commit += "-dirty"
        return commit or None
    except Exception:
        return None


def _unique_matches(df):
    """
    One row per distinct fixture, keeping whichever orientation the data has.

    This used to be `df[df["team"] < df["opponent"]]`, which assumes the dataset
    stores both orientations of every match. Before the data_filter.py fix it did
    not — only rows whose `team` was a WC-2026 participant were kept — so a
    fixture like Spain–Italy existed solely as the Spain row and was silently
    dropped for being alphabetically "backwards". That removed 7 of 61 matches
    from the 2018 World Cup window and 20 of 85 from the Euro 2024 window, and
    biased which fixtures survived by team name.

    Deduplicating on the unordered pair instead keeps every fixture exactly once.
    Orientation does not matter downstream: `outcome`, `goal_diff` and
    `elo_diff` are all stored from the row's own `team` perspective, and the
    head mirrors each training row anyway.
    """
    if len(df) == 0:
        return df.sort_values("date").reset_index(drop=True)
    out = df.sort_values(["date", "team", "opponent"], kind="stable").copy()
    a, b = out["team"].astype(str), out["opponent"].astype(str)
    swap = a > b
    out["_pair_lo"] = np.where(swap, b, a)
    out["_pair_hi"] = np.where(swap, a, b)
    out = out.drop_duplicates(subset=["date", "_pair_lo", "_pair_hi"],
                              keep="first")
    return (out.drop(columns=["_pair_lo", "_pair_hi"])
               .sort_values("date", kind="stable").reset_index(drop=True))


def _align_classes(raw, classes, n):
    a = np.zeros((n, 3), float)
    for k, c in enumerate(classes):
        a[:, int(c)] = raw[:, k]
    return a

# ---------------------------------------------------------------------------
# Feature vector construction
# ---------------------------------------------------------------------------

def _build_feature_vec(
    hmm:       GlobalGaussianHMM,
    pf_team:   np.ndarray,
    pf_opp:    np.ndarray,
    elo_diff:  float,
    is_ko:     int   = 0,
    tourn_w:   float = 1.0,
) -> np.ndarray:
    """
    Full feature vector for the logistic head.

    Layout (N=7 → 60 features total):
        outer(p_A, p_B).ravel()     (N²=49)  joint regime interaction
        max_p_A, max_p_B            (2)      HMM confidence
        entropy_A, entropy_B        (2)      HMM uncertainty
        elo_diff                    (1)      rating difference
        elo_diff * max_p_A          (1)      strength × confidence (team)
        elo_diff * max_p_B          (1)      strength × confidence (opp)
        is_knockout                 (1)  NEW tournament stage
        tournament_weight           (1)  NEW match importance
    """
    N       = hmm.n_states
    p_A     = pf_team[:N];  max_p_A = pf_team[N];  ent_A = pf_team[N + 1]
    p_B     = pf_opp[:N];   max_p_B = pf_opp[N];   ent_B = pf_opp[N + 1]
    outer   = np.outer(p_A, p_B).ravel()

    return np.concatenate([
        outer,
        [max_p_A, max_p_B],
        [ent_A,   ent_B],
        [elo_diff],
        [elo_diff * max_p_A],
        [elo_diff * max_p_B],
        [float(is_ko)],          # NEW
        [float(tourn_w)],        # NEW
    ])

# ---------------------------------------------------------------------------
# Build logistic head training data
# ---------------------------------------------------------------------------

def _build_head_features(
    train_df: pd.DataFrame,
    hmm:      GlobalGaussianHMM,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns X, y, elo_diffs, entropy_a_arr, entropy_b_arr for head + draw model.
    """
    sorted_df = train_df.sort_values("date").reset_index(drop=True)
    per_team  = {}
    for team, grp in sorted_df.groupby("team", sort=False):
        per_team[team] = {
            "dates":    grp["date"].to_numpy(),
            "features": grp[FEATURE_NAMES].fillna(0).to_numpy(dtype=float),
        }

    def posterior(team, date):
        rec = per_team.get(team)
        if rec is None:
            N    = hmm.n_states
            unif = np.full(N, 1.0 / N)
            return np.concatenate([unif, [1.0 / N, np.log(N)]])
        idx   = np.searchsorted(rec["dates"],
                                np.datetime64(pd.Timestamp(date)), side="left")
        feats = rec["features"][max(0, idx - WINDOW): idx]
        return hmm.posterior_features(feats)

    head_matches = _unique_matches(train_df).dropna(subset=["outcome", "elo_diff"])

    X_list, y_list       = [], []
    elo_list             = []
    ent_a_list, ent_b_list = [], []
    is_ko_list, tw_list  = [], []

    for _, row in head_matches.iterrows():
        pt = posterior(row["team"],     row["date"])
        po = posterior(row["opponent"], row["date"])

        is_ko  = _is_knockout(row.get("tournament", ""))
        tw     = _tournament_weight_val(row.get("tournament", ""))
        elo_d  = float(row["elo_diff"])
        out    = int(row["outcome"])

        # Forward ordering: (team, opponent)
        fv = _build_feature_vec(hmm, pt, po, elo_d, is_ko, tw)
        X_list.append(fv)
        y_list.append(out)
        elo_list.append(elo_d)
        ent_a_list.append(float(pt[hmm.n_states + 1]))
        ent_b_list.append(float(po[hmm.n_states + 1]))
        is_ko_list.append(is_ko)

        # Mirror ordering: (opponent, team) with flipped outcome and negated elo_diff.
        # This forces the outer-product weights to be position-invariant so no
        # alphabetical ordering bias leaks into the head's learned coefficients.
        flipped_out = 2 - out  # win↔loss, draw stays draw
        fv_mirror = _build_feature_vec(hmm, po, pt, -elo_d, is_ko, tw)
        X_list.append(fv_mirror)
        y_list.append(flipped_out)
        elo_list.append(-elo_d)
        ent_a_list.append(float(po[hmm.n_states + 1]))
        ent_b_list.append(float(pt[hmm.n_states + 1]))
        is_ko_list.append(is_ko)

    return (np.array(X_list), np.array(y_list),
            np.array(elo_list), np.array(ent_a_list),
            np.array(ent_b_list), np.array(is_ko_list))

# ---------------------------------------------------------------------------
# Global HMM runner  (with dynamic Elo + draw model + confidence gating)
# ---------------------------------------------------------------------------

def _run_global_hmm(train_df, test_matches, is_tournament=False, save_artifacts=False):
    # ── 1. Build per-team sequences ──────────────────────────────────────────
    per_team_feats = {}
    lengths        = []
    all_X          = []
    all_weights    = []

    for team, grp in train_df.groupby("team"):
        grp_sorted = grp.sort_values("date")
        feats      = grp_sorted[FEATURE_NAMES].fillna(0).to_numpy(float)
        if len(feats) >= 5:
            per_team_feats[team] = feats
            all_X.append(feats)
            lengths.append(len(feats))
            if "tournament" in train_df.columns:
                w = _tournament_sample_weight(grp_sorted["tournament"].to_numpy())
            else:
                w = np.ones(len(feats))
            all_weights.append(w)

    X_all = np.vstack(all_X)
    W_all = np.concatenate(all_weights) if all_weights else None

    # ── 2. Fit global HMM ────────────────────────────────────────────────────
    print(f"  Fitting global HMM on {X_all.shape[0]} observations, "
          f"{len(lengths)} team sequences …")
    if W_all is not None:
        unique_w = np.unique(np.round(W_all).astype(int))
        print(f"  Sample weight range: [{W_all.min():.1f}, {W_all.max():.1f}]  "
              f"(rounded int values: {unique_w})")

    hmm = GlobalGaussianHMM(n_states=N_STATES)
    hmm.fit(X_all, lengths=lengths, sample_weight=W_all)

    print("\n===== STATE MEANS =====")
    for i, mean in enumerate(hmm.model.means_):
        print(f"State {i}:")
        for feat, val in zip(FEATURE_NAMES, mean):
            print(f"  {feat}: {val:.3f}")

    print("\n===== TRANSITION MATRIX =====")
    print(np.round(hmm.model.transmat_, 3))

    # ── 3. Train logistic head + draw model ──────────────────────────────────
    print("  Training logistic head on posterior summary features …")
    X_head, y_head, elo_arr, ent_a_arr, ent_b_arr, is_ko_arr = \
        _build_head_features(train_df, hmm)
    X_head = np.nan_to_num(X_head, nan=0.0, posinf=0.0, neginf=0.0)

    head = LogisticRegression(max_iter=2000, C=1.0, random_state=RANDOM_SEED)
    head.fit(X_head, y_head)
    n_feats = X_head.shape[1]
    print(f"  Head trained on {len(y_head)} matches, {n_feats} features "
          f"(N²={N_STATES**2} joint + 2 conf + 2 ent + 1 elo "
          f"+ 2 elo×conf + 2 stage interactions)")

    # Train draw propensity model on training head predictions
    head_raw_probs = _align_classes(
        head.predict_proba(X_head), head.classes_, n=len(y_head)
    )
    X_draw = _draw_features(head_raw_probs, elo_arr, ent_a_arr, ent_b_arr, is_ko_arr)
    draw_model = _train_draw_model(X_draw, y_head)
    print(f"  Draw propensity model trained on {len(y_head)} matches.")

    # ── Save artifacts for wc2026_simulator.py ───────────────────────────────
    if save_artifacts:
        import pickle as _pickle
        _art = ARTIFACTS_DIR / "gaussian"
        _art.mkdir(parents=True, exist_ok=True)
        hmm.save(_art / "global_hmm.pkl")
        with open(_art / "head.pkl", "wb") as _f:
            _pickle.dump(head, _f)
        with open(_art / "draw_model.pkl", "wb") as _f:
            _pickle.dump(draw_model, _f)
        print(f"  Artifacts saved → {_art}")
    # ─────────────────────────────────────────────────────────────────────────

    # ── 4. Test-time prediction ──────────────────────────────────────────────
    if len(test_matches) == 0:
        empty = np.zeros((0, 3), float)
        return {
            mode: {
                "probs_raw":   empty,
                "probs_blend": empty,
                "elo_diff":    np.zeros(0),
                "ent_a":       np.zeros(0),
                "ent_b":       np.zeros(0),
                "is_ko":       np.zeros(0, int),
                "form_team":   np.zeros((0, len(FEATURE_NAMES))),
                "form_opp":    np.zeros((0, len(FEATURE_NAMES))),
                "elo_team":    np.zeros(0),
                "elo_opp":     np.zeros(0),
            }
            for mode in ELO_MODES
        }

    def _predict_window(elo_mode: str):
        """
        Score the test window once, under one Elo regime.

        elo_mode="published": use the `elo_diff` column, i.e. the same published
            differential the head was TRAINED on. merge_asof already folds in any
            rating published mid-tournament, so in-tournament information is kept
            and (post the data_filter.py fix) it is strictly pre-match.
        elo_mode="dynamic": seed ratings from history and roll them forward with
            the local K-factor formula. This is what the 2026 simulator must do,
            since future matches have no published rating yet — so it is reported
            to show what the production path actually costs.

        The head was fit on published differentials, so "published" is the
        matched distribution and is the primary number. Scoring it on "dynamic"
        evaluated the model on a distribution it never saw.
        """
        # Fresh per-run state so the two modes cannot contaminate each other.
        live_elo, default_elo = seed_live_elo(train_df)
        tracker = FormTracker(train_df)

        def posterior_for(team):
            feats = tracker.sequence(team, WINDOW)
            return hmm.posterior_features(feats)

        n = len(test_matches)
        probs_raw      = np.zeros((n, 3), float)
        elo_diffs_test = np.zeros(n)
        elo_team_test  = np.zeros(n)
        elo_opp_test   = np.zeros(n)
        ent_a_test     = np.zeros(n)
        ent_b_test     = np.zeros(n)
        is_ko_test     = np.zeros(n, dtype=int)
        form_team      = np.zeros((n, len(FEATURE_NAMES)))
        form_opp       = np.zeros((n, len(FEATURE_NAMES)))

        for i, (_, row) in enumerate(test_matches.iterrows()):
            team = row["team"];  opp = row["opponent"]

            r_team = live_elo.get(team, default_elo)
            r_opp  = live_elo.get(opp,  default_elo)
            elo_d  = (float(row["elo_diff"]) if elo_mode == "published"
                      else r_team - r_opp)

            is_ko = _is_knockout(row.get("tournament", ""))
            tw    = _tournament_weight_val(row.get("tournament", ""))

            # Both the HMM observation sequence and the tree baselines' inputs
            # come from the tracker, so every model sees the same form state —
            # one that already includes the earlier matches of this tournament.
            pt = posterior_for(team)
            po = posterior_for(opp)
            form_team[i] = tracker.pre_match_features(team)
            form_opp[i]  = tracker.pre_match_features(opp)

            fv  = _build_feature_vec(hmm, pt, po, elo_d, is_ko, tw)
            fv  = np.nan_to_num(fv, nan=0.0, posinf=0.0, neginf=0.0)
            raw = head.predict_proba(fv.reshape(1, -1))
            probs_raw[i] = _align_classes(raw, head.classes_, n=1)[0]

            elo_diffs_test[i] = elo_d
            elo_team_test[i]  = r_team
            elo_opp_test[i]   = r_opp
            ent_a_test[i]     = float(pt[hmm.n_states + 1])
            ent_b_test[i]     = float(po[hmm.n_states + 1])
            is_ko_test[i]     = is_ko

            # ---- Post-match updates, applied AFTER the prediction is locked in.
            # Form advances for BOTH teams from the actual goal difference; the
            # opponent's entry is the mirror image, not a copy of the team's row.
            outcome = int(row["outcome"])
            tracker.record_match(
                team, opp,
                goal_diff=float(row["goal_diff"]),
                outcome=outcome,
                team_elo=r_team, opp_elo=r_opp,
            )

            # Roll ratings forward so the next fixture is priced off the updated
            # ratings rather than the ones the teams brought into the tournament.
            new_r_team, new_r_opp = _elo_update(
                r_team, r_opp, _outcome_to_score(outcome)
            )
            live_elo[team] = new_r_team
            live_elo[opp]  = new_r_opp

        X_draw_test = _draw_features(probs_raw, elo_diffs_test,
                                     ent_a_test, ent_b_test, is_ko_test)
        draw_probs  = draw_model.predict_proba(X_draw_test)[:, 1]
        probs_blend = _blend_draw_probs(probs_raw, draw_probs, alpha=0.3)

        return {
            "probs_raw":   probs_raw,
            "probs_blend": probs_blend,
            "elo_diff":    elo_diffs_test,
            "ent_a":       ent_a_test,
            "ent_b":       ent_b_test,
            "is_ko":       is_ko_test,
            "form_team":   form_team,
            "form_opp":    form_opp,
            "elo_team":    elo_team_test,
            "elo_opp":     elo_opp_test,
        }

    return {mode: _predict_window(mode) for mode in ELO_MODES}

# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def _run_elo(train_df, elo_diff_test):
    """Raw-Elo baseline: 3-way logistic on the rating difference alone.

    `elo_diff_test` is supplied by the caller so the baseline is scored on the
    same match-by-match updated ratings the HMM sees, rather than on the static
    pre-tournament published differential.
    """
    train_u = _unique_matches(train_df).dropna(subset=["elo_diff", "outcome"])
    clf     = LogisticRegression(max_iter=1000)
    clf.fit(train_u[["elo_diff"]].to_numpy(float), train_u["outcome"].to_numpy(int))
    X_test = np.asarray(elo_diff_test, dtype=float).reshape(-1, 1)
    raw = clf.predict_proba(X_test)
    return _align_classes(raw, clf.classes_, n=len(X_test))


def _run_tree(train_df, form_test, model_type, elo_test):
    """RF / XGBoost baseline on the rolling-form features PLUS the Elo gap.

    `form_test` is the (n_matches, len(TREE_FEATURES)) matrix produced by the
    FormTracker, so these baselines also see form that advances within the
    tournament instead of the frozen pre-tournament snapshot. `elo_test` is the
    per-match Elo differential the GHMM head is given; feeding it to the trees
    too makes this a same-information comparison rather than one that withholds
    the rating from the baselines (form-only trees trail by ~15 accuracy points).
    """
    avail   = [f for f in TREE_FEATURES if f in train_df.columns]
    train_u = _unique_matches(train_df).dropna(subset=avail + ["elo_diff", "outcome"])
    clf = (
        RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=1)
        if model_type == "rf"
        else XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                           eval_metric="mlogloss",
                           random_state=42, verbosity=0)
    )
    X_train = np.column_stack([
        train_u[avail].to_numpy(float),
        train_u["elo_diff"].to_numpy(float),
    ])
    clf.fit(X_train, train_u["outcome"].to_numpy(int))
    idx    = [TREE_FEATURES.index(f) for f in avail]
    X_test = np.column_stack([
        np.asarray(form_test, dtype=float)[:, idx],
        np.asarray(elo_test, dtype=float),
    ])
    return _align_classes(clf.predict_proba(X_test), clf.classes_, n=len(X_test))

# ---------------------------------------------------------------------------
# Per-match prediction sheet
# ---------------------------------------------------------------------------

SHEET_COLUMNS = [
    "S No", "Team", "Opponent", "",
    "Predicted Win", "Predicted Draw", "Predicted Loss",
    "Result", "Accurate", "Accurate (W/L)", "",
    "Predicted Win (v2)", "Predicted Draw (v2)", "Predicted Loss (v2)",
    "Accurate (v2)", "Accurate W/L (v2)", "",
    "Elo Higher Team", "Elo Accurate",
]


def _write_predictions_sheet(path, test_matches, outcomes,
                             probs_v1, probs_v2, elo_team, elo_opp):
    """
    Write one row per fixture in the exact "Live Test 2026 WC/Predictions and Results.csv" layout —
    the same 19 columns in the same order, including the three unnamed spacers.

    Two model blocks, matching the sheet's v1 / v2 structure:
        base   = GlobalGHMM        (raw logistic-head output)
        (v2)   = GlobalGHMM+Draw   (after the draw-propensity blend)

    The Elo columns are NOT the fitted logistic baseline from the metrics
    tables. They are the plain rule the sheet uses: whichever team carries the
    higher rating into the match is predicted to win, scored 1 if that team won
    and 0 if it lost. Drawn fixtures are left blank here and in both
    `Accurate (W/L)` columns, exactly as the sheet leaves them — the rule cannot
    express a draw, so there is no prediction to mark right or wrong.

    Written through `csv` rather than pandas: the three spacer columns share the
    same empty name, and a DataFrame would rename them Unnamed: 3 / 10 / 16.
    """
    import csv as _csv

    path.parent.mkdir(parents=True, exist_ok=True)
    teams = test_matches["team"].to_numpy(str)
    opps  = test_matches["opponent"].to_numpy(str)

    def _block(probs):
        """(win%, draw%, loss%, accurate, accurate_wl) for one model."""
        acc3   = (np.argmax(probs, axis=1) == outcomes).astype(int)
        acc_wl = []
        for i, o in enumerate(outcomes):
            if o == 1:
                acc_wl.append("")            # sheet leaves drawn fixtures blank
            else:
                pred = 2 if probs[i, 2] > probs[i, 0] else 0
                acc_wl.append(int(pred == o))
        return (np.round(probs[:, 2] * 100, 1),
                np.round(probs[:, 1] * 100, 1),
                np.round(probs[:, 0] * 100, 1),
                acc3, acc_wl)

    w1, d1, l1, a1, awl1 = _block(probs_v1)
    w2, d2, l2, a2, awl2 = _block(probs_v2)

    higher = np.where(elo_team >= elo_opp, teams, opps)

    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = _csv.writer(fh)
        writer.writerow(SHEET_COLUMNS)
        for i, o in enumerate(outcomes):
            result = "Draw" if o == 1 else (teams[i] if o == 2 else opps[i])
            if o == 1:
                elo_acc = ""
            else:
                winner  = teams[i] if o == 2 else opps[i]
                elo_acc = int(higher[i] == winner)
            writer.writerow([
                i + 1, teams[i], opps[i], "",
                w1[i], d1[i], l1[i], result, a1[i], awl1[i], "",
                w2[i], d2[i], l2[i], a2[i], awl2[i], "",
                higher[i], elo_acc,
            ])
    print(f"  Per-match sheet -> {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    out_dir = ARTIFACTS_DIR / "gaussian"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data …")
    full_df = load_matches()

    all_results = {}

    for run in EVAL_RUNS:
        tag, cutoff, label = run["tag"], run["train_cutoff"], run["label"]
        is_tourn = run.get("is_tournament", False)

        print(f"\n{'=' * 60}")
        print(f"  {label}  (train < {cutoff})")
        print(f"{'=' * 60}")

        train_df = full_df[full_df["date"] < cutoff].copy()
        test_matches = (
            _unique_matches(run["test_filter"](full_df))
            .dropna(subset=["outcome", "elo_diff"])
            .reset_index(drop=True)
        )

        save_artifacts = run.get("save_artifacts", False)

        if len(test_matches) == 0:
            if save_artifacts:
                print(f"  Train: {len(train_df)}  |  No test matches (production run)")
                print("  Running Global Gaussian HMM (artifact save only) …")
                _run_global_hmm(train_df, test_matches, is_tourn, save_artifacts=True)
                print("  Production artifacts saved. Skipping evaluation.")
            else:
                print("  No test matches — skipping.")
            continue

        print(f"  Train: {len(train_df)}  |  Test: {len(test_matches)}")
        outcomes = test_matches["outcome"].to_numpy(int)

        print("  Running Global Gaussian HMM …")
        by_mode = _run_global_hmm(train_df, test_matches, is_tourn,
                                  save_artifacts=save_artifacts)
        primary    = by_mode[PRIMARY_ELO_MODE]
        ghmm_raw   = primary["probs_raw"]
        ghmm_blend = primary["probs_blend"]

        print("  Running Elo …")
        elo_probs = _run_elo(train_df, primary["elo_diff"])

        print("  Running RF …")
        rf_probs  = _run_tree(train_df, primary["form_team"], "rf", primary["elo_diff"])

        print("  Running XGBoost …")
        xgb_probs = _run_tree(train_df, primary["form_team"], "xgb", primary["elo_diff"])

        uniform = np.full((len(test_matches), 3), 1.0 / 3.0)

        results = {
            "GlobalGHMM":       _metrics(ghmm_raw,   outcomes),
            "GlobalGHMM+Draw":  _metrics(ghmm_blend, outcomes),   # NEW
            "XGBoost":          _metrics(xgb_probs,  outcomes),
            "RF":               _metrics(rf_probs,   outcomes),
            "Elo":              _metrics(elo_probs,  outcomes),
            "Uniform":          _metrics(uniform,    outcomes),
        }
        results_nodraw = {
            name: _metrics_no_draw(p, outcomes)
            for name, p in [
                ("GlobalGHMM",      ghmm_raw),
                ("GlobalGHMM+Draw", ghmm_blend),
                ("XGBoost",         xgb_probs),
                ("RF",              rf_probs),
                ("Elo",             elo_probs),
                ("Uniform",         uniform),
            ]
        }

        # Confidence-gated metrics for GlobalGHMM+Draw
        conf_metrics = _metrics_at_thresholds(ghmm_blend, outcomes)

        sig = _significance(
            {
                "GlobalGHMM":      ghmm_raw,
                "GlobalGHMM+Draw": ghmm_blend,
                "XGBoost":         xgb_probs,
                "RF":              rf_probs,
                "Elo":             elo_probs,
                "Uniform":         uniform,
            },
            outcomes,
        )

        # Both Elo regimes, so the cost of the production (dynamic) path is
        # recorded rather than inferred.
        elo_mode_metrics = {
            mode: {
                "GlobalGHMM":      _metrics(by_mode[mode]["probs_raw"],   outcomes),
                "GlobalGHMM+Draw": _metrics(by_mode[mode]["probs_blend"], outcomes),
            }
            for mode in ELO_MODES
        }

        all_results[tag] = {
            "label":       label,
            "models":      results,
            "nodraw":      results_nodraw,
            "conf_gated":  conf_metrics,   # NEW
            "significance": sig,           # bootstrap CIs + paired McNemar
            "elo_modes":   elo_mode_metrics,
            "primary_elo_mode": PRIMARY_ELO_MODE,
            "n_test":      int(len(outcomes)),
        }

        # Per-match sheet in the Live Test 2026 WC/Predictions and Results.csv layout.
        _write_predictions_sheet(
            ARTIFACTS_DIR.parent.parent / "results" / f"predictions_{tag}.csv",
            test_matches, outcomes,
            probs_v1=ghmm_raw, probs_v2=ghmm_blend,
            elo_team=primary["elo_team"], elo_opp=primary["elo_opp"],
        )

        n_nd = int((outcomes != 1).sum())
        head3 = (f"  {'Model':<20} | {'Acc':>7} | {'Log-loss':>8} | "
                 f"{'Brier':>6} | {'RPS':>6}")
        sep   = "  " + "-" * (len(head3) - 2)

        print(f"\n  W/D/L — all matches (n={len(outcomes)})")
        print(head3); print(sep)
        for name, m in results.items():
            print(f"  {name:<20} | {m['accuracy']:>7.4f} | {m['log_loss']:>8.4f} "
                  f"| {m['brier']:>6.4f} | {m['rps']:>6.4f}")

        print(f"\n  W/L — decided matches only, draw column ignored (n={n_nd})")
        print(head3); print(sep)
        for name, m in results_nodraw.items():
            if m:
                print(f"  {name:<20} | {m['accuracy']:>7.4f} | {m['log_loss']:>8.4f} "
                      f"| {m['brier']:>6.4f} | {m['rps']:>6.4f}")

        print(f"\n  Headline accuracy with {int(BOOTSTRAP_CI * 100)}% bootstrap CI "
              f"({len(outcomes)} matches, {n_nd} decided, "
              f"{len(outcomes) - n_nd} drawn)")
        print(f"  {'Model':<18} | {'W/D/L':>6} | {'95% CI':>16} "
              f"| {'W/L':>6} | {'95% CI':>16}")
        print("  " + "-" * 76)
        for name, s in sig.items():
            w3, w2 = s["wdl"], s["wl"]
            ci3 = f"[{w3['ci_lo']:.3f}, {w3['ci_hi']:.3f}]"
            ci2 = (f"[{w2['ci_lo']:.3f}, {w2['ci_hi']:.3f}]"
                   if w2["accuracy"] is not None else "—")
            a2  = f"{w2['accuracy']:>6.4f}" if w2["accuracy"] is not None else f"{'—':>6}"
            print(f"  {name:<18} | {w3['accuracy']:>6.4f} | {ci3:>16} "
                  f"| {a2} | {ci2:>16}")

        print(f"\n  Paired McNemar vs {SIG_REFERENCE} "
              f"(only discordant matches carry information)")
        print(f"  {'Model':<18} | {'ref only':>8} | {'other only':>10} "
              f"| {'p (W/D/L)':>9} | {'p (W/L)':>8}")
        print("  " + "-" * 68)
        for name, s in sig.items():
            vr = s.get("vs_reference")
            if not vr:
                continue
            m3, m2 = vr["wdl"], vr["wl"]
            print(f"  {name:<18} | {m3['ref_only_correct']:>8} "
                  f"| {m3['other_only_correct']:>10} | {m3['p_value']:>9.4f} "
                  f"| {m2['p_value']:>8.4f}")

        print(f"\n  Elo regime at test time  (primary = {PRIMARY_ELO_MODE})")
        print(f"  {'Regime':<12} | {'Log-loss':>8} | {'Brier':>6} | {'Acc':>6} | {'RPS':>6}")
        print("  " + "-" * 52)
        for mode in ELO_MODES:
            m = elo_mode_metrics[mode]["GlobalGHMM+Draw"]
            print(f"  {mode:<12} | {m['log_loss']:>8.4f} | {m['brier']:>6.4f} "
                  f"| {m['accuracy']:>6.4f} | {m['rps']:>6.4f}")

        print(f"\n  Confidence-gated accuracy (GlobalGHMM+Draw)")
        print(f"  {'Threshold':>10} | {'N':>5} | {'Coverage':>8} | {'Accuracy':>8}")
        print("  " + "-" * 42)
        for thresh_key, cm in conf_metrics.items():
            t_val = thresh_key.replace("thresh_", "") + "%"
            if cm["accuracy"] is not None:
                print(f"  {t_val:>10} | {cm['n']:>5} | {cm['coverage']:>8.2%} "
                      f"| {cm['accuracy']:>8.4f}")
            else:
                print(f"  {t_val:>10} | {'0':>5} | {'0.00%':>8} | {'N/A':>8}")

    # Stamp the run so every downstream table can be traced to the code and
    # settings that produced it. scripts/make_results.py reads this block.
    all_results["_meta"] = {
        "window":           WINDOW,
        "n_states":         N_STATES,
        "random_seed":      RANDOM_SEED,
        "primary_elo_mode": PRIMARY_ELO_MODE,
        "elo_modes":        list(ELO_MODES),
        "reproducible":     os.environ.get("REPRODUCIBLE", "1") != "0",
        "n_bootstrap":      N_BOOTSTRAP,
        "bootstrap_ci":     BOOTSTRAP_CI,
        "sig_reference":    SIG_REFERENCE,
        "commit":           _git_commit(),
        "generated_utc":    datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    out_json = out_dir / "metrics_global_ghmm.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll metrics written to: {out_json}")
    print("Build the results CSVs with:  python scripts/make_results.py")


if __name__ == "__main__":
    main()