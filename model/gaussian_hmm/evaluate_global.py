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
import warnings
from pathlib import Path

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
    _elo_update,
    _outcome_to_score,
    _is_knockout,
    _tournament_weight_val,
    _draw_features,
    _blend_draw_probs,
    _train_draw_model,
)

WINDOW = 7  # last N matches for state inference — empirically best

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*transmat_.*")

RANDOM_SEED = 42

EVAL_RUNS = [
    {
        "tag":           "wc_2018",
        "train_cutoff":  "2018-06-13",
        "test_filter":   lambda df: df[
            (df["date"] >= "2018-06-14") & (df["date"] <= "2018-07-15")
        ],
        "label":         "2018 World Cup",
        "is_tournament": True,
        "save_artifacts": False,
    },
    {
        "tag":           "wc_2022",
        "train_cutoff":  "2022-11-19",
        "test_filter":   lambda df: df[
            (df["date"] >= "2022-11-20") & (df["date"] <= "2022-12-18") & (df["tournament"] != "Friendly")
        ],
        "label":         "2022 World Cup",
        "is_tournament": True,
        "save_artifacts": False,
    },
    {
        "tag":           "2024_all_matches",
        "train_cutoff":  "2024-01-01",
        "test_filter":   lambda df: df[
            (df["date"] >= "2024-01-01") & (df["date"] <= "2024-12-31")
        ],
        "label":         "2024 All Matches",
        "is_tournament": False,
        "save_artifacts": False,
    },
    {
        "tag":           "wc2026_prod",
        "train_cutoff":  "2026-06-11",
        "test_filter":   lambda df: df[df["date"] >= "2026-06-11"],
        "label":         "WC 2026 Production Model (all data to kick-off)",
        "is_tournament": True,
        "save_artifacts": True,   # ← only this run saves the artifacts
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
    mask = outcomes != 1
    return _metrics(probs[mask], outcomes[mask]) if mask.sum() > 0 else {}


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


def _unique_matches(df):
    return df[df["team"] < df["opponent"]].sort_values("date").reset_index(drop=True)


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

    # ── 4. Elo ratings (live-updating dict) ──────────────────────────────────
    live_elo = (
        train_df.sort_values("date")
        .groupby("team")["team_elo"]
        .last()
        .to_dict()
    )
    default_elo = float(np.mean(list(live_elo.values()))) if live_elo else 1500.0

    # ── 5. Dynamic per-team feature history ──────────────────────────────────
    sorted_train = train_df.sort_values("date").reset_index(drop=True)
    per_team_hist = {}
    for team, grp in sorted_train.groupby("team", sort=False):
        per_team_hist[team] = {
            "dates":    grp["date"].to_numpy(),
            "features": grp[FEATURE_NAMES].fillna(0).to_numpy(dtype=float),
        }

    def posterior_for(team, date):
        rec = per_team_hist.get(team)
        if rec is None:
            N    = hmm.n_states
            unif = np.full(N, 1.0 / N)
            return np.concatenate([unif, [1.0 / N, np.log(N)]])
        idx   = np.searchsorted(rec["dates"],
                                np.datetime64(pd.Timestamp(date)), side="left")
        feats = rec["features"][max(0, idx - WINDOW): idx]
        return hmm.posterior_features(feats)

    def append_to_history(team, new_row_dict):
        rec = per_team_hist.setdefault(
            team,
            {"dates": np.array([], dtype="datetime64"),
             "features": np.empty((0, len(FEATURE_NAMES)))}
        )
        new_date = np.array([np.datetime64(pd.Timestamp(new_row_dict["date"]))],
                            dtype="datetime64")
        new_feat = np.array([[float(new_row_dict.get(f, 0) or 0)
                              for f in FEATURE_NAMES]])
        rec["dates"]    = np.concatenate([rec["dates"],    new_date])
        rec["features"] = np.vstack(     [rec["features"], new_feat])

    # ── 6. Predict with dynamic Elo + draw blending ──────────────────────────
    probs_raw   = np.zeros((len(test_matches), 3), float)
    probs_blend = np.zeros((len(test_matches), 3), float)
    elo_diffs_test = np.zeros(len(test_matches))
    ent_a_test     = np.zeros(len(test_matches))
    ent_b_test     = np.zeros(len(test_matches))
    is_ko_test     = np.zeros(len(test_matches), dtype=int)

    for i, (_, row) in enumerate(test_matches.iterrows()):
        team = row["team"];  opp = row["opponent"]

        # Dynamic Elo diff (updated after each match)
        r_team = live_elo.get(team, default_elo)
        r_opp  = live_elo.get(opp,  default_elo)
        dyn_elo_diff = r_team - r_opp

        is_ko = _is_knockout(row.get("tournament", ""))
        tw    = _tournament_weight_val(row.get("tournament", ""))

        pt = posterior_for(team, row["date"])
        po = posterior_for(opp,  row["date"])

        fv  = _build_feature_vec(hmm, pt, po, dyn_elo_diff, is_ko, tw)
        fv  = np.nan_to_num(fv, nan=0.0, posinf=0.0, neginf=0.0)
        raw = head.predict_proba(fv.reshape(1, -1))
        aligned = _align_classes(raw, head.classes_, n=1)[0]
        probs_raw[i] = aligned

        elo_diffs_test[i] = dyn_elo_diff
        ent_a_test[i]     = float(pt[hmm.n_states + 1])
        ent_b_test[i]     = float(po[hmm.n_states + 1])
        is_ko_test[i]     = is_ko

        # Dynamic update: add result to both teams' histories
        row_dict = row.to_dict()
        append_to_history(team, row_dict)
        opp_dict = {**row_dict,
                    "team":     opp,
                    "opponent": team,
                    "outcome":  2 - int(row["outcome"])}
        append_to_history(opp, opp_dict)

        # Dynamic Elo update AFTER prediction, BEFORE next match
        score_a = _outcome_to_score(int(row["outcome"]))
        new_r_team, new_r_opp = _elo_update(r_team, r_opp, score_a)
        live_elo[team] = new_r_team
        live_elo[opp]  = new_r_opp

    # ── 7. Draw propensity blending ──────────────────────────────────────────
    if len(test_matches) == 0:
        empty = np.zeros((0, 3), float)
        return empty, empty, np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0, int)

    X_draw_test  = _draw_features(probs_raw, elo_diffs_test,
                                  ent_a_test, ent_b_test, is_ko_test)
    draw_probs   = draw_model.predict_proba(X_draw_test)[:, 1]
    probs_blend  = _blend_draw_probs(probs_raw, draw_probs, alpha=0.3)

    return probs_raw, probs_blend, elo_diffs_test, ent_a_test, ent_b_test, is_ko_test

# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def _run_elo(train_df, test_matches):
    train_u = _unique_matches(train_df).dropna(subset=["elo_diff", "outcome"])
    clf     = LogisticRegression(max_iter=1000)
    clf.fit(train_u[["elo_diff"]].to_numpy(float), train_u["outcome"].to_numpy(int))
    raw = clf.predict_proba(test_matches[["elo_diff"]].to_numpy(float))
    return _align_classes(raw, clf.classes_, n=len(test_matches))


def _run_tree(train_df, test_matches, model_type):
    avail   = [f for f in TREE_FEATURES if f in train_df.columns]
    # Add is_knockout if not present
    if "is_knockout" not in test_matches.columns:
        test_matches = test_matches.copy()
        test_matches["is_knockout"] = test_matches.get(
            "tournament", pd.Series([""] * len(test_matches))
        ).apply(_is_knockout)
    train_u = _unique_matches(train_df).dropna(subset=avail + ["outcome"])
    if "is_knockout" not in train_u.columns:
        train_u = train_u.copy()
        train_u["is_knockout"] = train_u.get(
            "tournament", pd.Series([""] * len(train_u))
        ).apply(_is_knockout)
    avail = [f for f in avail if f in train_u.columns and f in test_matches.columns]
    clf = (
        RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=1)
        if model_type == "rf"
        else XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                           use_label_encoder=False, eval_metric="mlogloss",
                           random_state=42, verbosity=0)
    )
    clf.fit(train_u[avail].to_numpy(float), train_u["outcome"].to_numpy(int))
    X_test = test_matches[avail].fillna(1 / 3).to_numpy(float)
    return _align_classes(clf.predict_proba(X_test), clf.classes_, n=len(test_matches))

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
        ghmm_raw, ghmm_blend, elo_d, ent_a, ent_b, is_ko = \
            _run_global_hmm(train_df, test_matches, is_tourn, save_artifacts=save_artifacts)

        print("  Running Elo …")
        elo_probs = _run_elo(train_df, test_matches)

        print("  Running RF …")
        rf_probs  = _run_tree(train_df, test_matches, "rf")

        print("  Running XGBoost …")
        xgb_probs = _run_tree(train_df, test_matches, "xgb")

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

        all_results[tag] = {
            "label":       label,
            "models":      results,
            "nodraw":      results_nodraw,
            "conf_gated":  conf_metrics,   # NEW
        }

        header = f"  {'Model':<20} | {'Log-loss':>8} | {'Brier':>6} | {'Acc':>6} | {'RPS':>6}"
        sep    = "  " + "-" * (len(header) - 2)

        print(f"\n  All matches (n={len(outcomes)})")
        print(header); print(sep)
        for name, m in results.items():
            print(f"  {name:<20} | {m['log_loss']:>8.4f} | {m['brier']:>6.4f} "
                  f"| {m['accuracy']:>6.4f} | {m['rps']:>6.4f}")

        n_nd = int((outcomes != 1).sum())
        print(f"\n  W/L only (n={n_nd})")
        print(header); print(sep)
        for name, m in results_nodraw.items():
            if m:
                print(f"  {name:<20} | {m['log_loss']:>8.4f} | {m['brier']:>6.4f} "
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

    out_json = out_dir / "metrics_global_ghmm.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll metrics written to: {out_json}")


if __name__ == "__main__":
    main()