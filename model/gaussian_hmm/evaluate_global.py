"""
evaluate_global.py — Benchmark the global Gaussian HMM against baselines.

Architecture:
  - ONE GlobalGaussianHMM fitted on all training matches (learns match regimes)
    - Competitive matches are up-weighted during HMM fitting; friendlies down-weighted
  - Per-match posterior summary computed via forward algorithm (last 10 matches):
      [p, max_p, entropy]  (N_STATES + 2 features per team)
  - LogisticRegression head:
      outer(p_A, p_B) + elo_diff + confidence/entropy/momentum terms → P(W/D/L)
  - Dynamic updating: completed results added to history before next prediction

Run:
    python -m model.gaussian_hmm.evaluate_global
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
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
from model.gaussian_hmm.predictor_global import GlobalPredictor, WINDOW

WINDOW = 2  # override: last N matches for state inference (10 ≈ one competitive cycle)

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*transmat_.*")

RANDOM_SEED = 42

EVAL_RUNS = [
    {
        "tag":          "wc_2018",
        "train_cutoff": "2018-06-13",
        "test_filter":  lambda df: df[
            (df["date"] >= "2018-06-14") & (df["date"] <= "2018-07-15")
        ],
        "label": "2018 World Cup",
    },
    {
        "tag":          "wc_2022",
        "train_cutoff": "2022-11-19",
        "test_filter":  lambda df: df[
            (df["date"] >= "2022-11-20") & (df["date"] <= "2022-12-18")
        ],
        "label": "2022 World Cup",
    },
    {
        "tag":          "all_2024",
        "train_cutoff": "2024-01-01",
        "test_filter":  lambda df: df[
            (df["date"] >= "2024-01-01") & (df["date"] < "2025-01-01") 
        ],
        "label": "All 2024 Internationals",
    },
]

TREE_FEATURES = [
    "elo_diff",
    "tournament_weight",
    "neutral",
    "days_since_last_match",
    "rolling_win_rate_5",
    "rolling_goal_diff_5",
    "ewa_win_rate",
    "ewa_goal_diff",
    "rolling_win_vs_strong_5",
    "opp_elo_strength_5",
    "rolling_goal_diff_std_5",
    "rolling_win_rate_std_5",
    "ewa_win_rate_momentum",
    "ewa_goal_diff_momentum",
]

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _metrics(probs, outcomes):
    eps = 1e-12
    n   = len(outcomes)
    p_true   = probs[np.arange(n), outcomes]
    log_loss = float(-np.mean(np.log(np.clip(p_true, eps, 1.0))))
    one_hot  = np.zeros_like(probs)
    one_hot[np.arange(n), outcomes] = 1.0
    brier    = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))
    accuracy = float(np.mean(np.argmax(probs, axis=1) == outcomes))
    cum_p    = np.cumsum(probs,   axis=1)
    cum_a    = np.cumsum(one_hot, axis=1)
    rps      = float(np.mean(np.sum((cum_p - cum_a) ** 2, axis=1) / (probs.shape[1] - 1)))
    return {"n": int(n), "log_loss": round(log_loss,4), "brier": round(brier,4),
            "accuracy": round(accuracy,4), "rps": round(rps,4)}


def _metrics_no_draw(probs, outcomes):
    mask = outcomes != 1
    return _metrics(probs[mask], outcomes[mask]) if mask.sum() > 0 else {}


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
    hmm: GlobalGaussianHMM,
    pf_team: np.ndarray,
    pf_opp:  np.ndarray,
    elo_diff: float,
) -> np.ndarray:
    """
    Construct the full feature vector passed to the logistic head.

    Each posterior summary pf_* has length N + 2:
        p        (N)  — predictive state distribution
        max_p    (1)  — posterior confidence (peak probability)
        entropy  (1)  — posterior uncertainty (Shannon entropy, nats)

    Feature vector layout:
        outer(p_A, p_B).ravel()     (N²)  joint regime interaction
        max_p_A, max_p_B            (2)   HMM confidence per team
        entropy_A, entropy_B        (2)   HMM uncertainty per team
        elo_diff                    (1)   rating difference
        elo_diff * max_p_A          (1)   strength × regime confidence (team)
        elo_diff * max_p_B          (1)   strength × regime confidence (opponent)

    Total: N² + 7  (for N=7: 49 + 7 = 56 features)

    The interaction terms elo_diff * max_p_* are the key addition: they let
    the head distinguish "Elo-strong AND confidently in a dominant regime"
    from "Elo-strong but posterior-uncertain", which neither the outer product
    nor elo_diff alone can express.
    """
    N = hmm.n_states

    p_A     = pf_team[:N]
    max_p_A = pf_team[N]
    ent_A   = pf_team[N + 1]

    p_B     = pf_opp[:N]
    max_p_B = pf_opp[N]
    ent_B   = pf_opp[N + 1]

    outer = np.outer(p_A, p_B).ravel()

    return np.concatenate([
        outer,                                 # (N²,)
        [max_p_A, max_p_B],                    # (2,)
        [ent_A,   ent_B],                      # (2,)
        [elo_diff],                            # (1,)
        [elo_diff * max_p_A],                  # (1,)  interaction
        [elo_diff * max_p_B],                  # (1,)  interaction
    ])


# ---------------------------------------------------------------------------
# Build logistic head training data from train_df
# ---------------------------------------------------------------------------

def _build_head_features(
    train_df: pd.DataFrame,
    hmm: GlobalGaussianHMM,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each match in train_df (deduplicated), compute posterior summary
    using only prior matches, then build the full feature vector.
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
            # No history: uniform state dist, max_p=1/N, entropy=log(N)
            N = hmm.n_states
            unif = np.full(N, 1.0 / N)
            return np.concatenate([unif, [1.0 / N, np.log(N)]])
        idx   = np.searchsorted(rec["dates"], np.datetime64(pd.Timestamp(date)), side="left")
        feats = rec["features"][max(0, idx - WINDOW): idx]
        return hmm.posterior_features(feats)

    head_matches = _unique_matches(train_df).dropna(subset=["outcome", "elo_diff"])

    X_list, y_list = [], []
    for _, row in head_matches.iterrows():
        pt = posterior(row["team"],     row["date"])
        po = posterior(row["opponent"], row["date"])
        fv = _build_feature_vec(hmm, pt, po, float(row["elo_diff"]))
        X_list.append(fv)
        y_list.append(int(row["outcome"]))

    return np.array(X_list), np.array(y_list)


# ---------------------------------------------------------------------------
# Global HMM runner
# ---------------------------------------------------------------------------

def _run_global_hmm(train_df, test_matches):
    # 1. Build per-team sequences
    per_team_feats  = {}
    lengths         = []
    all_X           = []
    all_weights     = []

    tournament_col = train_df["tournament"].to_numpy() if "tournament" in train_df.columns else None

    for team, grp in train_df.groupby("team"):
        grp_sorted = grp.sort_values("date")
        feats = grp_sorted[FEATURE_NAMES].fillna(0).to_numpy(float)
        if len(feats) >= 5:
            per_team_feats[team] = feats
            all_X.append(feats)
            lengths.append(len(feats))
            if tournament_col is not None:
                w = _tournament_sample_weight(grp_sorted["tournament"].to_numpy())
            else:
                w = np.ones(len(feats))
            all_weights.append(w)

    X_all = np.vstack(all_X)
    W_all = np.concatenate(all_weights) if all_weights else None

    # 2. Fit global HMM with competitive-match weighting
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

    # 3. Train logistic head on richer posterior features
    print("  Training logistic head on posterior summary features …")
    X_head, y_head = _build_head_features(train_df, hmm)
    X_head = np.nan_to_num(X_head, nan=0.0, posinf=0.0, neginf=0.0)

    head = LogisticRegression(max_iter=2000, C=1.0, random_state=RANDOM_SEED)
    head.fit(X_head, y_head)
    n_feats = X_head.shape[1]
    print(f"  Head trained on {len(y_head)} matches, {n_feats} features "
          f"(N²={N_STATES**2} joint + 2 conf + 2 ent + 1 elo + 2 elo×conf interactions)")

    # 4. Elo ratings
    elo_ratings = (
        train_df.sort_values("date")
        .groupby("team")["team_elo"]
        .last()
        .to_dict()
    )

    # 5. Dynamic prediction — build per-team history lookup
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
            N = hmm.n_states
            unif = np.full(N, 1.0 / N)
            return np.concatenate([unif, [1.0 / N, np.log(N)]])
        idx   = np.searchsorted(rec["dates"], np.datetime64(pd.Timestamp(date)), side="left")
        feats = rec["features"][max(0, idx - WINDOW): idx]
        return hmm.posterior_features(feats)

    def append_to_history(team, new_row_dict):
        """Add a single observed match to the running per-team lookup."""
        rec = per_team_hist.setdefault(team, {"dates": np.array([], dtype="datetime64"),
                                               "features": np.empty((0, len(FEATURE_NAMES)))})
        new_date = np.array([np.datetime64(pd.Timestamp(new_row_dict["date"]))],
                            dtype="datetime64")
        new_feat = np.array([[float(new_row_dict.get(f, 0) or 0) for f in FEATURE_NAMES]])
        rec["dates"]    = np.concatenate([rec["dates"],    new_date])
        rec["features"] = np.vstack(     [rec["features"], new_feat])

    probs = np.zeros((len(test_matches), 3), float)
    for i, (_, row) in enumerate(test_matches.iterrows()):
        pt = posterior_for(row["team"],     row["date"])
        po = posterior_for(row["opponent"], row["date"])
        fv = _build_feature_vec(hmm, pt, po, float(row["elo_diff"]))
        fv = np.nan_to_num(fv, nan=0.0, posinf=0.0, neginf=0.0)

        raw = head.predict_proba(fv.reshape(1, -1))
        aligned = _align_classes(raw, head.classes_, n=1)[0]
        probs[i] = aligned   # [Loss, Draw, Win]

        # Dynamic update: add both perspectives to running history
        row_dict = row.to_dict()
        append_to_history(row["team"],     row_dict)
        opp_dict = {**row_dict,
                    "team":     row["opponent"],
                    "opponent": row["team"],
                    "outcome":  2 - int(row["outcome"])}
        append_to_history(row["opponent"], opp_dict)

    return probs


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def _run_elo(train_df, test_matches):
    train_u = _unique_matches(train_df).dropna(subset=["elo_diff", "outcome"])
    clf = LogisticRegression(max_iter=1000)
    clf.fit(train_u[["elo_diff"]].to_numpy(float), train_u["outcome"].to_numpy(int))
    raw = clf.predict_proba(test_matches[["elo_diff"]].to_numpy(float))
    return _align_classes(raw, clf.classes_, n=len(test_matches))


def _run_tree(train_df, test_matches, model_type):
    avail   = [f for f in TREE_FEATURES if f in train_df.columns]
    train_u = _unique_matches(train_df).dropna(subset=avail + ["outcome"])
    clf = (
        RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=1)
        if model_type == "rf"
        else XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                           use_label_encoder=False, eval_metric="mlogloss",
                           random_state=42, verbosity=0)
    )
    clf.fit(train_u[avail].to_numpy(float), train_u["outcome"].to_numpy(int))
    X_test = test_matches[avail].fillna(1/3).to_numpy(float)
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

        print(f"\n{'=' * 60}")
        print(f"  {label}  (train < {cutoff})")
        print(f"{'=' * 60}")

        train_df = full_df[full_df["date"] < cutoff].copy()
        test_matches = (
            _unique_matches(run["test_filter"](full_df))
            .dropna(subset=["outcome", "elo_diff"])
            .reset_index(drop=True)
        )

        if len(test_matches) == 0:
            print("  No test matches — skipping.")
            continue

        print(f"  Train: {len(train_df)}  |  Test: {len(test_matches)}")
        outcomes = test_matches["outcome"].to_numpy(int)

        print("  Running Global Gaussian HMM …")
        ghmm_probs = _run_global_hmm(train_df, test_matches)

        print("  Running Elo …")
        elo_probs  = _run_elo(train_df, test_matches)

        print("  Running RF …")
        rf_probs   = _run_tree(train_df, test_matches, "rf")

        print("  Running XGBoost …")
        xgb_probs  = _run_tree(train_df, test_matches, "xgb")

        uniform = np.full((len(test_matches), 3), 1.0 / 3.0)

        results = {
            "GlobalGHMM": _metrics(ghmm_probs, outcomes),
            "XGBoost":    _metrics(xgb_probs,  outcomes),
            "RF":         _metrics(rf_probs,    outcomes),
            "Elo":        _metrics(elo_probs,   outcomes),
            "Uniform":    _metrics(uniform,     outcomes),
        }
        results_nodraw = {
            name: _metrics_no_draw(p, outcomes)
            for name, p in [
                ("GlobalGHMM", ghmm_probs), ("XGBoost", xgb_probs),
                ("RF", rf_probs), ("Elo", elo_probs), ("Uniform", uniform),
            ]
        }
        all_results[tag] = {"label": label, "models": results, "nodraw": results_nodraw}

        header = f"  {'Model':<13} | {'Log-loss':>8} | {'Brier':>6} | {'Acc':>6} | {'RPS':>6}"
        print(f"\n  All matches (n={len(outcomes)})")
        print(header)
        print("  " + "-" * (len(header) - 2))
        for name, m in results.items():
            print(f"  {name:<13} | {m['log_loss']:>8.4f} | {m['brier']:>6.4f} "
                  f"| {m['accuracy']:>6.4f} | {m['rps']:>6.4f}")

        n_nd = int((outcomes != 1).sum())
        print(f"\n  W/L only (n={n_nd})")
        print(header)
        print("  " + "-" * (len(header) - 2))
        for name, m in results_nodraw.items():
            if m:
                print(f"  {name:<13} | {m['log_loss']:>8.4f} | {m['brier']:>6.4f} "
                      f"| {m['accuracy']:>6.4f} | {m['rps']:>6.4f}")

    out_json = out_dir / "metrics_global_ghmm.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll metrics written to: {out_json}")


if __name__ == "__main__":
    main()