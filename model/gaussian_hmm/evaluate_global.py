"""
evaluate_global.py — Benchmark the global Gaussian HMM against baselines.

Architecture:
  - ONE GlobalGaussianHMM fitted on all training matches (learns match regimes)
  - Per-match state distributions computed via forward algorithm (last 10 matches)
  - LogisticRegression head: outer(p_A, p_B) + elo features → P(W/D/L)
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
from model.gaussian_hmm.hmm_global import GlobalGaussianHMM, FEATURE_NAMES, N_STATES
from model.gaussian_hmm.predictor_global import GlobalPredictor, WINDOW

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
    "rolling_win_rate_5",
    "rolling_goal_diff_5",
    "tournament_weight",
    "neutral",
    "ewa_win_rate",
    "ewa_goal_diff",
    "opp_elo_strength_5",
    "rolling_win_vs_strong_5",
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
# Build logistic head training data from train_df
# ---------------------------------------------------------------------------

def _build_head_features(train_df: pd.DataFrame,
                         hmm: GlobalGaussianHMM) -> tuple[np.ndarray, np.ndarray]:
    """
    For each match in train_df (deduplicated), compute state distributions
    using only prior matches, then build outer-product feature vectors.
    Uses a 70/30 split: HMM trained on first 70%, head trained on last 30%
    to avoid the HMM memorising its own training data.
    """
    # Build per-team lookup
    sorted_df = train_df.sort_values("date").reset_index(drop=True)
    per_team  = {}
    for team, grp in sorted_df.groupby("team", sort=False):
        per_team[team] = {
            "dates":    grp["date"].to_numpy(),
            "features": grp[FEATURE_NAMES].fillna(0).to_numpy(dtype=float),
        }

    def state_dist(team, date):
        rec = per_team.get(team)
        if rec is None:
            return np.full(hmm.n_states, 1.0 / hmm.n_states)
        idx   = np.searchsorted(rec["dates"], np.datetime64(pd.Timestamp(date)), side="left")
        feats = rec["features"][max(0, idx - WINDOW): idx]
        return hmm.forward_state_dist(feats)

    unique = _unique_matches(train_df).dropna(subset=["outcome", "elo_diff"])
    # Use last 40% of matches for head training (chronological)
    n_head = max(int(len(unique) * 0.4), 50)
    head_matches = unique.iloc[-n_head:]

    X_list, y_list = [], []
    for _, row in head_matches.iterrows():
        p_t = state_dist(row["team"],     row["date"])
        p_o = state_dist(row["opponent"], row["date"])
        elo_diff = float(row["elo_diff"])
        outer    = np.outer(p_t, p_o).ravel()
        fv       = np.concatenate([outer, [elo_diff, elo_diff**2, abs(elo_diff)]])
        X_list.append(fv)
        y_list.append(int(row["outcome"]))

    return np.array(X_list), np.array(y_list)


# ---------------------------------------------------------------------------
# Global HMM runner
# ---------------------------------------------------------------------------

def _run_global_hmm(train_df, test_matches):
    # 1. Build sequences: one per team, chronological
    per_team_feats = {}
    lengths        = []
    all_X          = []

    for team, grp in train_df.groupby("team"):
        feats = grp.sort_values("date")[FEATURE_NAMES].fillna(0).to_numpy(float)
        if len(feats) >= 5:
            per_team_feats[team] = feats
            all_X.append(feats)
            lengths.append(len(feats))

    X_all = np.vstack(all_X)

    # 2. Fit global HMM
    print(f"  Fitting global HMM on {X_all.shape[0]} observations, "
          f"{len(lengths)} team sequences …")
    hmm = GlobalGaussianHMM(n_states=N_STATES)
    hmm.fit(X_all, lengths=lengths)

    # 3. Train logistic head on last 40% of training matches
    print("  Training logistic head …")
    X_head, y_head = _build_head_features(train_df, hmm)
    # Fill NaNs from new rolling features before fitting head
    X_head = np.nan_to_num(X_head, nan=0.0)

    head = LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_SEED)
    head.fit(X_head, y_head)
    print(f"  Head trained on {len(y_head)} matches")

    # 4. Elo ratings
    elo_ratings = (
        train_df.sort_values("date")
        .groupby("team")["team_elo"]
        .last()
        .to_dict()
    )

    # 5. Dynamic prediction
    running_history = train_df.copy()
    predictor = GlobalPredictor(
        global_hmm=hmm,
        head=head,
        history_df=running_history,
    )

    probs = np.zeros((len(test_matches), 3), float)
    for i, (_, row) in enumerate(test_matches.iterrows()):
        r = predictor.predict(
            row["team"], row["opponent"], row["date"],
            elo_ratings=elo_ratings
        )
        probs[i] = [r["Loss"], r["Draw"], r["Win"]]

        # Append result to running history
        new_rows = pd.DataFrame([
            row.to_dict(),
            {**row.to_dict(), "team": row["opponent"], "opponent": row["team"],
             "outcome": 2 - int(row["outcome"])},
        ])
        running_history = pd.concat(
            [running_history, new_rows], ignore_index=True
        ).sort_values("date").reset_index(drop=True)
        predictor.update_history(running_history)

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
        RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
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
    print(f"\nAll metrics → {out_json}")


if __name__ == "__main__":
    main()