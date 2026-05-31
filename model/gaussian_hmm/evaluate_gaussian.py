"""
evaluate_gaussian.py — Benchmark the Gaussian HMM (3-state) against baselines.

Run:
    python -m model.gaussian_hmm.evaluate_gaussian
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from model.config import ARTIFACTS_DIR
from model.data_loader import load_matches
from model.gaussian_hmm.hmm_team_gaussian import TeamGaussianHMM, FEATURE_NAMES
from model.gaussian_hmm.joint_emission_gaussian import build_joint_tensor_gaussian
from model.gaussian_hmm.predictor_gaussian import GaussianPredictor

warnings.filterwarnings("ignore")

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
]

# 3-state Gaussian HMM: 3*7*2 (means+vars) + 3*3 (trans) + 3 (start) = 54 params
# 40 matches is a safe minimum for diagonal covariance
MIN_MATCHES_GAUSSIAN = 20
N_STATES_GAUSSIAN    = 3


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _metrics(probs: np.ndarray, outcomes: np.ndarray) -> dict:
    eps = 1e-12
    n   = len(outcomes)
    p_true   = probs[np.arange(n), outcomes]
    log_loss = float(-np.mean(np.log(np.clip(p_true, eps, 1.0))))
    one_hot  = np.zeros_like(probs)
    one_hot[np.arange(n), outcomes] = 1.0
    brier    = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))
    accuracy = float(np.mean(np.argmax(probs, axis=1) == outcomes))
    cum_probs  = np.cumsum(probs,   axis=1)
    cum_actual = np.cumsum(one_hot, axis=1)
    rps = float(np.mean(
        np.sum((cum_probs - cum_actual) ** 2, axis=1) / (probs.shape[1] - 1)
    ))
    return {
        "n":        int(n),
        "log_loss": round(log_loss, 4),
        "brier":    round(brier,    4),
        "accuracy": round(accuracy, 4),
        "rps":      round(rps,      4),
    }


def _metrics_no_draw(probs, outcomes):
    mask = outcomes != 1
    return _metrics(probs[mask], outcomes[mask]) if mask.sum() > 0 else {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rebuild_per_team(df: pd.DataFrame) -> dict:
    per_team = {}
    for team, grp in df.sort_values("date").groupby("team", sort=False):
        per_team[team] = {
            "dates":    grp["date"].to_numpy(),
            "features": grp[FEATURE_NAMES].fillna(0).to_numpy(dtype=float),
        }
    return per_team


def _unique_matches(df):
    return df[df["team"] < df["opponent"]].sort_values("date").reset_index(drop=True)


def _align_classes(raw, classes, n):
    aligned = np.zeros((n, 3), float)
    for k, cls in enumerate(classes):
        aligned[:, int(cls)] = raw[:, k]
    return aligned


# ---------------------------------------------------------------------------
# Gaussian HMM runner
# ---------------------------------------------------------------------------

def _run_gaussian_hmm(train_df, test_matches):
    import model.gaussian_hmm.hmm_team_gaussian as _m; print("scaler" in dir(_m.TeamGaussianHMM()))
    team_seqs = {}
    for team, grp in train_df.groupby("team"):
        feat = grp.sort_values("date")[FEATURE_NAMES].fillna(0).to_numpy(float)
        if len(feat) >= MIN_MATCHES_GAUSSIAN:
            team_seqs[team] = feat

    team_hmms = {}
    failed = 0
    for team, feat in team_seqs.items():
        try:
            team_hmms[team] = TeamGaussianHMM(n_states=N_STATES_GAUSSIAN).fit(feat)
        except Exception as e:
            failed += 1
            if failed <= 3:
                print(f"  ERROR fitting {team}: {type(e).__name__}: {e}")

    print(f"  Fitted {len(team_hmms)} Gaussian HMMs, {failed} failed, "
          f"{len(team_seqs) - len(team_hmms) - failed} skipped (< {MIN_MATCHES_GAUSSIAN} matches)")

    if not team_hmms:
        print("  WARNING: no Gaussian HMMs fitted — returning uniform")
        return np.full((len(test_matches), 3), 1.0 / 3.0)

    joint_tensor, diag = build_joint_tensor_gaussian(train_df, team_hmms, smoothing=2.0)
    print(f"  Tensor: {diag['matches_used']} matches used, {diag['matches_skipped']} skipped")

    elo_ratings = (
        train_df.sort_values("date")
        .groupby("team")["team_elo"]
        .last()
        .to_dict()
    )

    running_history = train_df.copy()
    predictor = GaussianPredictor(
        team_hmms=team_hmms,
        joint_tensor=joint_tensor,
        history_df=running_history,
        elo_ratings=elo_ratings,
    )

    probs = np.zeros((len(test_matches), 3), float)
    for i, (_, row) in enumerate(test_matches.iterrows()):
        r = predictor.predict(row["team"], row["opponent"], row["date"])
        probs[i] = [r["Loss"], r["Draw"], r["Win"]]

        new_rows = pd.DataFrame([
            row.to_dict(),
            {**row.to_dict(), "team": row["opponent"], "opponent": row["team"],
             "outcome": 2 - int(row["outcome"])},
        ])
        running_history = pd.concat(
            [running_history, new_rows], ignore_index=True
        ).sort_values("date").reset_index(drop=True)
        predictor._per_team = _rebuild_per_team(running_history)

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
    available = [f for f in TREE_FEATURES if f in train_df.columns]
    train_u   = _unique_matches(train_df).dropna(subset=available + ["outcome"])
    clf = (
        RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        if model_type == "rf"
        else XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                           use_label_encoder=False, eval_metric="mlogloss",
                           random_state=42, verbosity=0)
    )
    clf.fit(train_u[available].to_numpy(float), train_u["outcome"].to_numpy(int))
    X_test = test_matches[available].fillna(1 / 3).to_numpy(float)
    return _align_classes(clf.predict_proba(X_test), clf.classes_, n=len(test_matches))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
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
            print("  No test matches found — skipping.")
            continue

        print(f"  Train: {len(train_df)}  |  Test: {len(test_matches)}")
        outcomes = test_matches["outcome"].to_numpy(int)

        print("  Running Gaussian HMM …")
        ghmm_probs = _run_gaussian_hmm(train_df, test_matches)

        print("  Running Elo baseline …")
        elo_probs  = _run_elo(train_df, test_matches)

        print("  Running Random Forest …")
        rf_probs   = _run_tree(train_df, test_matches, "rf")

        print("  Running XGBoost …")
        xgb_probs  = _run_tree(train_df, test_matches, "xgb")

        uniform = np.full((len(test_matches), 3), 1.0 / 3.0)

        results = {
            "GaussianHMM": _metrics(ghmm_probs, outcomes),
            "XGBoost":     _metrics(xgb_probs,  outcomes),
            "RF":          _metrics(rf_probs,    outcomes),
            "Elo":         _metrics(elo_probs,   outcomes),
            "Uniform":     _metrics(uniform,     outcomes),
        }
        results_nodraw = {
            name: _metrics_no_draw(p, outcomes)
            for name, p in [
                ("GaussianHMM", ghmm_probs), ("XGBoost", xgb_probs),
                ("RF", rf_probs), ("Elo", elo_probs), ("Uniform", uniform),
            ]
        }
        all_results[tag] = {"label": label, "models": results, "nodraw": results_nodraw}

        header = f"  {'Model':<14} | {'Log-loss':>8} | {'Brier':>6} | {'Acc':>6} | {'RPS':>6}"
        print(f"\n  All matches (n={len(outcomes)})")
        print(header)
        print("  " + "-" * (len(header) - 2))
        for name, m in results.items():
            print(f"  {name:<14} | {m['log_loss']:>8.4f} | {m['brier']:>6.4f} "
                  f"| {m['accuracy']:>6.4f} | {m['rps']:>6.4f}")

        n_nodraw = int((outcomes != 1).sum())
        print(f"\n  W/L only (draws excluded, n={n_nodraw})")
        print(header)
        print("  " + "-" * (len(header) - 2))
        for name, m in results_nodraw.items():
            if m:
                print(f"  {name:<14} | {m['log_loss']:>8.4f} | {m['brier']:>6.4f} "
                      f"| {m['accuracy']:>6.4f} | {m['rps']:>6.4f}")

    out_json = out_dir / "metrics_gaussian.json"
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(all_results, fh, indent=2)
    print(f"\nAll metrics written to: {out_json}")


if __name__ == "__main__":
    main()