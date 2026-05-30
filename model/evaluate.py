"""
evaluate.py — Evaluate the trained HMM predictor against RF and XGBoost baselines.

Runs three evaluation rounds:
  1. Train < 2018-01-01, Test = 2018 WC matches
  2. Train < 2022-01-01, Test = 2022 WC matches
  3. Train < 2024-01-01, Test = all 2024 matches

For each round, scores all models on:
  - Multiclass log loss   (lower is better; uniform baseline = log(3) ≈ 1.0986)
  - Brier score           (lower is better)
  - Top-1 accuracy        (higher is better)
  - RPS                   (lower is better; standard football forecasting metric)

Also writes per-round calibration plots to artifacts/calibration_<tag>.png
and a combined metrics summary to artifacts/metrics_all.json.

Run:
    python -m model.evaluate
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

from model.config import ARTIFACTS_DIR, OUTCOME_LABELS
from model.data_loader import load_matches
from model.hmm_team import TeamHMM
from model.joint_emission import build_joint_tensor
from model.predictor import Predictor

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Evaluation configs
# ---------------------------------------------------------------------------

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

# Features already present in filtered_matches.csv — no extra engineering needed
TREE_FEATURES = [
    "elo_diff",
    "rolling_win_rate_5",
    "rolling_goal_diff_5",
    "tournament_weight",
]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _metrics(probs: np.ndarray, outcomes: np.ndarray) -> dict:
    """
    probs   : (N, 3) float  — columns are [Loss, Draw, Win]
    outcomes: (N,)   int    — values in {0, 1, 2}
    """
    eps = 1e-12
    n = len(outcomes)

    # Log loss
    p_true   = probs[np.arange(n), outcomes]
    log_loss = float(-np.mean(np.log(np.clip(p_true, eps, 1.0))))

    # Brier score
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(n), outcomes] = 1.0
    brier = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))

    # Accuracy
    accuracy = float(np.mean(np.argmax(probs, axis=1) == outcomes))

    # RPS — Ranked Probability Score (standard football forecasting metric)
    cum_probs  = np.cumsum(probs, axis=1)
    cum_actual = np.cumsum(one_hot, axis=1)
    rps = float(np.mean(
        np.sum((cum_probs - cum_actual) ** 2, axis=1) / (probs.shape[1] - 1)
    ))

    return {
        "n":        int(n),
        "log_loss": round(log_loss, 4),
        "brier":    round(brier, 4),
        "accuracy": round(accuracy, 4),
        "rps":      round(rps, 4),
    }


# ---------------------------------------------------------------------------
# HMM — train from scratch on train_df, predict on test_matches
# ---------------------------------------------------------------------------

def _run_hmm(
    train_df: pd.DataFrame,
    test_matches: pd.DataFrame,
    full_df: pd.DataFrame,
) -> np.ndarray:
    """
    Trains TeamHMMs + joint tensor on train_df, then predicts each test match.
    full_df is passed to Predictor so date-gating can look up pre-match history.
    """
    # Build per-team sequences from training data only
    team_seqs: dict[str, np.ndarray] = {}
    for team, grp in train_df.groupby("team"):
        seq = grp.sort_values("date")["outcome"].to_numpy(int)
        if len(seq) >= 20:
            team_seqs[team] = seq

    # Fit one HMM per team
    team_hmms: dict[str, TeamHMM] = {}
    for team, seq in team_seqs.items():
        try:
            team_hmms[team] = TeamHMM().fit(seq)
        except Exception:
            pass  # skip teams that fail to converge

    # Build joint tensor from training matches only
    joint_tensor, _ = build_joint_tensor(train_df, team_hmms)

    # Most recent Elo per team within the training window (no leakage)
    elo_ratings = (
        train_df.sort_values("date")
        .groupby("team")["team_elo"]
        .last()
        .to_dict()
    )

    predictor = Predictor(
        team_hmms=team_hmms,
        joint_tensor=joint_tensor,
        history_df=full_df,        # date-gating prevents leakage
        elo_ratings=elo_ratings,   # Bayesian Elo update in state dist
    )

    probs = np.zeros((len(test_matches), 3), float)
    for i, (_, row) in enumerate(test_matches.iterrows()):
        r = predictor.predict(row["team"], row["opponent"], row["date"])
        probs[i] = [r["Loss"], r["Draw"], r["Win"]]
    return probs


# ---------------------------------------------------------------------------
# Elo logistic baseline
# ---------------------------------------------------------------------------

def _run_elo(
    train_df: pd.DataFrame,
    test_matches: pd.DataFrame,
) -> np.ndarray:
    train_u = _unique_matches(train_df).dropna(subset=["elo_diff", "outcome"])
    clf = LogisticRegression(max_iter=1000)
    clf.fit(
        train_u[["elo_diff"]].to_numpy(float),
        train_u["outcome"].to_numpy(int),
    )
    raw = clf.predict_proba(test_matches[["elo_diff"]].to_numpy(float))
    return _align_classes(raw, clf.classes_, n=len(test_matches))


# ---------------------------------------------------------------------------
# RF / XGBoost baselines
# ---------------------------------------------------------------------------

def _run_tree(
    train_df: pd.DataFrame,
    test_matches: pd.DataFrame,
    model_type: str,
) -> np.ndarray:
    """
    model_type: 'rf' | 'xgb'
    All features are pre-computed columns in filtered_matches.csv — no
    additional engineering needed.
    """
    available = [f for f in TREE_FEATURES if f in train_df.columns]

    train_u = _unique_matches(train_df).dropna(subset=available + ["outcome"])
    X_train = train_u[available].to_numpy(float)
    y_train = train_u["outcome"].to_numpy(int)

    if model_type == "rf":
        clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    else:
        clf = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=42,
            verbosity=0,
        )

    clf.fit(X_train, y_train)

    X_test = test_matches[available].fillna(1 / 3).to_numpy(float)
    raw = clf.predict_proba(X_test)
    return _align_classes(raw, clf.classes_, n=len(test_matches))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unique_matches(df: pd.DataFrame) -> pd.DataFrame:
    """One row per match — keep the perspective where team < opponent."""
    return (
        df[df["team"] < df["opponent"]]
        .sort_values("date")
        .reset_index(drop=True)
    )


def _align_classes(raw: np.ndarray, classes: np.ndarray, n: int) -> np.ndarray:
    """Re-order predict_proba columns to [Loss=0, Draw=1, Win=2]."""
    aligned = np.zeros((n, 3), float)
    for k, cls in enumerate(classes):
        aligned[:, int(cls)] = raw[:, k]
    return aligned


def _calibration_plot(
    probs: np.ndarray,
    outcomes: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("  (matplotlib unavailable, skipping calibration plot)")
        return

    p_win = probs[:, 2]
    y_win = (outcomes == 2).astype(int)
    bins  = np.linspace(0.0, 1.0, 11)
    idx   = np.clip(np.digitize(p_win, bins) - 1, 0, 9)
    bin_pred = np.array([
        p_win[idx == b].mean() if (idx == b).any() else np.nan for b in range(10)
    ])
    bin_obs = np.array([
        y_win[idx == b].mean() if (idx == b).any() else np.nan for b in range(10)
    ])

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    ax.plot(bin_pred, bin_obs, "o-", label="HMM P(Win)")
    ax.set_xlabel("Predicted P(Win)")
    ax.set_ylabel("Observed Win rate")
    ax.set_title(f"Calibration — {title}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Calibration plot → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data …")
    full_df = load_matches()

    all_results: dict[str, dict] = {}

    for run in EVAL_RUNS:
        tag    = run["tag"]
        cutoff = run["train_cutoff"]
        label  = run["label"]

        print(f"\n{'=' * 60}")
        print(f"  {label}  (train < {cutoff})")
        print(f"{'=' * 60}")

        train_df = full_df[full_df["date"] < cutoff].copy()
        test_df  = run["test_filter"](full_df).copy()

        # De-duplicate to one row per match
        test_matches = (
            _unique_matches(test_df)
            .dropna(subset=["outcome", "elo_diff"])
            .reset_index(drop=True)
        )

        if len(test_matches) == 0:
            print("  No test matches found — skipping.")
            continue

        print(f"  Train matches : {len(train_df)}")
        print(f"  Test  matches : {len(test_matches)}")

        outcomes = test_matches["outcome"].to_numpy(int)

        # ---- models -------------------------------------------------------
        print("  Running HMM …")
        hmm_probs = _run_hmm(train_df, test_matches, full_df)
        print("\n--- HMM Diagnostics ---")
        print("Prediction distribution:", pd.Series(np.argmax(hmm_probs, axis=1)).value_counts().to_dict())
        print("Mean predicted probs (Loss, Draw, Win):", hmm_probs.mean(axis=0).round(4))
        print("Actual outcome distribution:", pd.Series(outcomes).value_counts().to_dict())
        print("Sample probs (first 5):\n", hmm_probs[:5].round(4))

        print("  Running Elo baseline …")
        elo_probs = _run_elo(train_df, test_matches)

        print("  Running Random Forest …")
        rf_probs  = _run_tree(train_df, test_matches, "rf")

        print("  Running XGBoost …")
        xgb_probs = _run_tree(train_df, test_matches, "xgb")

        uniform   = np.full((len(test_matches), 3), 1.0 / 3.0)

        # ---- score --------------------------------------------------------
        results = {
            "HMM":     _metrics(hmm_probs,  outcomes),
            "XGBoost": _metrics(xgb_probs,  outcomes),
            "RF":      _metrics(rf_probs,   outcomes),
            "Elo":     _metrics(elo_probs,  outcomes),
            "Uniform": _metrics(uniform,    outcomes),
        }
        all_results[tag] = {"label": label, "models": results}

        # ---- print table --------------------------------------------------
        header = f"  {'Model':<12} | {'Log-loss':>8} | {'Brier':>6} | {'Acc':>6} | {'RPS':>6}"
        print("\n" + header)
        print("  " + "-" * (len(header) - 2))
        for name, m in results.items():
            print(
                f"  {name:<12} | {m['log_loss']:>8.4f} | {m['brier']:>6.4f} "
                f"| {m['accuracy']:>6.4f} | {m['rps']:>6.4f}"
            )

        # ---- calibration plot (HMM only) ----------------------------------
        _calibration_plot(
            hmm_probs, outcomes,
            ARTIFACTS_DIR / f"calibration_{tag}.png",
            title=label,
        )

    # ---- save combined JSON -----------------------------------------------
    out_json = ARTIFACTS_DIR / "metrics_all.json"
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(all_results, fh, indent=2)
    print(f"\nAll metrics written to: {out_json}")


if __name__ == "__main__":
    main()