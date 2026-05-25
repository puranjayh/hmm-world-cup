"""
evaluate.py — Evaluate the trained HMM predictor on the held-out test split.

Walks the test matches in chronological order. For each unique match
(deduplicated by team < opponent), it asks the Predictor for P(W/D/L)
using ONLY observations strictly before that match's date (the Predictor
handles that internally). Then it scores predictions against the true
outcome.

Metrics:
  - multiclass log loss (lower is better, uniform = log(3) ≈ 1.0986)
  - Brier score (mean squared error vs one-hot target, lower is better)
  - top-1 accuracy
  - Elo logistic-regression baseline (sanity check) trained on train split

Also writes a calibration plot to artifacts/calibration.png.

Run:
    python -m model.evaluate
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from model.config import ARTIFACTS_DIR, OUTCOME_LABELS
from model.data_loader import load_matches, split_train_test
from model.predictor import Predictor


def _load_artifacts() -> tuple[dict, np.ndarray]:
    with open(ARTIFACTS_DIR / "team_hmms.pkl", "rb") as fh:
        team_hmms = pickle.load(fh)
    joint_tensor = np.load(ARTIFACTS_DIR / "joint_tensor.npy")
    return team_hmms, joint_tensor


def _unique_matches(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one row per match by alphabetical perspective."""
    mask = df["team"] < df["opponent"]
    return df.loc[mask].sort_values("date").reset_index(drop=True)


def _metrics(probs: np.ndarray, outcomes: np.ndarray) -> dict:
    """probs shape (N,3) over [Loss, Draw, Win]; outcomes shape (N,) in {0,1,2}."""
    eps = 1e-12
    n = len(outcomes)
    p_true = probs[np.arange(n), outcomes]
    log_loss = float(-np.mean(np.log(np.clip(p_true, eps, 1.0))))

    one_hot = np.zeros_like(probs)
    one_hot[np.arange(n), outcomes] = 1.0
    brier = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))

    preds = np.argmax(probs, axis=1)
    accuracy = float(np.mean(preds == outcomes))

    return {"n": int(n), "log_loss": log_loss, "brier": brier, "accuracy": accuracy}


def _elo_baseline(train_df: pd.DataFrame, test_matches: pd.DataFrame) -> np.ndarray:
    """Multinomial logistic regression on elo_diff -> Loss/Draw/Win."""
    from sklearn.linear_model import LogisticRegression

    train_unique = _unique_matches(train_df).dropna(subset=["elo_diff", "outcome"])
    X_train = train_unique[["elo_diff"]].to_numpy(dtype=float)
    y_train = train_unique["outcome"].to_numpy(dtype=int)
    clf = LogisticRegression(multi_class="multinomial", max_iter=1000)
    clf.fit(X_train, y_train)

    X_test = test_matches[["elo_diff"]].to_numpy(dtype=float)
    # Align column order to [Loss(0), Draw(1), Win(2)] regardless of clf.classes_.
    raw = clf.predict_proba(X_test)
    aligned = np.zeros((len(X_test), 3), dtype=float)
    for k, cls in enumerate(clf.classes_):
        aligned[:, int(cls)] = raw[:, k]
    return aligned


def _calibration_plot(probs: np.ndarray, outcomes: np.ndarray, out_path: Path) -> None:
    """Reliability diagram for P(Win) — saves PNG; ignored if matplotlib missing."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("  (matplotlib unavailable, skipping calibration plot)")
        return

    p_win = probs[:, 2]
    y_win = (outcomes == 2).astype(int)
    bins = np.linspace(0.0, 1.0, 11)
    idx = np.clip(np.digitize(p_win, bins) - 1, 0, 9)
    bin_pred = np.array([p_win[idx == b].mean() if (idx == b).any() else np.nan for b in range(10)])
    bin_obs = np.array([y_win[idx == b].mean() if (idx == b).any() else np.nan for b in range(10)])

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    ax.plot(bin_pred, bin_obs, "o-", label="HMM P(Win)")
    ax.set_xlabel("Predicted P(Win)")
    ax.set_ylabel("Observed Win rate")
    ax.set_title("Calibration — HMM P(Win) on holdout")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    df = load_matches()
    train_df, test_df = split_train_test(df)
    test_matches = _unique_matches(test_df).dropna(subset=["outcome", "elo_diff"]).reset_index(drop=True)
    print(f"Evaluating on {len(test_matches)} unique test matches "
          f"({test_matches['date'].min().date()} -> {test_matches['date'].max().date()})")

    team_hmms, joint_tensor = _load_artifacts()
    predictor = Predictor(team_hmms=team_hmms, joint_tensor=joint_tensor, history_df=df)

    probs = np.zeros((len(test_matches), 3), dtype=float)
    outcomes = np.zeros(len(test_matches), dtype=int)
    for i, row in test_matches.iterrows():
        r = predictor.predict(row["team"], row["opponent"], row["date"])
        probs[i] = [r["Loss"], r["Draw"], r["Win"]]
        outcomes[i] = int(row["outcome"])

    hmm_metrics = _metrics(probs, outcomes)

    # Uniform baseline.
    uniform = np.full_like(probs, 1.0 / 3.0)
    uni_metrics = _metrics(uniform, outcomes)

    # Elo logistic baseline.
    elo_probs = _elo_baseline(train_df, test_matches)
    elo_metrics = _metrics(elo_probs, outcomes)

    summary = {
        "test_matches": int(len(test_matches)),
        "test_date_range": [str(test_matches["date"].min().date()),
                            str(test_matches["date"].max().date())],
        "hmm": hmm_metrics,
        "elo_baseline": elo_metrics,
        "uniform_baseline": uni_metrics,
        "outcome_labels": OUTCOME_LABELS,
    }

    out_json = ARTIFACTS_DIR / "metrics.json"
    with open(out_json, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(summary, indent=2))

    _calibration_plot(probs, outcomes, ARTIFACTS_DIR / "calibration.png")

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    for name, m in (("HMM", hmm_metrics),
                    ("Elo baseline", elo_metrics),
                    ("Uniform", uni_metrics)):
        print(f"  {name:13s} | log_loss={m['log_loss']:.4f} "
              f"| brier={m['brier']:.4f} | acc={m['accuracy']:.4f}")
    print(f"\nMetrics written to: {out_json}")


if __name__ == "__main__":
    main()
