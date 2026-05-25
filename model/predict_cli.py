"""
Command-line predictor for the HMM football model.

Loads the trained per-team HMMs and the joint emission tensor from
`ARTIFACTS_DIR`, builds a `Predictor`, and prints Win/Draw/Loss probabilities
plus the inferred current-form state distribution for both teams.

Usage:
    python -m model.predict_cli --team Brazil --opponent Argentina --date 2022-12-13
"""
from __future__ import annotations

import argparse
import pickle
from datetime import date as _date

import numpy as np
import pandas as pd

from model.config import ARTIFACTS_DIR, STATE_LABELS
from model.data_loader import load_matches
from model.predictor import Predictor


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict a football match outcome with the HMM model."
    )
    parser.add_argument("--team", required=True, help="Team name (must match CSV).")
    parser.add_argument("--opponent", required=True, help="Opponent team name.")
    parser.add_argument(
        "--date",
        default=_date.today().isoformat(),
        help="Match date YYYY-MM-DD (default: today). Only matches strictly "
             "before this date are used for state inference.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print extra detail.")
    return parser.parse_args()


def _load_artifacts() -> tuple[dict, np.ndarray]:
    hmms_path  = ARTIFACTS_DIR / "team_hmms.pkl"
    joint_path = ARTIFACTS_DIR / "joint_tensor.npy"
    if not hmms_path.exists() or not joint_path.exists():
        raise FileNotFoundError(
            f"Missing artifacts in {ARTIFACTS_DIR}. Run `python -m model.train` first."
        )
    with open(hmms_path, "rb") as fh:
        team_hmms = pickle.load(fh)
    joint_tensor = np.load(joint_path)
    return team_hmms, joint_tensor


def _fmt_state(name: str, dist) -> str:
    short = ["Poor", "Neutral", "Peak"]
    parts = [f"{short[i]} {float(dist[i]):.2f}" for i in range(len(short))]
    return f"  state_{name}= " + "  ".join(parts)


def main() -> None:
    args = _parse_args()
    match_date = pd.to_datetime(args.date)

    team_hmms, joint_tensor = _load_artifacts()
    history_df = load_matches()

    predictor = Predictor(
        team_hmms=team_hmms,
        joint_tensor=joint_tensor,
        history_df=history_df,
    )

    result = predictor.predict(args.team, args.opponent, match_date)

    # Pretty-print a small summary table.
    print(f"\n{args.team} vs {args.opponent} on {match_date.date()}")
    print(f"  P(Win)  = {float(result['Win']):.4f}")
    print(f"  P(Draw) = {float(result['Draw']):.4f}")
    print(f"  P(Loss) = {float(result['Loss']):.4f}")
    print(_fmt_state(args.team, result["state_team"]))
    print(_fmt_state(args.opponent, result["state_opp"]))

    if args.verbose:
        print("\n[verbose] state label legend:", STATE_LABELS)
        print("[verbose] raw result dict:", result)


if __name__ == "__main__":
    main()
