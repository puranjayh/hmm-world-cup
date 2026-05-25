"""
Training pipeline for the HMM football predictor.

Steps:
  1. Load and chronologically split the match data.
  2. Build per-team observation sequences (Loss/Draw/Win).
  3. Fit one 3-state HMM per team (provided len(seq) >= MIN_MATCHES).
  4. Build the joint emission tensor P(outcome | state_team, state_opp) from
     the train split using the sibling `joint_emission` module.
  5. Persist HMMs, the combined dict, the joint tensor and a metadata JSON
     into `ARTIFACTS_DIR` so the predictor CLI can reload everything.
"""
from __future__ import annotations

import json
import pickle
import re
import traceback

import numpy as np

from model.config import (
    ARTIFACTS_DIR,
    MIN_MATCHES,
    OUTCOME_LABELS,
    STATE_LABELS,
)
from model.data_loader import load_matches, split_train_test, team_sequences
from model.hmm_team import TeamHMM
from model.joint_emission import build_joint_tensor

try:  # tqdm is nice-to-have; fall back to a no-op wrapper if missing.
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(it, **kwargs):  # type: ignore
        return it


def _safe_name(team: str) -> str:
    """Make a filesystem-safe filename from a team name."""
    return re.sub(r"[^A-Za-z0-9_-]+", "_", team).strip("_") or "team"


def _date_range(df) -> list:
    if df.empty:
        return [None, None]
    return [df["date"].min(), df["date"].max()]


def main() -> None:
    # 1. Load + split.
    df = load_matches()
    train_df, test_df = split_train_test(df)
    print(
        f"Loaded {len(df)} matches | train={len(train_df)} "
        f"({train_df['date'].min().date() if len(train_df) else '-'} -> "
        f"{train_df['date'].max().date() if len(train_df) else '-'}) | "
        f"test={len(test_df)} "
        f"({test_df['date'].min().date() if len(test_df) else '-'} -> "
        f"{test_df['date'].max().date() if len(test_df) else '-'})"
    )

    # 2. Per-team sequences.
    seqs = team_sequences(train_df)
    print(f"Found {len(seqs)} unique teams in train set.")

    # 3. Fit one HMM per team that has enough history.
    team_hmms: dict[str, TeamHMM] = {}
    skipped: list[str] = []

    for team in tqdm(sorted(seqs.keys()), desc="Fitting HMMs"):
        seq = seqs[team]
        if len(seq) < MIN_MATCHES:
            skipped.append(team)
            continue
        try:
            hmm = TeamHMM().fit(seq)
            team_hmms[team] = hmm
            print(f"Trained HMM for {team} (N={len(seq)})")
        except Exception as exc:  # noqa: BLE001
            print(f"  ! Failed to fit HMM for {team}: {exc}")
            traceback.print_exc()
            skipped.append(team)

    # 4. Persist each HMM individually + as one combined pickle.
    for team, hmm in team_hmms.items():
        hmm.save(ARTIFACTS_DIR / f"hmm_{_safe_name(team)}.pkl")
    combined_path = ARTIFACTS_DIR / "team_hmms.pkl"
    with open(combined_path, "wb") as fh:
        pickle.dump(team_hmms, fh)

    # 5. Joint emission tensor P(outcome | state_team, state_opp).
    joint_tensor, diag = build_joint_tensor(train_df, team_hmms)
    np.save(ARTIFACTS_DIR / "joint_tensor.npy", joint_tensor)

    # 6. Metadata JSON.
    metadata = {
        "trained_teams": sorted(team_hmms.keys()),
        "skipped_teams": sorted(skipped),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_date_range": _date_range(train_df),
        "test_date_range": _date_range(test_df),
        "joint_diagnostics": diag,
        "state_labels": STATE_LABELS,
        "outcome_labels": OUTCOME_LABELS,
    }
    meta_path = ARTIFACTS_DIR / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(metadata, indent=2, default=str))

    # 7. Summary block.
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Teams trained : {len(team_hmms)}")
    print(f"  Teams skipped : {len(skipped)}")
    print(f"  Joint tensor  : shape={joint_tensor.shape}")
    print(f"  Artifacts dir : {ARTIFACTS_DIR}")
    print(f"    - team_hmms.pkl   ({combined_path})")
    print(f"    - joint_tensor.npy")
    print(f"    - metadata.json   ({meta_path})")


if __name__ == "__main__":
    main()
