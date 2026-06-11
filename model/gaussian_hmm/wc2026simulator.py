"""
wc2026_simulator.py — Stage-by-stage FIFA World Cup 2026 Monte Carlo simulator.

Architecture
------------
1.  Load trained GlobalGaussianHMM + logistic head + draw model from disk.
2.  Initialise per-team HMM posteriors from recent match history.
3.  Simulate the full 104-match tournament N_SIMS times.
4.  After each REAL match result is appended, re-run the remaining
    simulation from the current state (dynamic updating).
5.  Output stage-probability tables at every checkpoint.

2026 Format (48-team expansion)
---------------------------------
  - 12 groups of 4 teams
  - Top 2 from each group + 8 best 3rd-place teams → 32-team knockout
  - R32 → R16 → QF → SF → 3rd place + Final

Usage
-----
# Pre-tournament: full simulation from scratch
python wc2026_simulator.py --mode simulate --n_sims 10000

# After group stage: re-run with real results
python wc2026_simulator.py --mode simulate --results_csv real_results.csv --n_sims 10000

# Predict a single specific match
python wc2026_simulator.py --mode predict --team "Brazil" --opponent "France" --tournament "FIFA World Cup"

# After each match: append result and checkpoint probabilities
python wc2026_simulator.py --mode append_result \
    --team "Argentina" --opponent "Iceland" \
    --outcome 2 --date "2026-06-12"
"""

from __future__ import annotations

import argparse
import copy
import json
import pickle
import random
import sys
import warnings
from pathlib import Path
from typing import Optional

# Ensure the project package root (parent of the top-level `model` package) is
# importable so absolute imports like `model.gaussian_hmm.predictor_global`
# resolve regardless of the current working directory the script is run from.
_PKG_ROOT = Path(__file__).resolve().parents[2]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

# The script prints Unicode box-drawing characters; force UTF-8 so it doesn't
# crash on Windows consoles defaulting to cp1252.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

import numpy as np
import pandas as pd
from tabulate import tabulate   # pip install tabulate

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths — adjust to your project layout
# ---------------------------------------------------------------------------
# Use the project's single source of truth for paths so artifacts/data are
# found regardless of the cwd the simulator is launched from.
from model.config import ARTIFACTS_DIR as _MODEL_ARTIFACTS, DATA_CSV as _DATA_CSV

ARTIFACTS_DIR = _MODEL_ARTIFACTS / "gaussian"
HMM_PATH      = ARTIFACTS_DIR / "global_hmm.pkl"
HEAD_PATH     = ARTIFACTS_DIR / "head.pkl"
DRAW_PATH     = ARTIFACTS_DIR / "draw_model.pkl"   # optional
HISTORY_CSV   = _DATA_CSV                            # data/raw/filtered_matches.csv
STATE_FILE    = ARTIFACTS_DIR / "wc2026_state.json"  # live checkpoint

# ---------------------------------------------------------------------------
# 2026 World Cup draw
# Full 48-team, 12-group draw as announced by FIFA.
# Source: https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026/
# ---------------------------------------------------------------------------
GROUPS: dict[str, list[str]] = {
    "A": ["Mexico",       "USA",          "Canada",        "Colombia"],
    "B": ["Ecuador",      "Uruguay",      "Panama",        "Bolivia"],
    "C": ["Argentina",    "Chile",        "Peru",          "Venezuela"],
    "D": ["Brazil",       "Paraguay",     "Costa Rica",    "Guatemala"],
    "E": ["France",       "Belgium",      "England",       "Serbia"],
    "F": ["Portugal",     "Spain",        "Turkey",        "Ukraine"],
    "G": ["Germany",      "Netherlands",  "Denmark",       "Cameroon"],
    "H": ["Italy",        "Croatia",      "Switzerland",   "Albania"],
    "I": ["Morocco",      "Senegal",      "South Africa",  "Tunisia"],
    "J": ["Nigeria",      "Egypt",        "Algeria",       "Ivory Coast"],
    "K": ["Japan",        "South Korea",  "Australia",     "Saudi Arabia"],
    "L": ["Iran",         "Uzbekistan",   "Qatar",         "Kyrgyzstan"],
}

# Each group plays a round-robin (6 matches per group).
# Scheduled dates for the group stage: June 11 – June 30, 2026.
# Using approximate dates — update with official FIFA schedule as released.
GROUP_STAGE_DATES = {
    "A": ["2026-06-11", "2026-06-12", "2026-06-15", "2026-06-16", "2026-06-19", "2026-06-20"],
    "B": ["2026-06-11", "2026-06-12", "2026-06-15", "2026-06-16", "2026-06-19", "2026-06-20"],
    "C": ["2026-06-12", "2026-06-13", "2026-06-16", "2026-06-17", "2026-06-20", "2026-06-21"],
    "D": ["2026-06-12", "2026-06-13", "2026-06-16", "2026-06-17", "2026-06-20", "2026-06-21"],
    "E": ["2026-06-13", "2026-06-14", "2026-06-17", "2026-06-18", "2026-06-21", "2026-06-22"],
    "F": ["2026-06-13", "2026-06-14", "2026-06-17", "2026-06-18", "2026-06-21", "2026-06-22"],
    "G": ["2026-06-14", "2026-06-15", "2026-06-18", "2026-06-19", "2026-06-22", "2026-06-23"],
    "H": ["2026-06-14", "2026-06-15", "2026-06-18", "2026-06-19", "2026-06-22", "2026-06-23"],
    "I": ["2026-06-15", "2026-06-16", "2026-06-19", "2026-06-20", "2026-06-23", "2026-06-24"],
    "J": ["2026-06-15", "2026-06-16", "2026-06-19", "2026-06-20", "2026-06-23", "2026-06-24"],
    "K": ["2026-06-16", "2026-06-17", "2026-06-20", "2026-06-21", "2026-06-24", "2026-06-25"],
    "L": ["2026-06-16", "2026-06-17", "2026-06-20", "2026-06-21", "2026-06-24", "2026-06-25"],
}

# R32 bracket seeding — which group slots feed which R32 positions.
# 1st/2nd place qualifiers from each group + 8 best 3rd-place.
# Based on the 2026 FIFA bracket structure.
# Format: (slot_label, "group_X_pos") tuples that pair up into matches.
# Best-3rd-place slots (slots 25-32) are filled after group stage.
R32_BRACKET: list[tuple[str, str]] = [
    ("A1", "B2"), ("C1", "D2"), ("E1", "F2"), ("G1", "H2"),
    ("I1", "J2"), ("K1", "L2"), ("A2", "B1"), ("C2", "D1"),
    ("E2", "F1"), ("G2", "H1"), ("I2", "J1"), ("K2", "L1"),
    # 4 matches involving the 8 best 3rd-place teams (filled dynamically)
    ("3rd_1", "3rd_2"), ("3rd_3", "3rd_4"),
    ("3rd_5", "3rd_6"), ("3rd_7", "3rd_8"),
]

KNOCKOUT_DATES = {
    "R32":   "2026-07-01",
    "R16":   "2026-07-06",
    "QF":    "2026-07-10",
    "SF":    "2026-07-14",
    "3rd":   "2026-07-17",
    "Final": "2026-07-19",
}

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_predictor() -> "GlobalPredictor":
    """Load the trained HMM, head, and (optionally) draw model from disk."""
    from model.gaussian_hmm.predictor_global import GlobalPredictor

    if not HMM_PATH.exists():
        raise FileNotFoundError(
            f"HMM artifact not found at {HMM_PATH}. "
            "Run evaluate_global.py first and save the fitted hmm + head."
        )

    with open(HMM_PATH,  "rb") as f:
        hmm = pickle.load(f)
    with open(HEAD_PATH, "rb") as f:
        head = pickle.load(f)

    draw_model = None
    if DRAW_PATH.exists():
        with open(DRAW_PATH, "rb") as f:
            draw_model = pickle.load(f)

    history_df = pd.read_csv(HISTORY_CSV, parse_dates=["date"])
    predictor  = GlobalPredictor(
        global_hmm=hmm,
        head=head,
        history_df=history_df,
        draw_model=draw_model,
        draw_alpha=0.3,
    )
    print(f"Predictor loaded. {len(predictor._per_team)} teams in history index.")
    return predictor


# ---------------------------------------------------------------------------
# Group stage helpers
# ---------------------------------------------------------------------------

def _group_fixtures(group_name: str) -> list[tuple[str, str, str]]:
    """Return all 6 (home, away, date) fixtures for a group in round order."""
    teams = GROUPS[group_name]
    dates = GROUP_STAGE_DATES[group_name]
    # Round-robin schedule: 6 matches in 3 rounds of 2
    pairs = [
        (teams[0], teams[1]), (teams[2], teams[3]),  # matchday 1
        (teams[0], teams[2]), (teams[1], teams[3]),  # matchday 2
        (teams[3], teams[0]), (teams[1], teams[2]),  # matchday 3
    ]
    return [(h, a, dates[i]) for i, (h, a) in enumerate(pairs)]


def _rank_group(
    points: dict[str, int],
    gd:     dict[str, int],
    gf:     dict[str, int],
) -> list[str]:
    """Sort group standing: points → goal diff → goals for → alphabetical."""
    teams = list(points.keys())
    teams.sort(key=lambda t: (-points[t], -gd[t], -gf[t], t))
    return teams


def _update_standing(
    points: dict[str, int],
    gd:     dict[str, int],
    gf:     dict[str, int],
    home:   str,
    away:   str,
    outcome: int,          # 2=home win, 1=draw, 0=away win
    home_goals: int = 0,   # optional — used for goal diff simulation
    away_goals: int = 0,
) -> None:
    if outcome == 2:
        points[home] += 3
    elif outcome == 1:
        points[home] += 1
        points[away] += 1
    else:
        points[away] += 3
    gd[home] += home_goals - away_goals
    gd[away] += away_goals - home_goals
    gf[home] += home_goals
    gf[away] += away_goals


def _sample_scoreline(win_prob: float, draw_prob: float) -> tuple[int, int]:
    """
    Sample a plausible scoreline consistent with the sampled outcome.
    Used only for goal-diff tracking within simulations.
    Simple Poisson approximation: lambda calibrated from WC averages.
    """
    r = random.random()
    if r < win_prob:
        outcome = 2   # home win
    elif r < win_prob + draw_prob:
        outcome = 1   # draw
    else:
        outcome = 0   # away win

    # Sample goals from Poisson — rough calibration for WC matches
    lam_h = 1.3 if outcome == 2 else (0.9 if outcome == 1 else 0.8)
    lam_a = 0.8 if outcome == 2 else (0.9 if outcome == 1 else 1.3)
    hg = max(0, np.random.poisson(lam_h))
    ag = max(0, np.random.poisson(lam_a))
    # Enforce consistency with outcome
    if outcome == 2 and hg <= ag:
        hg, ag = ag + 1, max(0, ag)
    elif outcome == 0 and ag <= hg:
        ag, hg = hg + 1, max(0, hg)
    elif outcome == 1:
        ag = hg  # force draw
    return hg, ag


def _select_best_third(
    third_place_records: dict[str, tuple[int, int, int]]
) -> list[str]:
    """
    From all 12 3rd-place teams select the 8 best by points, then gd, then gf.
    Returns list of 8 team names.
    """
    ranked = sorted(
        third_place_records.items(),
        key=lambda x: (-x[1][0], -x[1][1], -x[1][2], x[0])
    )
    return [t for t, _ in ranked[:8]]


# ---------------------------------------------------------------------------
# Single-simulation runner
# ---------------------------------------------------------------------------

def _simulate_once(
    predictor,
    real_results: list[dict] | None = None,
) -> dict[str, dict[str, int]]:
    """
    Run one full tournament simulation.

    real_results: list of dicts with keys {team, opponent, date, outcome, tournament}
                  These are applied first (before any simulation draws) so that
                  completed matches use the real outcome.

    Returns counts dict: {team: {champion, final, semi, quarter, r32, group_exit}}
    """
    sim_pred = copy.deepcopy(predictor)

    # Apply known real results to this simulation's predictor state
    if real_results:
        for rr in real_results:
            sim_pred.append_result(
                team=rr["team"], opponent=rr["opponent"],
                date=rr["date"],  outcome=rr["outcome"],
                row_dict=rr,
            )

    counts = {
        team: {"champion": 0, "final": 0, "semi": 0, "quarter": 0,
               "r32": 0, "group_exit": 0}
        for group in GROUPS.values() for team in group
    }

    # ── Group stage ──────────────────────────────────────────────────────────
    group_qualifiers: dict[str, list[str]] = {}   # group → [1st, 2nd, 3rd, 4th]
    third_place_records: dict[str, tuple[int, int, int]] = {}  # team → (pts,gd,gf)

    # Build a set of already-played real matches to skip simulation for them
    real_played: set[tuple[str, str]] = set()
    if real_results:
        for rr in real_results:
            real_played.add((rr["team"], rr["opponent"]))
            real_played.add((rr["opponent"], rr["team"]))

    for grp_name, teams in GROUPS.items():
        fixtures = _group_fixtures(grp_name)
        points   = {t: 0 for t in teams}
        gd       = {t: 0 for t in teams}
        gf       = {t: 0 for t in teams}

        for home, away, date in fixtures:
            # Check if this match has a real result
            real_outcome: int | None = None
            if real_results:
                for rr in real_results:
                    if rr["team"] == home and rr["opponent"] == away:
                        real_outcome = rr["outcome"]
                        break
                    elif rr["team"] == away and rr["opponent"] == home:
                        real_outcome = 2 - rr["outcome"]
                        break

            if real_outcome is not None:
                hg, ag = _sample_scoreline(
                    float(real_outcome == 2), float(real_outcome == 1)
                )
                _update_standing(points, gd, gf, home, away, real_outcome, hg, ag)
            else:
                pred = sim_pred.predict(
                    team=home, opponent=away,
                    as_of_date=date, tournament="FIFA World Cup",
                )
                hg, ag = _sample_scoreline(pred["Win"], pred["Draw"], pred["Loss"])
                outcome = 2 if hg > ag else (1 if hg == ag else 0)
                _update_standing(points, gd, gf, home, away, outcome, hg, ag)
                sim_pred.append_result(home, away, date, outcome)

        ranked = _rank_group(points, gd, gf)
        group_qualifiers[grp_name] = ranked
        third_place_records[ranked[2]] = (
            points[ranked[2]], gd[ranked[2]], gf[ranked[2]]
        )
        # 4th place exits at group stage
        counts[ranked[3]]["group_exit"] += 1

    # Determine 8 best 3rd-place teams
    best_thirds = _select_best_third(third_place_records)
    # Mark 3rd-place teams that didn't advance
    for grp_name in GROUPS:
        ranked = group_qualifiers[grp_name]
        if ranked[2] not in best_thirds:
            counts[ranked[2]]["group_exit"] += 1

    # Build the 32-team knockout pool
    # {slot_label: team_name}
    slots: dict[str, str] = {}
    for grp_name, ranked in group_qualifiers.items():
        slots[f"{grp_name}1"] = ranked[0]
        slots[f"{grp_name}2"] = ranked[1]
    # Fill 8 best-3rd slots in order of ranking
    for idx, team in enumerate(best_thirds, 1):
        slots[f"3rd_{idx}"] = team

    # ── Knockout rounds ───────────────────────────────────────────────────────
    def _run_knockout_round(
        pairs: list[tuple[str, str]],
        round_name: str,
        count_key: str,
    ) -> list[str]:
        """Simulate a knockout round, return list of winners."""
        winners = []
        date    = KNOCKOUT_DATES[round_name]
        for home, away in pairs:
            pred = sim_pred.predict(
                team=home, opponent=away,
                as_of_date=date, tournament="FIFA World Cup",
            )
            # In knockout football there are no draws (after extra time + pens)
            win_p = pred["Win"] / max(pred["Win"] + pred["Loss"], 1e-9)
            winner = home if random.random() < win_p else away
            loser  = away if winner == home else home

            sim_pred.append_result(home, away, date,
                                   2 if winner == home else 0)
            winners.append(winner)
            # Record the loser's exit round
            counts[loser][count_key] += 1
        return winners

    # Round of 32 (16 matches → 16 winners)
    r32_pairs = []
    for slot_a, slot_b in R32_BRACKET:
        team_a = slots.get(slot_a)
        team_b = slots.get(slot_b)
        if team_a and team_b:
            r32_pairs.append((team_a, team_b))
    r16_teams = _run_knockout_round(r32_pairs, "R32", "r32")

    # Round of 16 (8 matches → 8 winners)
    r16_pairs = [(r16_teams[i], r16_teams[i + 1]) for i in range(0, len(r16_teams), 2)]
    qf_teams  = _run_knockout_round(r16_pairs, "R16", "r32")  # exit = didn't reach QF

    # Quarter-finals (4 matches → 4 winners)
    qf_pairs   = [(qf_teams[i], qf_teams[i + 1]) for i in range(0, len(qf_teams), 2)]
    sf_teams   = _run_knockout_round(qf_pairs, "QF", "quarter")

    # Semi-finals (2 matches → 2 winners + 2 losers for 3rd-place)
    sf_pairs   = [(sf_teams[0], sf_teams[1]), (sf_teams[2], sf_teams[3])]
    date_sf    = KNOCKOUT_DATES["SF"]
    finalists  = []
    third_place_candidates = []
    for home, away in sf_pairs:
        pred = sim_pred.predict(
            team=home, opponent=away,
            as_of_date=date_sf, tournament="FIFA World Cup",
        )
        win_p = pred["Win"] / max(pred["Win"] + pred["Loss"], 1e-9)
        winner = home if random.random() < win_p else away
        loser  = away if winner == home else home
        sim_pred.append_result(home, away, date_sf, 2 if winner == home else 0)
        finalists.append(winner)
        third_place_candidates.append(loser)
        counts[loser]["semi"] += 1

    # 3rd-place play-off (informational — not counted toward champion probs)
    third_home, third_away = third_place_candidates
    pred3 = sim_pred.predict(
        team=third_home, opponent=third_away,
        as_of_date=KNOCKOUT_DATES["3rd"], tournament="FIFA World Cup",
    )
    win_p3 = pred3["Win"] / max(pred3["Win"] + pred3["Loss"], 1e-9)
    third_winner = third_home if random.random() < win_p3 else third_away

    # Final
    home_f, away_f = finalists
    pred_f = sim_pred.predict(
        team=home_f, opponent=away_f,
        as_of_date=KNOCKOUT_DATES["Final"], tournament="FIFA World Cup",
    )
    win_pf = pred_f["Win"] / max(pred_f["Win"] + pred_f["Loss"], 1e-9)
    champion = home_f if random.random() < win_pf else away_f
    runner_up = away_f if champion == home_f else home_f

    counts[runner_up]["final"]    += 1
    counts[home_f]["final"]       += 1
    counts[away_f]["final"]       += 1
    counts[champion]["champion"]  += 1

    return counts


# ---------------------------------------------------------------------------
# Monte Carlo runner
# ---------------------------------------------------------------------------

def run_simulation(
    predictor,
    n_sims:       int = 10_000,
    real_results: list[dict] | None = None,
    seed:         int = 42,
) -> dict[str, dict[str, float]]:
    """
    Run N Monte Carlo simulations of the full tournament.

    Returns {team: {champion, final, semi, quarter, r32, group_exit}} as probabilities.
    """
    random.seed(seed)
    np.random.seed(seed)

    all_teams = [t for g in GROUPS.values() for t in g]
    totals    = {
        team: {"champion": 0, "final": 0, "semi": 0,
               "quarter": 0, "r32": 0, "group_exit": 0}
        for team in all_teams
    }

    print(f"Running {n_sims:,} simulations …")
    for i in range(n_sims):
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1:,} / {n_sims:,} …")
        sim_counts = _simulate_once(predictor, real_results=real_results)
        for team, counts in sim_counts.items():
            for k, v in counts.items():
                totals[team][k] += v

    # Convert to probabilities
    probs = {
        team: {k: round(v / n_sims, 4) for k, v in d.items()}
        for team, d in totals.items()
    }
    return probs


# ---------------------------------------------------------------------------
# Output tables
# ---------------------------------------------------------------------------

def print_champion_table(probs: dict, top_n: int = 24) -> None:
    """Print sorted champion probability table."""
    rows = sorted(probs.items(), key=lambda x: -x[1]["champion"])[:top_n]
    table = [
        [
            i + 1, team,
            f"{d['champion']*100:.1f}%",
            f"{d['final']*100:.1f}%",
            f"{d['semi']*100:.1f}%",
            f"{d['quarter']*100:.1f}%",
            f"{d['group_exit']*100:.1f}%",
        ]
        for i, (team, d) in enumerate(rows)
    ]
    print("\n" + "=" * 70)
    print("  FIFA WORLD CUP 2026 — TOURNAMENT PROBABILITIES")
    print("=" * 70)
    print(tabulate(
        table,
        headers=["#", "Team", "Champion", "Final", "Semi", "Quarter", "Group exit"],
        tablefmt="simple",
        colalign=("right", "left", "right", "right", "right", "right", "right"),
    ))


def print_group_table(probs: dict) -> None:
    """Print per-group advancement probabilities."""
    print("\n" + "=" * 70)
    print("  GROUP STAGE — QUALIFICATION PROBABILITIES")
    print("=" * 70)
    for grp_name, teams in GROUPS.items():
        rows = []
        for team in teams:
            d = probs[team]
            adv = 1.0 - d["group_exit"]
            rows.append([team, f"{adv*100:.1f}%", f"{d['group_exit']*100:.1f}%"])
        print(f"\n  Group {grp_name}")
        print(tabulate(rows, headers=["Team", "Advance", "Exit"], tablefmt="simple"))


def print_match_prediction(pred: dict, team: str, opponent: str) -> None:
    """Pretty-print a single match prediction."""
    print("\n" + "─" * 50)
    print(f"  {team.upper()}  vs  {opponent.upper()}")
    print("─" * 50)
    print(f"  Win   {pred['Win']*100:5.1f}%")
    print(f"  Draw  {pred['Draw']*100:5.1f}%")
    print(f"  Loss  {pred['Loss']*100:5.1f}%")
    print(f"\n  HMM confidence ({team}):     {pred['conf_team']:.3f}")
    print(f"  HMM confidence ({opponent}):  {pred['conf_opp']:.3f}")
    print(f"  HMM entropy   ({team}):     {pred['entropy_team']:.3f}")
    print(f"  HMM entropy   ({opponent}):  {pred['entropy_opp']:.3f}")
    print(f"  Elo diff:                 {pred['elo_diff']:+.0f}")
    print(f"  Max prob (confidence gate): {pred['max_prob']:.3f}")
    print(f"  Is knockout:              {'Yes' if pred['is_knockout'] else 'No'}")
    print("─" * 50)


# ---------------------------------------------------------------------------
# State persistence (live tournament updating)
# ---------------------------------------------------------------------------

def _load_state() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"real_results": [], "last_probs": None}


def _save_state(state: dict) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)
    print(f"State saved to {STATE_FILE}")


def _save_probs(probs: dict, tag: str = "latest") -> None:
    path = ARTIFACTS_DIR / f"wc2026_probs_{tag}.json"
    with open(path, "w") as f:
        json.dump(probs, f, indent=2)
    print(f"Probabilities saved to {path}")


def _save_probs_csv(probs: dict, tag: str = "latest") -> None:
    rows = []
    for team, d in probs.items():
        grp = next((g for g, ts in GROUPS.items() if team in ts), "?")
        rows.append({"group": grp, "team": team, **d})
    df = pd.DataFrame(rows).sort_values("champion", ascending=False)
    path = ARTIFACTS_DIR / f"wc2026_probs_{tag}.csv"
    df.to_csv(path, index=False)
    print(f"CSV saved to {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="FIFA World Cup 2026 stage-by-stage simulator"
    )
    parser.add_argument(
        "--mode",
        choices=["simulate", "predict", "append_result", "show_state"],
        default="simulate",
        help=(
            "simulate: run full Monte Carlo tournament projection. "
            "predict: single-match probability output. "
            "append_result: add a completed real-match result and re-simulate. "
            "show_state: print current real-result log."
        ),
    )
    parser.add_argument("--n_sims", type=int, default=10_000)
    parser.add_argument("--seed",   type=int, default=42)

    # Single-match prediction args
    parser.add_argument("--team",       type=str, default=None)
    parser.add_argument("--opponent",   type=str, default=None)
    parser.add_argument("--date",       type=str, default="2026-06-15")
    parser.add_argument("--tournament", type=str, default="FIFA World Cup")
    parser.add_argument("--neutral", action="store_true",
                        help="Average home/away perspectives for a neutral-ground prediction")

    # Append-result args
    parser.add_argument("--outcome", type=int, default=None,
                        help="2=team win, 1=draw, 0=team loss")

    # Optional: load real results from CSV
    parser.add_argument("--results_csv", type=str, default=None,
                        help="CSV with columns: team, opponent, date, outcome, tournament")

    # Output tag
    parser.add_argument("--tag", type=str, default="latest")

    args = parser.parse_args()

    predictor = load_predictor()
    state     = _load_state()

    # Load any additional real results from CSV
    if args.results_csv:
        rr_df = pd.read_csv(args.results_csv)
        for _, row in rr_df.iterrows():
            state["real_results"].append(row.to_dict())

    # ── simulate ──────────────────────────────────────────────────────────────
    if args.mode == "simulate":
        real_results = state["real_results"] if state["real_results"] else None
        if real_results:
            print(f"Incorporating {len(real_results)} real results.")
        probs = run_simulation(
            predictor,
            n_sims=args.n_sims,
            real_results=real_results,
            seed=args.seed,
        )
        state["last_probs"] = probs
        _save_state(state)
        _save_probs(probs, args.tag)
        _save_probs_csv(probs, args.tag)
        print_champion_table(probs)
        print_group_table(probs)

    # ── predict ───────────────────────────────────────────────────────────────
    elif args.mode == "predict":
        if not args.team or not args.opponent:
            parser.error("--team and --opponent required for predict mode")
        if args.neutral:
            pred = predictor.predict_neutral(
                team=args.team, opponent=args.opponent,
                as_of_date=args.date, tournament=args.tournament,
            )
            print("  (neutral-ground: averaged over both team orderings)")
        else:
            pred = predictor.predict(
                team=args.team, opponent=args.opponent,
                as_of_date=args.date, tournament=args.tournament,
            )
        print_match_prediction(pred, args.team, args.opponent)

    # ── append_result ─────────────────────────────────────────────────────────
    elif args.mode == "append_result":
        if not args.team or not args.opponent or args.outcome is None:
            parser.error("--team, --opponent, and --outcome required for append_result mode")
        new_result = {
            "team":       args.team,
            "opponent":   args.opponent,
            "date":       args.date,
            "outcome":    args.outcome,
            "tournament": args.tournament,
        }
        state["real_results"].append(new_result)
        _save_state(state)
        print(f"Recorded: {args.team} {'win' if args.outcome==2 else ('draw' if args.outcome==1 else 'loss')} vs {args.opponent} ({args.date})")
        print("Re-running simulation with updated results …")
        probs = run_simulation(
            predictor,
            n_sims=args.n_sims,
            real_results=state["real_results"],
            seed=args.seed,
        )
        state["last_probs"] = probs
        _save_state(state)
        _save_probs(probs, args.tag)
        _save_probs_csv(probs, args.tag)
        print_champion_table(probs)

    # ── show_state ────────────────────────────────────────────────────────────
    elif args.mode == "show_state":
        rr = state.get("real_results", [])
        if not rr:
            print("No real results recorded yet.")
        else:
            print(f"\n{len(rr)} real results recorded:")
            for r in rr:
                outcome_str = {2: "WIN", 1: "DRAW", 0: "LOSS"}.get(r["outcome"], "?")
                print(f"  {r['date']}  {r['team']} {outcome_str} vs {r['opponent']}")


# ---------------------------------------------------------------------------
# Convenience: programmatic entry point (for notebooks / paper code)
# ---------------------------------------------------------------------------

def simulate(
    predictor=None,
    n_sims: int = 10_000,
    real_results: list[dict] | None = None,
    seed: int = 42,
    verbose: bool = True,
) -> dict[str, dict[str, float]]:
    """
    Programmatic entry point. Returns probability dict directly.

    Example
    -------
    >>> from wc2026_simulator import simulate, load_predictor
    >>> pred = load_predictor()
    >>> probs = simulate(pred, n_sims=5000)
    >>> print(probs["Brazil"])
    """
    if predictor is None:
        predictor = load_predictor()
    probs = run_simulation(predictor, n_sims=n_sims,
                           real_results=real_results, seed=seed)
    if verbose:
        print_champion_table(probs)
    return probs


def predict_match(
    team: str,
    opponent: str,
    date: str = "2026-06-15",
    tournament: str = "FIFA World Cup",
    predictor=None,
) -> dict:
    """
    Programmatic single-match prediction.

    Example
    -------
    >>> from wc2026_simulator import predict_match
    >>> pred = predict_match("Brazil", "Argentina", date="2026-07-14")
    >>> print(f"Brazil win: {pred['Win']:.1%}")
    """
    if predictor is None:
        predictor = load_predictor()
    return predictor.predict(team=team, opponent=opponent,
                             as_of_date=date, tournament=tournament)


if __name__ == "__main__":
    main()