"""
live_baseline.py — Contemporaneous Elo baseline for the 2026 World Cup forecast log.

The committed forecast log (Live Test/Predictions.csv) records what the deployed
GHMM predicted before each round, but carries no baseline on those same fixtures,
so the headline accuracies are not directly interpretable. This script supplies one.

A baseline does NOT need to have been committed in advance — only the model's
predictions do. So we may fairly reconstruct, now, what a pure-Elo forecaster
would have said on the same fixtures.

Elo source — IMPORTANT:
  We use the `team_elo` column of data/raw/filtered_matches.csv, i.e. the *same*
  rating the GHMM head is trained and evaluated on. This is deliberate: the point
  of the baseline is a like-for-like comparison against the model's own rating.
  data/raw/eloratings.csv is a DIFFERENT rating series that does not track this
  dataset's results and must NOT be used here — doing so produces a spuriously
  weak Elo baseline. Each team's snapshot is its last pre-tournament team_elo
  (the file ends at the pre-WC friendlies, so "last" is already the static
  pre-tournament value). Ratings are not updated as the bracket progresses; a
  dynamically-updated Elo would move by only a few points over ~100 neutral
  knockout matches and does not change the conclusion.

Head: a logistic map elo_diff -> P(W/D/L) trained on all pre-tournament unique
matches in filtered_matches.csv — identical recipe to evaluate_global._run_elo.

Scoring matches the forecast log's own convention: three-way uses argmax over
W/D/L; "decisive" collapses the draw and asks whether the forecaster picked the
correct side, scored only on matches whose actual result was decisive.

Three teams in the log (Bosnia, Cabo Verde, Curaçao) have no pre-WC match in
filtered_matches and therefore no model-consistent rating; their fixtures are
excluded and the effective n is reported.

Run:  python scripts/live_baseline.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "Live Test" / "Predictions.csv"
HIST = ROOT / "data" / "raw" / "filtered_matches.csv"
TOURNAMENT_START = "2026-06-12"

# Normalise name quirks in the forecast log into the model's rating namespace:
# a "Frrance" typo, and "Turkiye" for the "Turkey" of filtered_matches.
NAME = {"Frrance": "France", "Turkiye": "Turkey"}


def _norm(x: str) -> str:
    return NAME.get(x, x)


def _load() -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """Static pre-WC team_elo snapshot + (elo_diff, outcome) training rows."""
    snapshot: dict[str, tuple[str, float]] = {}
    X, y = [], []
    with open(HIST) as f:
        for r in csv.DictReader(f):
            team = r["team"].strip()
            try:
                e = float(r["team_elo"])
            except (ValueError, TypeError):
                e = None
            if e is not None and (team not in snapshot or r["date"] > snapshot[team][0]):
                snapshot[team] = (r["date"], e)
            if r["date"] < TOURNAMENT_START and r["team"] < r["opponent"]:
                try:
                    X.append([float(r["elo_diff"])])
                    y.append(int(r["outcome"]))
                except (ValueError, TypeError):
                    continue
    return {t: v[1] for t, v in snapshot.items()}, np.array(X), np.array(y)


def _mcnemar(model_hits: list[int], base_hits: list[int]):
    b = sum(1 for m, e in zip(model_hits, base_hits) if m and not e)
    c = sum(1 for m, e in zip(model_hits, base_hits) if e and not m)
    p = binomtest(b, b + c, 0.5).pvalue if (b + c) else 1.0
    return b, c, p


def main() -> None:
    elo, X, y = _load()
    head = LogisticRegression(max_iter=1000).fit(X, y)
    cls = list(head.classes_)

    m3, e3, mD, eD, favD = [], [], [], [], []
    skipped: set[str] = set()

    with open(PRED) as f:
        for r in csv.DictReader(f):
            team, opp, res = _norm(r["Team"].strip()), _norm(r["Opponent"].strip()), _norm(r["Result"].strip())
            if not team or not opp:
                continue
            if team not in elo or opp not in elo:
                skipped.update(t for t in (team, opp) if t not in elo)
                continue

            raw = head.predict_proba([[elo[team] - elo[opp]]])[0]
            prob = {int(c): raw[k] for k, c in enumerate(cls)}
            p_win, p_draw, p_loss = prob[2], prob[1], prob[0]

            if r["Accurate"].strip() != "":
                m3.append(int(r["Accurate"]))
                pred = max([(p_win, "W"), (p_draw, "D"), (p_loss, "L")])[1]
                actual = "D" if res == "Draw" else ("W" if res == team else "L")
                e3.append(int(pred == actual))

            if res != "Draw" and res in (team, opp) and r["Accurate (W/L)"].strip() != "":
                mD.append(int(r["Accurate (W/L)"]))
                actual = "W" if res == team else "L"
                eD.append(int(("W" if p_win >= p_loss else "L") == actual))
                fav = team if elo[team] > elo[opp] else opp
                favD.append(int(fav == res))

    def line(name, hits):
        return f"{name:16s} {sum(hits):>3d}/{len(hits):<3d} = {sum(hits)/len(hits):.3f}"

    b3, c3, p3 = _mcnemar(m3, e3)
    bD, cD, pD = _mcnemar(mD, eD)

    print("Same 2026 World Cup fixtures — GHMM (deployed) vs contemporaneous static Elo")
    print("Elo source: filtered_matches.csv team_elo (the model's own rating series)\n")
    if skipped:
        print(f"Excluded (no pre-WC rating in filtered_matches): {sorted(skipped)}\n")
    print("Three-way (W/D/L):")
    print("  " + line("GHMM", m3))
    print("  " + line("Elo (logistic)", e3))
    print(f"  McNemar: model-only={b3}, elo-only={c3}, p={p3:.4f}\n")
    print("Decisive matches (pick the winner):")
    print("  " + line("GHMM", mD))
    print("  " + line("Elo (logistic)", eD))
    print("  " + line("Elo (pick favorite)", favD))
    print(f"  McNemar (GHMM vs Elo-logistic): model-only={bD}, elo-only={cD}, p={pD:.4f}")


if __name__ == "__main__":
    main()
