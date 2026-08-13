"""
utils.py — Shared utilities for the global Gaussian HMM pipeline.

Imported by both evaluate_global.py and predictor_global.py to avoid
circular imports.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from model.gaussian_hmm.hmm_global import TOURNAMENT_WEIGHTS, FEATURE_NAMES

# ---------------------------------------------------------------------------
# Elo constants + helpers
# ---------------------------------------------------------------------------
ELO_K     = 30
ELO_SCALE = 400


def seed_live_elo(df: "pd.DataFrame") -> tuple[dict[str, float], float]:
    """
    Build {team: most-recent known Elo} from a history dataframe.

    Reads BOTH the `team_elo` and `opponent_elo` columns. This matters: the
    dataset only keeps rows where `team` is a WC 2026 participant, so a
    non-participant (Italy, Poland, Serbia, Denmark, Wales, Hungary, Ukraine …)
    appears solely as `opponent` and has no `team_elo` row at all. Seeding from
    `team_elo` alone left those teams unrated, and they silently fell back to the
    global mean — which affected 30/65 Euro 2024 and 11/57 WC 2022 test matches
    and produced simulated-vs-published Elo gaps of up to 664 points.

    Their rating is present all along in `opponent_elo`; it just was not read.

    Returns (live_elo, default_elo). default_elo is the mean of known ratings and
    is now only reached by a team that appears nowhere in the history.
    """
    obs = []
    if {"date", "team", "team_elo"}.issubset(df.columns):
        obs.append(
            df[["date", "team", "team_elo"]]
            .rename(columns={"team": "_t", "team_elo": "_r"})
        )
    if {"date", "opponent", "opponent_elo"}.issubset(df.columns):
        obs.append(
            df[["date", "opponent", "opponent_elo"]]
            .rename(columns={"opponent": "_t", "opponent_elo": "_r"})
        )
    if not obs:
        return {}, 1500.0

    stacked = pd.concat(obs, ignore_index=True).dropna(subset=["_r"])
    if stacked.empty:
        return {}, 1500.0

    live_elo = (
        stacked.sort_values("date")
        .groupby("_t")["_r"]
        .last()
        .astype(float)
        .to_dict()
    )
    default_elo = float(np.mean(list(live_elo.values())))
    return live_elo, default_elo

def _elo_expected(r_a: float, r_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / ELO_SCALE))

def _elo_update(r_a: float, r_b: float, score_a: float) -> tuple[float, float]:
    exp_a   = _elo_expected(r_a, r_b)
    new_r_a = r_a + ELO_K * (score_a - exp_a)
    new_r_b = r_b + ELO_K * ((1 - score_a) - (1 - exp_a))
    return float(new_r_a), float(new_r_b)

def _outcome_to_score(outcome: int) -> float:
    return {2: 1.0, 1: 0.5, 0: 0.0}[outcome]

# ---------------------------------------------------------------------------
# Online form tracking
# ---------------------------------------------------------------------------
# Test-time form used to be frozen: the form features for every match in a test
# window were the ones precomputed in the CSV, and the "append to history" step
# copied the *team's* feature row onto the opponent, so a team's regime sequence
# was polluted with its opponents' form. Both are fixed here by recomputing the
# features from raw results (goal difference + win + opponent rating) after every
# match, for both teams, using the same definitions as data/raw/data_filter.py.

STRONG_ELO = 1500.0   # data_filter.py: global_median_elo


def form_columns(win, goal_diff, opp_elo) -> pd.DataFrame:
    """
    Recompute the pre-match form features for one team's match sequence.

    Mirrors data/raw/data_filter.py expression-for-expression, so the output
    reproduces the CSV columns (verified: median absolute difference 0.0 across
    all rows past each team's warm-up; the residual on the first few rows comes
    from the trailing dropna removing matches that still counted as history).

    Every column is `.shift()`ed, so row i depends only on matches 0..i-1 — the
    value at row i is legitimately available before match i kicks off.
    """
    win = pd.Series(np.asarray(win,       dtype=float))
    gd  = pd.Series(np.asarray(goal_diff, dtype=float))
    oe  = pd.Series(np.asarray(opp_elo,   dtype=float))

    rolling_win_rate_5 = win.shift().rolling(5).mean()
    ewa_win_rate  = win.shift().ewm(span=5, min_periods=3).mean()
    ewa_goal_diff = gd .shift().ewm(span=5, min_periods=3).mean()

    # Win rate against top-half opposition, falling back to the overall rate.
    win_vs_strong = win.where(oe >= STRONG_ELO)
    rolling_wvs_5 = (win_vs_strong.shift().rolling(5, min_periods=2).mean()
                     .fillna(rolling_win_rate_5))

    return pd.DataFrame({
        "ewa_win_rate":            ewa_win_rate,
        "ewa_goal_diff":           ewa_goal_diff,
        "rolling_win_vs_strong_5": rolling_wvs_5,
        "rolling_goal_diff_std_5": gd .shift().rolling(5).std(),
        "rolling_win_rate_std_5":  win.shift().rolling(5).std(),
        "ewa_win_rate_momentum":   ewa_win_rate  - ewa_win_rate .shift(5),
        "ewa_goal_diff_momentum":  ewa_goal_diff - ewa_goal_diff.shift(5),
    })[FEATURE_NAMES]


class FormTracker:
    """
    Per-team rolling form, seeded from history and advanced match by match.

    Holds two aligned per-team sequences:
      * raw       — (goal_diff, win, opponent_elo) for each match played
      * feat_hist — the pre-match feature vector that was current for each match

    `feat_hist` is what the HMM consumes: the observation for match i is the
    feature vector describing the team's form going *into* match i.

    Seeding reads both orientations of every historical match, so a team that
    appears in the data only as `opponent` still gets a sequence of its own
    rather than falling through to a uniform state prior.
    """

    def __init__(self, history_df: pd.DataFrame):
        self.raw:       dict[str, dict[str, list]] = {}
        self.feat_hist: dict[str, list]            = {}
        self._seed(history_df)

    # -- seeding ----------------------------------------------------------
    def _seed(self, history_df: pd.DataFrame) -> None:
        df = history_df.sort_values(["date", "team", "opponent"])

        forward = pd.DataFrame({
            "date":      df["date"].to_numpy(),
            "team":      df["team"].to_numpy(),
            "opponent":  df["opponent"].to_numpy(),
            "goal_diff": df["goal_diff"].to_numpy(float),
            "win":       (df["outcome"].to_numpy(int) == 2).astype(float),
            "opp_elo":   df["opponent_elo"].to_numpy(float),
            "from_csv":  True,
        })
        # Opponent's view of the same match: goal difference and result flip,
        # and the rating they faced is the tracked team's rating.
        mirror = pd.DataFrame({
            "date":      df["date"].to_numpy(),
            "team":      df["opponent"].to_numpy(),
            "opponent":  df["team"].to_numpy(),
            "goal_diff": -df["goal_diff"].to_numpy(float),
            "win":       (df["outcome"].to_numpy(int) == 0).astype(float),
            "opp_elo":   df["team_elo"].to_numpy(float),
            "from_csv":  False,
        })

        # from_csv rows sort first, so a match already present in both
        # orientations keeps the original row rather than the synthesised one.
        stacked = (
            pd.concat([forward, mirror], ignore_index=True)
            .sort_values(["from_csv"], ascending=False, kind="stable")
            .drop_duplicates(subset=["date", "team", "opponent"], keep="first")
            .sort_values(["team", "date"], kind="stable")
        )

        csv_feats = None
        if set(FEATURE_NAMES).issubset(history_df.columns):
            csv_feats = (
                df.set_index(["date", "team"])[FEATURE_NAMES]
                .astype(float)
            )
            csv_feats = csv_feats[~csv_feats.index.duplicated(keep="first")]

        for team, grp in stacked.groupby("team", sort=False):
            self.raw[team] = {
                "goal_diff": grp["goal_diff"].tolist(),
                "win":       grp["win"].tolist(),
                "opp_elo":   grp["opp_elo"].tolist(),
            }
            recomputed = form_columns(
                grp["win"], grp["goal_diff"], grp["opp_elo"]
            ).fillna(0.0).to_numpy(float).copy()

            # Prefer the CSV's own values where the row came from the CSV, so
            # the training-time observations the HMM was fitted on are exactly
            # what it saw before; recomputation only fills the mirrored rows the
            # CSV never carried features for.
            if csv_feats is not None:
                for j, (dt, is_csv) in enumerate(
                    zip(grp["date"], grp["from_csv"])
                ):
                    if not is_csv:
                        continue
                    try:
                        vals = csv_feats.loc[(dt, team)].to_numpy(float)
                    except KeyError:
                        continue
                    recomputed[j] = np.nan_to_num(vals, nan=0.0)

            self.feat_hist[team] = [row for row in recomputed]

    # -- reads ------------------------------------------------------------
    def sequence(self, team: str, window: int) -> np.ndarray:
        """Last `window` pre-match feature vectors — the HMM's observation seq."""
        hist = self.feat_hist.get(team)
        if not hist:
            return np.empty((0, len(FEATURE_NAMES)), dtype=float)
        return np.array(hist[-window:], dtype=float)

    def pre_match_features(self, team: str) -> np.ndarray:
        """Form vector for a team's *next* match, from everything played so far."""
        rec = self.raw.get(team)
        if rec is None or len(rec["win"]) == 0:
            return np.zeros(len(FEATURE_NAMES), dtype=float)
        # A trailing placeholder gives form_columns a row to shift into; its own
        # values are never read, since every expression is shifted.
        cols = form_columns(
            rec["win"]       + [np.nan],
            rec["goal_diff"] + [np.nan],
            rec["opp_elo"]   + [np.nan],
        )
        return np.nan_to_num(cols.iloc[-1].to_numpy(float), nan=0.0)

    # -- writes -----------------------------------------------------------
    def record(self, team: str, goal_diff: float, win: float,
               opp_elo: float) -> None:
        """Append a played match, advancing the team's form for the next one."""
        self.feat_hist.setdefault(team, []).append(
            self.pre_match_features(team)
        )
        rec = self.raw.setdefault(
            team, {"goal_diff": [], "win": [], "opp_elo": []}
        )
        rec["goal_diff"].append(float(goal_diff))
        rec["win"].append(float(win))
        rec["opp_elo"].append(float(opp_elo))

    def record_match(self, team: str, opponent: str, goal_diff: float,
                     outcome: int, team_elo: float, opp_elo: float) -> None:
        """Record one match from both teams' perspectives."""
        self.record(team,     goal_diff, 1.0 if outcome == 2 else 0.0, opp_elo)
        self.record(opponent, -goal_diff, 1.0 if outcome == 0 else 0.0, team_elo)


# ---------------------------------------------------------------------------
# Tournament stage helpers
# ---------------------------------------------------------------------------
_KNOCKOUT_KEYWORDS = [
    "final", "semi", "quarter", "round of", "last 16",
    "knockout", "elimination", "third place",
]

def _is_knockout(tournament_str: str) -> int:
    if not isinstance(tournament_str, str):
        return 0
    t = tournament_str.lower()
    return int(any(k in t for k in _KNOCKOUT_KEYWORDS))

def _tournament_weight_val(tournament_str: str) -> float:
    if not isinstance(tournament_str, str):
        return 1.0
    for key, w in TOURNAMENT_WEIGHTS.items():
        if key.lower() in tournament_str.lower():
            return w
    return 1.0

# ---------------------------------------------------------------------------
# Draw propensity helpers
# ---------------------------------------------------------------------------

def _draw_features(probs_3way: np.ndarray,
                   elo_diffs:  np.ndarray,
                   entropy_a:  np.ndarray,
                   entropy_b:  np.ndarray,
                   is_knockout: np.ndarray) -> np.ndarray:
    elo_closeness = 1.0 / (1.0 + np.abs(elo_diffs) / 100.0)
    return np.column_stack([
        probs_3way[:, 1],
        elo_closeness,
        entropy_a,
        entropy_b,
        entropy_a + entropy_b,
        is_knockout.astype(float),
    ])

def _train_draw_model(X_draw_feats: np.ndarray,
                      outcomes:     np.ndarray,
                      random_seed:  int = 42):
    """Train a binary logistic classifier: draw (1) vs no-draw (0)."""
    from sklearn.linear_model import LogisticRegression
    y_draw = (outcomes == 1).astype(int)
    clf    = LogisticRegression(max_iter=1000, C=0.5, random_state=random_seed)
    clf.fit(X_draw_feats, y_draw)
    return clf


def _blend_draw_probs(probs_3way: np.ndarray,
                      draw_probs: np.ndarray,
                      alpha:      float = 0.3) -> np.ndarray:
    blended  = probs_3way.copy()
    new_draw = (1 - alpha) * probs_3way[:, 1] + alpha * draw_probs
    wl_mass  = probs_3way[:, 0] + probs_3way[:, 2]
    scale    = np.where(wl_mass > 1e-9, (1.0 - new_draw) / wl_mass, 0.5)
    blended[:, 0] = probs_3way[:, 0] * scale
    blended[:, 1] = new_draw
    blended[:, 2] = probs_3way[:, 2] * scale
    row_sums = blended.sum(axis=1, keepdims=True)
    return blended / np.where(row_sums > 0, row_sums, 1.0)