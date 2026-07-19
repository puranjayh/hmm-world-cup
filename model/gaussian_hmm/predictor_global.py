"""
predictor_global.py — Prediction head for the global Gaussian HMM.

Improvements over v1:
  1. Dynamic Elo updating  — live_elo dict updated after each result.
  2. Tournament stage      — is_knockout + tournament_weight in feature vec.
  3. Draw propensity blend — secondary binary classifier blended in.
  4. Confidence gating     — predict() returns max_prob so callers can gate.

Feature vector layout (N=7 → 60 features):
    outer(p_A, p_B).ravel()     (N²=49)
    max_p_A, max_p_B            (2)
    entropy_A, entropy_B        (2)
    elo_diff                    (1)
    elo_diff * max_p_A          (1)
    elo_diff * max_p_B          (1)
    is_knockout                 (1)  NEW
    tournament_weight           (1)  NEW
"""

import numpy as np
import pandas as pd

from model.gaussian_hmm.hmm_global import GlobalGaussianHMM, FEATURE_NAMES
from model.gaussian_hmm.utils import (
    _is_knockout,
    _tournament_weight_val,
    _elo_update,
    _outcome_to_score,
    _draw_features,
    _blend_draw_probs,
    ELO_K,
    ELO_SCALE,
)

# Keep in sync with evaluate_global.WINDOW — see the note there on why 3.
WINDOW = 3

# NOTE — known inconsistency with the benchmark path.
# This module feeds the head a *simulated* live_elo differential (plus the
# form adjustment below), while the head is trained in evaluate_global.py on the
# *published* elo_diff column. That train/test mismatch costs roughly 0.04 nats
# and is fixed on the benchmark side but deliberately NOT fixed here: this is the
# code path that generated the committed 2026 World Cup predictions, and altering
# it would break the correspondence between the repository and that prospective
# record. Fix this before using the module for any new live forecasting.

GLOBAL_MED_ELO = 1500.0   # median Elo threshold for "win vs strong", matches data_filter.py

# Form-quality adjustment constants.
# FORM_ELO_SCALE: virtual Elo points added per unit of quality-weighted EWA
#   goal-diff differential. Bigger = more reactive to recent results.
# FORM_ELO_CAP:  maximum absolute adjustment (prevents one blowout dominating).
FORM_ELO_SCALE: float = 40.0
FORM_ELO_CAP:   float = 200.0


class GlobalPredictor:

    def __init__(self,
                 global_hmm:  GlobalGaussianHMM,
                 head,
                 history_df:  pd.DataFrame,
                 draw_model=None,
                 draw_alpha:  float = 0.3):
        self.hmm         = global_hmm
        self.head        = head
        self.draw_model  = draw_model   # optional draw propensity classifier
        self.draw_alpha  = draw_alpha
        self._build_index(history_df)

        # Live Elo — initialised from the most recent rating per team in history
        self.live_elo: dict[str, float] = (
            history_df.sort_values("date")
            .groupby("team")["team_elo"]
            .last()
            .to_dict()
        )
        self._default_elo = float(
            np.mean(list(self.live_elo.values())) if self.live_elo else 1500.0
        )

    # ------------------------------------------------------------------
    # History index
    # ------------------------------------------------------------------

    def _build_index(self, df: pd.DataFrame) -> None:
        df = df.sort_values("date").reset_index(drop=True)
        self._per_team: dict = {}
        for team, grp in df.groupby("team", sort=False):
            self._per_team[team] = {
                "dates":       grp["date"].to_numpy(),
                "features":    grp[FEATURE_NAMES].fillna(0).to_numpy(dtype=float),
                "raw_gd":      grp["goal_diff"].to_numpy(dtype=float)    if "goal_diff"    in grp else np.array([], dtype=float),
                "raw_win":     grp["win"].to_numpy(dtype=float)          if "win"          in grp else np.array([], dtype=float),
                "raw_opp_elo": grp["opponent_elo"].to_numpy(dtype=float) if "opponent_elo" in grp else np.array([], dtype=float),
            }

    def update_history(self, df: pd.DataFrame) -> None:
        """Full rebuild from a new dataframe (e.g. after tournament stage ends)."""
        self._build_index(df)

    def _compute_live_row(self, rec: dict,
                          span: int = 5, min_periods: int = 3) -> np.ndarray:
        """
        Compute a feature row from raw history.

        span / min_periods control EWA behaviour:
          - span=5, min_periods=3 (defaults): training-compatible, used for
            the pre-match row stored on append_result.
          - span=3, min_periods=1: fast-decay, used for the post-match row so
            a single tournament result meaningfully shifts the form signal.
        """
        def _last(s: pd.Series, fallback: float) -> float:
            v = s.iloc[-1] if len(s) > 0 else np.nan
            return fallback if pd.isna(v) else float(v)

        s_gd  = pd.Series(rec["raw_gd"],      dtype=float)
        s_win = pd.Series(rec["raw_win"],      dtype=float)
        s_opp = pd.Series(rec["raw_opp_elo"], dtype=float)

        ewa_gd  = s_gd.ewm(span=span, min_periods=min_periods).mean()
        ewa_win = s_win.ewm(span=span, min_periods=min_periods).mean()

        new_ewa_gd  = _last(ewa_gd,  0.0)
        new_ewa_win = _last(ewa_win, 0.5)

        gd_std  = _last(s_gd.rolling(5).std(),  0.0)
        win_std = _last(s_win.rolling(5).std(), 0.0)

        s_wvs = pd.Series(np.where(s_opp >= GLOBAL_MED_ELO, s_win, np.nan), dtype=float)
        win_vs_strong = _last(s_wvs.rolling(5, min_periods=2).mean(), new_ewa_win)

        ewa_win_mom = _last(ewa_win.diff(5), 0.0)
        ewa_gd_mom  = _last(ewa_gd.diff(5),  0.0)

        return np.array([
            new_ewa_win, new_ewa_gd, win_vs_strong,
            gd_std, win_std, ewa_win_mom, ewa_gd_mom,
        ], dtype=float)

    def _quality_weighted_form(self, team: str) -> float:
        """
        EWA goal-diff weighted by opponent quality.

        Each match's goal diff is scaled by (opponent_elo / GLOBAL_MED_ELO),
        so a 5-1 win vs a 1750-Elo opponent counts more than 5-1 vs a 1200-Elo
        opponent.  Clipped to [0.5, 2.0] to avoid extreme distortions.
        The EWA (span=5, min_periods=1) then smooths across recent history so
        that the most recent match is weighted at ~33% and older ones decay.
        """
        rec = self._per_team.get(team)
        if rec is None:
            return 0.0
        n = min(len(rec["raw_gd"]), len(rec["raw_opp_elo"]))
        if n == 0:
            return 0.0
        quality = np.clip(
            rec["raw_opp_elo"][:n] / GLOBAL_MED_ELO, 0.5, 2.0
        )
        weighted_gd = rec["raw_gd"][:n] * quality
        s   = pd.Series(weighted_gd, dtype=float)
        val = s.ewm(span=5, min_periods=1).mean().iloc[-1]
        return float(val) if pd.notna(val) else 0.0

    def append_result(self,
                      team:          str,
                      opponent:      str,
                      date,
                      outcome:       int,
                      goals_for:     int | None = None,
                      goals_against: int | None = None,
                      opponent_elo:  float | None = None,
                      update_elo:    bool = False,
                      row_dict:      dict | None = None) -> None:
        """
        Incrementally add one completed result for both teams.

        Pass goals_for / goals_against to have goal difference reflected in the
        HMM features (ewa_goal_diff, rolling_goal_diff_std_5, momentum).
        Without them the feature row falls back to row_dict or zeros.

        update_elo=False (default) leaves live Elo unchanged — useful during
        a tournament where you want form to update but not ratings.
        outcome: 2=win, 1=draw, 0=loss  (from `team`'s perspective).
        """
        unknown = [t for t in (team, opponent) if t not in self._per_team]
        if unknown:
            known = self.known_teams()
            raise ValueError(
                f"Unknown team(s): {unknown}. "
                f"Valid names: {known}"
            )

        r_team = self.live_elo.get(team,     self._default_elo)
        r_opp  = self.live_elo.get(opponent, self._default_elo)
        opp_elo_for_team = float(opponent_elo) if opponent_elo is not None else r_opp

        empty_rec = lambda: {
            "dates":       np.array([], dtype="datetime64"),
            "features":    np.empty((0, len(FEATURE_NAMES))),
            "raw_gd":      np.array([], dtype=float),
            "raw_win":     np.array([], dtype=float),
            "raw_opp_elo": np.array([], dtype=float),
        }

        entries = [
            (team,     outcome,     goals_for,     goals_against, opp_elo_for_team),
            (opponent, 2 - outcome, goals_against, goals_for,     r_team),
        ]

        for t, res, gf, ga, opp_elo_val in entries:
            rec      = self._per_team.setdefault(t, empty_rec())
            new_date = np.array([np.datetime64(pd.Timestamp(date))], dtype="datetime64")

            if gf is not None and ga is not None:
                goal_diff = float(gf - ga)
                win       = float(gf > ga)
                # Compute features from history BEFORE appending — no leakage
                new_feat = self._compute_live_row(rec).reshape(1, -1)
                rec["raw_gd"]      = np.append(rec["raw_gd"],      goal_diff)
                rec["raw_win"]     = np.append(rec["raw_win"],      win)
                rec["raw_opp_elo"] = np.append(rec["raw_opp_elo"], opp_elo_val)
                # Post-match features incorporate the result just recorded.
                # span=5 keeps values within the training distribution so the
                # HMM stays well-calibrated; min_periods=1 ensures teams with
                # limited history still get a meaningful form update.
                post_feat = self._compute_live_row(rec, span=5, min_periods=1).reshape(1, -1)
            else:
                base      = row_dict or {}
                new_feat  = np.array([[float(base.get(f, 0) or 0) for f in FEATURE_NAMES]])
                post_feat = None

            rec["dates"]    = np.concatenate([rec["dates"],    new_date])
            rec["features"] = np.vstack(     [rec["features"], new_feat])
            if post_feat is not None:
                rec["dates"]    = np.concatenate([rec["dates"],    new_date])
                rec["features"] = np.vstack(     [rec["features"], post_feat])

        if update_elo:
            score_a = _outcome_to_score(outcome)
            new_r_team, new_r_opp = _elo_update(r_team, r_opp, score_a)
            self.live_elo[team]     = new_r_team
            self.live_elo[opponent] = new_r_opp

    # ------------------------------------------------------------------
    # Posterior
    # ------------------------------------------------------------------

    def _posterior(self, team: str, as_of_date) -> np.ndarray:
        N   = self.hmm.n_states
        rec = self._per_team.get(team)
        if rec is None:
            unif = np.full(N, 1.0 / N)
            return np.concatenate([unif, [1.0 / N, np.log(N)]])
        idx   = np.searchsorted(
            rec["dates"], np.datetime64(pd.Timestamp(as_of_date)), side="left"
        )
        feats = rec["features"][max(0, idx - WINDOW): idx]
        return self.hmm.posterior_features(feats)

    # ------------------------------------------------------------------
    # Feature vector
    # ------------------------------------------------------------------

    def _build_feature_vec(self,
                           pf_team:  np.ndarray,
                           pf_opp:   np.ndarray,
                           elo_diff: float,
                           is_ko:    int   = 0,
                           tourn_w:  float = 1.0) -> np.ndarray:
        N       = self.hmm.n_states
        p_A     = pf_team[:N];  max_p_A = pf_team[N];  ent_A = pf_team[N + 1]
        p_B     = pf_opp[:N];   max_p_B = pf_opp[N];   ent_B = pf_opp[N + 1]
        return np.concatenate([
            np.outer(p_A, p_B).ravel(),
            [max_p_A, max_p_B],
            [ent_A,   ent_B],
            [elo_diff],
            [elo_diff * max_p_A],
            [elo_diff * max_p_B],
            [float(is_ko)],
            [float(tourn_w)],
        ])

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict_neutral(self,
                        team:       str,
                        opponent:   str,
                        as_of_date,
                        tournament: str = "",
                        elo_diff:   float | None = None) -> dict:
        """
        Symmetric neutral-ground prediction: averages predict(A,B) and predict(B,A)
        so the result is identical regardless of which team is listed first.
        Win/Draw/Loss are all from `team`'s perspective.
        """
        ab = self.predict(team, opponent, as_of_date, tournament, elo_diff)
        ba = self.predict(opponent, team, as_of_date, tournament,
                          -elo_diff if elo_diff is not None else None)
        def _avg(k_ab, k_ba):
            return (ab[k_ab] + ba[k_ba]) / 2.0
        return {
            "Win":              _avg("Win",  "Loss"),
            "Draw":             _avg("Draw", "Draw"),
            "Loss":             _avg("Loss", "Win"),
            "Win_raw":          _avg("Win_raw",  "Loss_raw"),
            "Draw_raw":         _avg("Draw_raw", "Draw_raw"),
            "Loss_raw":         _avg("Loss_raw", "Win_raw"),
            "draw_prob_model":  None,
            "state_team":       ab["state_team"],
            "state_opp":        ab["state_opp"],
            "conf_team":        ab["conf_team"],
            "conf_opp":         ab["conf_opp"],
            "entropy_team":     ab["entropy_team"],
            "entropy_opp":      ab["entropy_opp"],
            "elo_diff":         ab["elo_diff"],
            "form_adj":         ab.get("form_adj", 0),
            "is_knockout":      ab["is_knockout"],
            "tournament_weight": ab["tournament_weight"],
            "max_prob":         max(_avg("Win", "Loss"), _avg("Draw", "Draw"),
                                    _avg("Loss", "Win")),
        }

    def known_teams(self) -> list[str]:
        """Return sorted list of teams with match history."""
        return sorted(self._per_team.keys())

    def predict(self,
                team:        str,
                opponent:    str,
                as_of_date,
                tournament:  str   = "",
                elo_diff:    float | None = None) -> dict:
        """
        Returns a dict with:
            Win / Draw / Loss      — final (blended) probabilities
            Win_raw / Draw_raw / Loss_raw — pre-blend 3-way head probs
            draw_prob_model        — draw propensity model output
            state_team / state_opp — predictive HMM state distributions
            conf_team / conf_opp   — HMM confidence (max_p)
            entropy_team / entropy_opp
            elo_diff               — dynamic Elo diff used
            is_knockout            — 1 if knockout match
            max_prob               — max(Win, Draw, Loss) for confidence gating

        Team ordering is normalised internally to match training convention
        (alphabetically smaller team = team A in the feature vector), so
        Win/Draw/Loss are always returned from `team`'s perspective regardless
        of which name comes first alphabetically.
        """
        unknown = [t for t in (team, opponent) if t not in self._per_team]
        if unknown:
            raise ValueError(
                f"Unknown team(s): {unknown}. "
                f"Call predictor.known_teams() to see valid team names."
            )

        # Normalise ordering to match training convention (alphabetical).
        # If the caller passes (B, A) where B > A, swap internally and flip
        # Win/Loss in the returned dict so callers always get `team`'s view.
        swapped = team > opponent
        if swapped:
            team, opponent = opponent, team
            if elo_diff is not None:
                elo_diff = -elo_diff

        # Dynamic Elo
        r_team = self.live_elo.get(team,     self._default_elo)
        r_opp  = self.live_elo.get(opponent, self._default_elo)
        if elo_diff is None:
            elo_diff = r_team - r_opp
        else:
            elo_diff = float(elo_diff)

        # Quality-weighted form adjustment.
        # Reads each team's quality-weighted EWA goal diff (opponent_elo-scaled)
        # and converts the differential into virtual Elo points added to elo_diff.
        # live_elo is NOT modified — this only adjusts the feature passed to the head.
        form_a   = self._quality_weighted_form(team)
        form_b   = self._quality_weighted_form(opponent)
        form_adj = float(np.clip(
            (form_a - form_b) * FORM_ELO_SCALE, -FORM_ELO_CAP, FORM_ELO_CAP
        ))
        elo_diff += form_adj

        is_ko  = _is_knockout(tournament)
        tourn_w = _tournament_weight_val(tournament)

        pf_team = self._posterior(team,     as_of_date)
        pf_opp  = self._posterior(opponent, as_of_date)

        fv  = self._build_feature_vec(pf_team, pf_opp, elo_diff, is_ko, tourn_w)
        fv  = np.nan_to_num(fv, nan=0.0, posinf=0.0, neginf=0.0).reshape(1, -1)
        raw = self.head.predict_proba(fv)[0]

        probs = np.zeros(3, dtype=float)
        for k, cls in enumerate(self.head.classes_):
            probs[int(cls)] = raw[k]

        # Draw propensity blending
        draw_model_prob = None
        final_probs = probs.copy()
        if self.draw_model is not None:
            N      = self.hmm.n_states
            ent_a  = float(pf_team[N + 1])
            ent_b  = float(pf_opp[N + 1])
            X_draw = _draw_features(
                probs.reshape(1, -1),
                np.array([elo_diff]),
                np.array([ent_a]),
                np.array([ent_b]),
                np.array([is_ko]),
            )
            draw_model_prob = float(self.draw_model.predict_proba(X_draw)[0, 1])
            final_probs = _blend_draw_probs(
                probs.reshape(1, -1),
                np.array([draw_model_prob]),
                alpha=self.draw_alpha,
            )[0]

        N = self.hmm.n_states
        # If we swapped for alphabetical normalisation, flip Win↔Loss back so
        # the caller always receives probabilities from their `team`'s perspective.
        if swapped:
            final_probs = final_probs[[2, 1, 0]]
            probs       = probs[[2, 1, 0]]
            elo_diff    = -elo_diff
            pf_team, pf_opp = pf_opp, pf_team

        return {
            # Final (blended) probabilities
            "Win":             float(final_probs[2]),
            "Draw":            float(final_probs[1]),
            "Loss":            float(final_probs[0]),
            # Raw 3-way head (pre-blend)
            "Win_raw":         float(probs[2]),
            "Draw_raw":        float(probs[1]),
            "Loss_raw":        float(probs[0]),
            "draw_prob_model": draw_model_prob,
            # HMM internals
            "state_team":      pf_team[:N].tolist(),
            "state_opp":       pf_opp[:N].tolist(),
            "conf_team":       float(pf_team[N]),
            "conf_opp":        float(pf_opp[N]),
            "entropy_team":    float(pf_team[N + 1]),
            "entropy_opp":     float(pf_opp[N + 1]),
            # Match context
            "elo_diff":        float(elo_diff),
            "form_adj":        float(form_adj) * (-1 if swapped else 1),
            "is_knockout":     int(is_ko),
            "tournament_weight": float(tourn_w),
            # Confidence gating helper
            "max_prob":        float(final_probs.max()),
        }