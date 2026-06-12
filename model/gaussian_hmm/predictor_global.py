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

WINDOW = 3   # last N matches used for state inference — empirically best


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
                "dates":    grp["date"].to_numpy(),
                "features": grp[FEATURE_NAMES].fillna(0).to_numpy(dtype=float),
            }

    def update_history(self, df: pd.DataFrame) -> None:
        """Full rebuild from a new dataframe (e.g. after tournament stage ends)."""
        self._build_index(df)

    def append_result(self,
                      team:     str,
                      opponent: str,
                      date,
                      outcome:  int,
                      row_dict: dict | None = None) -> None:
        """
        Incrementally add one completed result for both teams and update
        live Elo ratings.  outcome: 2=win, 1=draw, 0=loss  (for `team`).
        """
        for t, res, opp in [
            (team,     outcome,     opponent),
            (opponent, 2 - outcome, team),
        ]:
            rec = self._per_team.setdefault(
                t,
                {"dates": np.array([], dtype="datetime64"),
                 "features": np.empty((0, len(FEATURE_NAMES)))}
            )
            new_date = np.array([np.datetime64(pd.Timestamp(date))], dtype="datetime64")
            base     = row_dict or {}
            new_feat = np.array([[float(base.get(f, 0) or 0) for f in FEATURE_NAMES]])
            rec["dates"]    = np.concatenate([rec["dates"],    new_date])
            rec["features"] = np.vstack(     [rec["features"], new_feat])

        # Update Elo
        r_team = self.live_elo.get(team,     self._default_elo)
        r_opp  = self.live_elo.get(opponent, self._default_elo)
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
            "is_knockout":     int(is_ko),
            "tournament_weight": float(tourn_w),
            # Confidence gating helper
            "max_prob":        float(final_probs.max()),
        }