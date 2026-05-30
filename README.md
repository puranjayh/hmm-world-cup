# HMM World Cup 2026 Prediction System

A research-oriented football prediction framework using Hidden Markov Models to forecast match outcomes and tournament progression for the 2026 FIFA World Cup.

---

## Overview

Traditional football prediction systems rely on static rankings or short-term form windows. This project treats each team as a dynamic system transitioning between latent performance states - learned entirely from historical match data - and updates predictions after every observed result.

The framework combines:
- **Hidden Markov Models (HMMs)** for latent state inference
- **Bayesian updating** for dynamic in-tournament recalibration
- **Elo ratings** as a structural prior
- **Rolling form metrics** for short-term momentum signals
- **Expected Goals (xG)** for shot-quality-adjusted performance measurement

---

## Research Premise

The HMM assumes teams occupy one of several unobservable performance states (e.g. Dominant, Stable, Declining, Unstable) at any point in time. Match results are observable emissions from these hidden states. After each match:

1. The observed result (Win / Draw / Loss) updates the posterior probability over hidden states via Bayes' rule.
2. Transition matrices encode how likely a team is to shift states between matches.
3. Emission matrices encode how likely each state is to produce a given result.

This allows the system to adapt mid-tournament - a team that wins its first two group stage matches has its state distribution updated before the third prediction is made.

---

## Methodology

### 1. Data Collection

International football match data from 2008 onwards, covering all major national teams across tournament and friendly fixtures.

Sources:
- Kaggle International Football Results
- Elo Ratings Dataset
- StatsBomb Open Data (for xG)

### 2. Feature Engineering

Per-match features engineered for the prediction pipeline:

| Feature | Description |
|---|---|
| `goals_for` / `goals_against` | Raw scoreline |
| `goal_diff` | Signed goal difference |
| `result` | Win / Draw / Loss |
| `team_elo` / `opponent_elo` | Elo ratings at match date |
| `elo_diff` | Signed Elo differential |
| `rolling_win_rate_5` | Win rate over last 5 matches |
| `rolling_goal_diff_5` | Average goal difference over last 5 matches |
| `tournament_weight` | Match importance proxy (5.0 = WC, 1.0 = Friendly) |
| `neutral` | Neutral venue flag |
| `xg_for` / `xg_against` | Expected goals (where available) |

### 3. Hidden Markov Model

Each team's match history is modelled as an observation sequence. The HMM learns:
- **Transition matrix** - probability of moving between hidden states
- **Emission matrix** - probability of each result given each hidden state
- **Initial state distribution** - starting state probabilities

Hidden states (example):
- `Dominant` - consistently winning, strong form
- `Stable` - competitive, balanced results
- `Declining` - deteriorating form, inconsistent results
- `Unstable` - volatile, unpredictable outcomes

The forward algorithm is used at inference time to compute the current state distribution from a team's match history up to (but not including) the prediction date - enforcing strict no-leakage guarantees.

### 4. Bayesian Updating

After each observed match result, the model recalculates the posterior probability distribution over hidden states. This makes the system dynamic:
- Pre-tournament predictions are based on historical form only
- Group stage predictions update as results come in
- Knockout predictions incorporate all prior tournament results

### 5. Benchmarking

The model is evaluated against Random Forest and XGBoost baselines across three held-out evaluation sets:

| Evaluation Set | Train Period | Purpose |
|---|---|---|
| 2018 FIFA World Cup | 2008–2017 | Tournament holdout 1 |
| 2022 FIFA World Cup | 2008–2021 | Tournament holdout 2 |
| All 2024 internationals | 2008–2023 | Statistical robustness (N ≈ 600–800) |

Metrics reported for all models:
- **Log-loss** - primary metric; rewards calibrated probability outputs
- **Brier score** - mean squared error of predicted probabilities
- **Accuracy** - proportion of correct winner predictions
- **RPS (Ranked Probability Score)** - standard football forecasting metric; accounts for ordinal nature of W/D/L

Both static (pre-tournament) and dynamic (updated after each match) evaluation modes are run and compared.

---

## Dataset Structure

```
data/
│
├── raw/
│   ├── all_matches.csv          # Full international match history
│   ├── eloratings.csv           # Historical Elo ratings per team
│   ├── match_ids.csv            # StatsBomb match identifiers
│   ├── matches/                 # StatsBomb match-level JSON
│   └── events/                  # StatsBomb event-level JSON (for xG)
│
└── processed/
    ├── filtered_matches.csv     # Cleaned match data with rolling features + Elo
    ├── xg_dataset.csv           # xG metrics merged from StatsBomb
    └── final_dataset.csv        # Full feature matrix used for training
```

---

## Project Structure

```
model/
├── hmm_team.py          # Per-team HMM: training, forward algorithm, state inference
├── predictor.py         # Match outcome predictor with strict date-gating
├── evaluate.py          # Benchmarking pipeline (HMM, RF, XGBoost) across all eval sets
├── config.py            # Training cutoffs, state counts, feature lists
└── tests/
    └── test_sanity.py   # Property-based sanity checks (no CSV required)

data/
├── raw/
└── processed/
```

---

## Running the Benchmarks

```bash
python -m model.evaluate
```

This runs all three evaluation sets (2018 WC, 2022 WC, 2024 all matches) for HMM, RF, and XGBoost, and outputs:
- `metrics_all.json` - full results table across all models and splits
- Calibration plots per evaluation run

---

## Key Design Decisions

**No feature engineering for the HMM.** The HMM learns form states directly from raw Win/Draw/Loss sequences. RF and XGBoost use hand-crafted rolling features (`elo_diff`, `rolling_win_rate_5`, `rolling_goal_diff_5`, `tournament_weight`). If they perform comparably, this demonstrates the HMM extracts equivalent signal without manual feature design.

**Strict temporal separation.** All models are retrained from scratch per evaluation cutoff. The predictor's date-gating ensures no match result is used before its date - both during training and at inference time.

**Dynamic updating is the default.** Predictions for later matches in a tournament use earlier match results as observations. This mirrors real deployment conditions and is consistent with best practice in football forecasting literature.
