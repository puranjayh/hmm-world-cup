# Forecasting International Football with a Global Gaussian HMM

A latent-state model for international football match outcomes, benchmarked against
Elo, Random Forest and XGBoost, and validated by a prospectively committed forecast
log of the 2026 FIFA World Cup.

---

## Result in one line

The model is **statistically indistinguishable from Elo** and **significantly better
than Random Forest and XGBoost**, while exposing an interpretable latent form state
per team.

Held-out tournaments: 2018 WC, 2022 WC, UEFA Euro 2024, Copa América 2024.
Paired bootstrap, 20,000 resamples, on per-match log-loss.

**All matches (n = 227)** — accuracy: GHMM 54.6%, Elo 53.3%, XGBoost 48.0%, RF 44.9%

| vs | Δ log-loss | 95% CI | p | |
|---|---|---|---|---|
| Elo | +0.008 | [−0.007, +0.023] | 0.31 | not significant |
| XGBoost | −0.106 | [−0.171, −0.040] | 0.001 | **significant** |
| Random Forest | −0.151 | [−0.226, −0.073] | <0.001 | **significant** |

**Decisive matches only (n = 166)** — accuracy: GHMM 74.7%, Elo 72.9%, XGBoost 64.5%, RF 57.2%

| vs | Δ log-loss | 95% CI | p | |
|---|---|---|---|---|
| Elo | −0.013 | [−0.030, +0.003] | 0.11 | not significant |
| XGBoost | −0.142 | [−0.224, −0.057] | 0.001 | **significant** |
| Random Forest | −0.192 | [−0.284, −0.099] | <0.001 | **significant** |

Negative Δ means the HMM is better. We do **not** claim the model beats Elo — the
confidence interval does not support it.

---

## Method

Every team's match history is one sequence from a **single global Gaussian HMM**
with 7 latent states, fitted across all teams rather than per team. States are
learned from eight standardised form features:

`ewa_win_rate`, `ewa_goal_diff`, `rolling_win_vs_strong_5`, `rolling_goal_diff_std_5`,
`rolling_win_rate_std_5`, `ewa_win_rate_momentum`, `ewa_goal_diff_momentum`

At prediction time the forward algorithm gives each team a posterior over states
using only matches strictly before the fixture date. For a match, the two posteriors
are combined into a 58-dimensional vector — the 7×7 outer product of the two state
distributions, each side's max-probability and entropy, the Elo differential and its
interactions, plus knockout and tournament-weight flags — and passed to a multinomial
logistic head. A secondary draw-propensity classifier is blended into the output.

**Latent states carry signal beyond Elo.** Ablating the feature blocks: full model
0.999 log-loss, Elo features only 1.012, posterior features only 1.048. The state
block is worth roughly 0.013 nats on top of the rating.

---

## Reproducing the results

```bash
pip install -r requirements.txt

# 1. Build the feature table from raw match + Elo data
cd data/raw && python data_filter.py && cd ../..

# 2. Run all four held-out evaluations (writes the metrics JSON)
python -m model.gaussian_hmm.evaluate_global

# 3. Regenerate results/*.csv from that JSON
python scripts/make_results.py
```

Runs are **bit-identical between invocations**. BLAS threads are pinned to 1 before
numpy is imported, because multithreaded reductions perturbed the logistic head
enough to flip predictions near decision boundaries and move reported accuracy by a
full match. Set `REPRODUCIBLE=0` for the faster, non-deterministic path.

Every row of `results/*.csv` is stamped with the window and the commit that produced
it. The tables are generated, never transcribed.

---

## Layout

```
model/gaussian_hmm/
├── hmm_global.py         # global Gaussian HMM: fit, forward algorithm, posteriors
├── evaluate_global.py    # benchmark vs Elo / RF / XGBoost — entry point
├── predictor_global.py   # live prediction head (see caveat below)
├── wc2026simulator.py    # tournament bracket simulation
├── utils.py              # Elo update, draw blending, tournament weights
└── config.py             # state count, seed, paths

data/raw/
├── data_filter.py        # builds filtered_matches.csv from raw sources
├── update_elo.py         # appends current Elo snapshots from eloratings.net
├── all_matches.csv       # international results, 1872–2026
└── eloratings.csv        # historical Elo ratings

scripts/
├── make_results.py       # regenerates results/*.csv from the metrics JSON
└── window_ablation.py    # sweeps WINDOW, writes results/window_ablation.csv
Live Test/Predictions.csv # prospective 2026 World Cup forecast log
results/                  # generated benchmark tables (main, gating, states, ablation)
```

---

## The 2026 World Cup forecast log

`Live Test/Predictions.csv` holds 104 predictions made during the 2026 World Cup and
committed **before** each round was played — R32 on 28 June, R16 on 4 July, quarters
on 8 July, semis on 13 July, final on 16 July. The git history is the timestamp.

| | correct | n | accuracy | 95% CI |
|---|---|---|---|---|
| Three-way (W/D/L) | 62 | 102 | 60.8% | [51.3%, 70.3%] |
| Decisive matches | 67 | 82 | 81.7% | [73.3%, 90.1%] |

Two things must be stated plainly:

1. **These predictions have never been recomputed and never will be.** Their only
   value is that they preceded the matches. Re-running them with an improved model
   would turn a prospective test into a backtest.
2. **They were produced by a model carrying a known defect** — see the caveat below.
   The figures above are what the deployed system actually achieved, not what a
   corrected model would achieve.

There is no baseline on these same 104 fixtures yet, so the accuracies are not
directly comparable to the benchmark table above.

---

## Known limitations

**`predictor_global.py` retains a train/test Elo mismatch.** It feeds the head a
simulated dynamically-updated Elo plus a form adjustment, while the head is trained
on the published Elo column. This costs about 0.04 nats and is fixed in
`evaluate_global.py`. It is deliberately left in place here because this is the code
path that generated the 2026 predictions; changing it would break the correspondence
between the repository and that record. **Fix it before any new live forecasting.**

**The evaluation is small.** 227 matches across four tournaments. Differences below
roughly 0.02 nats are not resolvable at this sample size.

**The lookback window is not tuned.** Sweeping N over [1, 20]
(`python scripts/window_ablation.py`, see `results/window_ablation.csv`) moves mean
log-loss by under 0.01 nats — within run-to-run scale. `WINDOW = 3` is retained
because it is the value the live 2026 predictions used, not because it is optimal.

**Dataset cutoff is 2008-01-01.** Earlier iterations used a 2002 cutoff; that variant
is a one-line change in `data_filter.py` and produces a larger training set.

---

## Data sources

- International match results — Kaggle international football results
- Elo ratings — [eloratings.net](https://www.eloratings.net)
- StatsBomb open data — match identifiers (xG features are not used in the current model)

Elo ratings are merged with `allow_exact_matches=False`. eloratings.net dates a
rating to the day whose matches it already reflects, so an exact-date merge would
pull post-match ratings into the features of the match being predicted.
