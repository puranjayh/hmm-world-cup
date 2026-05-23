# HMM World Cup Prediction System

A research-oriented football prediction framework using:

- Hidden Markov Models (HMMs)
- Bayesian Updating
- Elo Ratings
- Rolling Form Metrics
- Expected Goals (xG)

The project aims to model latent team performance states and dynamically update match outcome probabilities throughout international tournaments.

---

# Research Premise

Traditional football prediction systems rely heavily on static rankings or short-term form. This project models football teams as hidden dynamic states (e.g. strong form, unstable form, declining form) using Hidden Markov Models.

After every match:
- the observed result updates the estimated hidden state probabilities,
- which then influence future match predictions.

The system combines:
- historical international match data,
- Elo ratings,
- rolling statistical features,
- and modern event-level xG metrics.

---

# Core Methodology

## 1. Data Collection

International football match data:
- 2010 onwards
- all major international teams
- tournament + friendly matches

Sources:
- Kaggle international football results
- Elo ratings dataset
- StatsBomb Open Data

---

## 2. Feature Engineering

Engineered features include:

- Goals scored
- Goals conceded
- Goal difference
- Match result
- Elo rating
- Elo difference
- Rolling average form
- Rolling goals scored/conceded
- Tournament type
- Neutral venue flag
- Expected Goals (xG)

---

## 3. Hidden Markov Model

The HMM assumes:
- teams transition between hidden performance states,
- match results are observable emissions from those states.

Example hidden states:
- Dominant
- Stable
- Declining
- Unstable

The model updates state probabilities after every observed match result using Bayesian updating.

---

## 4. Bayesian Updating

After each match:
- prior probabilities are updated,
- posterior confidence in each hidden state is recalculated,
- predictions for future matches are adjusted dynamically.

This allows the system to adapt during tournaments.

---

# Dataset Structure

```text
data/
│
├── raw/
│   ├── all_matches.csv
│   ├── eloratings.csv
│   ├── match_ids.csv
│   ├── matches/
│   └── events/
│
└── processed/
    ├── filtered_matches.csv
    ├── xg_dataset.csv
    └── final_dataset.csv