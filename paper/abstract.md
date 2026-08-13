# A Global Gaussian Hidden Markov Model for Interpretable Football Match Forecasting

## Abstract

Most international football forecasts come from strength ratings like Elo or from
black-box machine learning. Neither gives an interpretable, time-varying picture of
team form, which is what analysts and coaches actually reason about. We take a
different route: a single global Gaussian hidden Markov model pools every national
team's match history to track form as a hidden, time-varying state. Our approach asks
whether this hidden form carries information a single rating does not, and whether it
can be expressed as a calibrated probability rather than a point estimate.

We pool 16,722 international matches (33,455 team-perspective records; 2008–June 2026,
217 teams; results from a Kaggle dataset, ratings from eloratings.net) into one global
Gaussian HMM. Seven latent states, shared across teams, are learned unsupervised from
seven signals: exponentially weighted win rate and goal differential, win rate against
strong opponents, two volatility measures, and two momentum measures, forming an
interpretable ladder from "Poor" to "Elite." A forward pass gives each team a
leakage-free state distribution from its earlier matches only. The 7×7 joint
distribution of both teams, each team's confidence and entropy, and a form-adjusted
Elo gap feed a logistic classifier for Win/Draw/Loss, blended with a specialist draw
classifier. After each result, both the team's form-state distribution and its Elo
rating are updated and rolled forward, so form and rating shifts compound
match-to-match.

Out of sample on four unseen tournaments (2018 and 2022 World Cups, Euro 2024, Copa
América 2024; n = 211), three-way accuracy is 55.5%, on par with Elo (55.0%) and with
gradient boosting given the same rating (XGBoost 55.0%, Random Forest 51.2%). Accuracy
does not separate the methods — the model's contribution is its interpretable output,
not a higher pick rate. Decisive-match (Win/Loss) accuracy is 74.5%. Every prediction
also carries a confidence and entropy score describing how settled each team's form
is; gating on confidence lifts accuracy as coverage narrows (e.g., 58.6% to 75.0% in
the 2018 sample). In an out-of-sample test on the 2026 World Cup, after the training
cutoff, the model scored 65.4% overall (68/104) and 82.1% on decisive matches (69/84),
matching a contemporaneous Elo baseline on decisive matches.

Conclusion. The value of this model is not a higher pick rate but a richer output: a
distribution over seven interpretable form states, plus a confidence and uncertainty
score for every team in every match. This makes it a base model to build on —
confidence can drive position-sizing or staking rules that scale exposure to
prediction certainty, and the same state-and-uncertainty representation offers a
structured slot for incorporating qualitative factors such as injuries, lineups, or
travel that the model itself ignores.
