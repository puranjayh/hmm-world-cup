# HMM Football Match Predictor

A small, educational implementation of a **3-state Hidden Markov Model** for
international football. Each national team has its own HMM whose hidden
states represent latent **form**:

- `0 = Poor Form`
- `1 = Neutral Form`
- `2 = Peak Form`

Each match is an emission from the team's current state and takes one of
three observable values: `0 = Loss`, `1 = Draw`, `2 = Win`.

To predict the outcome of `team A vs team B`, we:

1. Use each HMM to infer the **predictive distribution over current state**
   for both teams (Bayesian update from their match history).
2. Look up the **joint emission tensor** `P(outcome | state_team, state_opp)`
   learned from all training matches.
3. Marginalise over the joint state distribution to get `P(Win/Draw/Loss)`.

## Folder layout

| File | Purpose |
|------|---------|
| `config.py`         | Constants: paths, labels, train/test split date, seeds. |
| `data_loader.py`    | CSV loading, train/test split, per-team observation sequences. |
| `hmm_team.py`       | `TeamHMM` — per-team 3-state HMM (fit/save/load/predict). |
| `joint_emission.py` | Builds the `(state_team, state_opp) -> P(outcome)` tensor. |
| `bayesian_update.py`| Posterior state distribution given recent results. |
| `predictor.py`      | High-level `Predictor` combining HMMs + joint tensor. |
| `train.py`          | End-to-end training pipeline. |
| `predict_cli.py`    | Command-line prediction tool. |
| `artifacts/`        | Saved HMMs, joint tensor, metadata. |

## Quick start

```bash
pip install -r model/requirements.txt
python -m model.train
python -m model.predict_cli --team Brazil --opponent Argentina --date 2022-12-13
pytest tests/test_model.py -q
```

## Train / test split

Training uses every match with `date < 2022-01-01`. Everything from
2022 onwards (including the 2022 FIFA World Cup in Qatar) is held out as
the evaluation set, so the model never sees those matches during fitting.
