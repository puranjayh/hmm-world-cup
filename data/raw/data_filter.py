import pandas as pd
import numpy as np

df = pd.read_csv('all_matches.csv')
df['date'] = pd.to_datetime(df['date'])
df = df[df['date'] >= '2008-01-01']

team_rows = []

for _, row in df.iterrows():
    home_result = (
        1 if row['home_score'] > row['away_score']
        else 0 if row['home_score'] == row['away_score']
        else -1
    )
    team_rows.append({
        'date': row['date'],
        'team': row['home_team'],
        'opponent': row['away_team'],
        'goals_for': row['home_score'],
        'goals_against': row['away_score'],
        'goal_diff': row['home_score'] - row['away_score'],
        'result': home_result,
        'tournament': row['tournament'],
        'neutral': row['neutral'],
    })

    away_result = (
        1 if row['away_score'] > row['home_score']
        else 0 if row['away_score'] == row['home_score']
        else -1
    )
    team_rows.append({
        'date': row['date'],
        'team': row['away_team'],
        'opponent': row['home_team'],
        'goals_for': row['away_score'],
        'goals_against': row['home_score'],
        'goal_diff': row['away_score'] - row['home_score'],
        'result': away_result,
        'tournament': row['tournament'],
        'neutral': row['neutral'],
    })

team_df = pd.DataFrame(team_rows)

top_teams = [
    "France", "Spain", "Argentina", "England",
    "Portugal", "Brazil", "Netherlands", "Morocco",
    "Belgium", "Germany", "Croatia", "Italy",
    "Colombia", "Senegal", "Mexico", "United States",
    "Uruguay", "Japan", "Switzerland", "Denmark",
    "Iran", "Turkey", "Ecuador", "Austria",
    "South Korea", "Nigeria", "Australia", "Algeria",
    "Egypt", "Canada", "Norway", "Ukraine",
    "Panama", "Ivory Coast", "Poland", "Russia",
    "Wales", "Sweden", "Serbia", "Paraguay",
    "Czechia", "Hungary", "Scotland", "Tunisia",
    "Cameroon", "DR Congo", "Greece", "Slovakia",
    "Venezuela", "Uzbekistan"
]
team_df = team_df[
    team_df['team'].isin(top_teams) &
    team_df['opponent'].isin(top_teams)
]
team_df = team_df.sort_values(['team', 'date']).reset_index(drop=True)

# ── Rolling window features (standard) ──────────────────────────────────────
team_df['rolling_goal_diff_5'] = (
    team_df.groupby('team')['goal_diff']
    .transform(lambda x: x.shift().rolling(5).mean())
)
team_df['win'] = (team_df['result'] == 1).astype(int)
team_df['rolling_win_rate_5'] = (
    team_df.groupby('team')['win']
    .transform(lambda x: x.shift().rolling(5).mean())
)

# ── Exponentially weighted win rate (span=5) — weights recent matches more ──
team_df['ewa_win_rate'] = (
    team_df.groupby('team')['win']
    .transform(lambda x: x.shift().ewm(span=5, min_periods=3).mean())
)

# ── Exponentially weighted goal diff ─────────────────────────────────────────
team_df['ewa_goal_diff'] = (
    team_df.groupby('team')['goal_diff']
    .transform(lambda x: x.shift().ewm(span=5, min_periods=3).mean())
)

# ── Tournament weight ────────────────────────────────────────────────────────
weights = {
    'World Cup': 5,
    'World Cup qualifier': 4,
    'European Championship qual': 3,
    'Copa America': 3,
    'African Nations Cup': 3,
    'Friendly': 1,
}
team_df['tournament_weight'] = team_df['tournament'].map(weights).fillna(2)

# ── Drop rows missing core rolling features ──────────────────────────────────
team_df = team_df.dropna(subset=['rolling_goal_diff_5', 'rolling_win_rate_5'])

# ── Merge Elo ratings ────────────────────────────────────────────────────────
elo_df = pd.read_csv("eloratings.csv")
elo_df['date'] = pd.to_datetime(elo_df['date'], format='mixed')
team_df = team_df.sort_values('date')
elo_df  = elo_df.sort_values('date')

team_df = pd.merge_asof(
    team_df,
    elo_df[['date', 'team', 'rating']],
    on='date', by='team', direction='backward'
)
team_df = team_df.rename(columns={'rating': 'team_elo'})

elo_opp = elo_df.rename(columns={'team': 'opponent', 'rating': 'opponent_elo'})
team_df = pd.merge_asof(
    team_df.sort_values('date'),
    elo_opp[['date', 'opponent', 'opponent_elo']],
    on='date', by='opponent', direction='backward'
)
team_df['elo_diff'] = team_df['team_elo'] - team_df['opponent_elo']
team_df = team_df.dropna(subset=['team_elo', 'opponent_elo'])

# ── Opponent strength features ───────────────────────────────────────────────
# Average Elo of last 5 opponents — measures strength of schedule
team_df = team_df.sort_values(['team', 'date']).reset_index(drop=True)
team_df['opp_elo_strength_5'] = (
    team_df.groupby('team')['opponent_elo']
    .transform(lambda x: x.shift().rolling(5).mean())
)

# Win rate against top-half teams (opponent_elo >= median at time of match)
global_median_elo = 1500
team_df['win_vs_strong'] = np.where(
    team_df['opponent_elo'] >= global_median_elo,
    team_df['win'], np.nan
)
team_df['rolling_win_vs_strong_5'] = (
    team_df.groupby('team')['win_vs_strong']
    .transform(lambda x: x.shift().rolling(5, min_periods=2).mean())
).fillna(team_df['rolling_win_rate_5'])   # fallback to overall win rate

team_df.to_csv('filtered_matches.csv', index=False)

print("Done. Columns:", team_df.columns.tolist())
print("Shape:", team_df.shape)
print("NaNs in new features:")
new_cols = ['ewa_win_rate', 'ewa_goal_diff', 'opp_elo_strength_5', 'rolling_win_vs_strong_5']
print(team_df[new_cols].isna().sum())