import pandas as pd
import numpy as np

df = pd.read_csv('all_matches.csv')
df['date'] = pd.to_datetime(df['date'], format='mixed', dayfirst=True)
df = df[df['date'] >= '2008-01-01']

# Normalise team names to match the simulator's conventions
NAME_MAP = {'United States': 'USA'}
df['home_team'] = df['home_team'].replace(NAME_MAP)
df['away_team'] = df['away_team'].replace(NAME_MAP)

# ── Deduplicate source matches (same date/home/away, different tournament name) ──
df = df.drop_duplicates(subset=['date', 'home_team', 'away_team'], keep='first')

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

WC2026_TEAMS = [
    "Mexico", "USA", "Canada", "Colombia",
    "Ecuador", "Uruguay", "Panama", "Bolivia",
    "Argentina", "Chile", "Peru", "Venezuela",
    "Brazil", "Paraguay", "Costa Rica", "Guatemala",
    "France", "Belgium", "England", "Serbia",
    "Portugal", "Spain", "Turkey", "Ukraine",
    "Germany", "Netherlands", "Denmark", "Cameroon",
    "Italy", "Croatia", "Switzerland", "Albania",
    "Morocco", "Senegal", "South Africa", "Tunisia",
    "Nigeria", "Egypt", "Algeria", "Ivory Coast",
    "Japan", "South Korea", "Australia", "Saudi Arabia",
    "Iran", "Uzbekistan", "Qatar", "Kyrgyzstan", "Bosnia and Herzegovina", "Hungary",
    "Georgia", "Slovakia", "Slovenia", "Austria",
    "Romania", "Burkina Faso", "Cape Verde", "Guinea", "Equatorial Guinea", 
    "Mozambique", "Tanzania", "Angola", "Mali", "Gambia",
]
# Keep any match where the team we're tracking is a WC 2026 participant.
# Opponent can be anyone — their Elo comes from the eloratings merge (~240 teams).
team_df = team_df[team_df['team'].isin(WC2026_TEAMS)]
team_df = team_df.sort_values(['team', 'date']).reset_index(drop=True)

# ── Outcome column (2=win, 1=draw, 0=loss) — required by evaluate_global.py ──
team_df['outcome'] = team_df['result'].map({1: 2, 0: 1, -1: 0})

# ── Days since previous match (NO LEAKAGE) ─────────────────────────────────
team_df['days_since_last_match'] = (
    team_df.groupby('team')['date']
    .diff()
    .dt.days
)

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
# Clean non-breaking spaces and normalise names to match match data
elo_df['team'] = elo_df['team'].str.replace('\xa0', ' ', regex=False)
elo_df['team'] = elo_df['team'].replace({'United States': 'USA'})
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

# ── Rolling volatility features (NO LEAKAGE) ────────────────────────────────
team_df['rolling_goal_diff_std_5'] = (
    team_df.groupby('team')['goal_diff']
    .transform(lambda x: x.shift().rolling(5).std())
)

team_df['rolling_win_rate_std_5'] = (
    team_df.groupby('team')['win']
    .transform(lambda x: x.shift().rolling(5).std())
)

# ── Days since previous match (NO LEAKAGE) ─────────────────────────────────
team_df['days_since_last_match'] = (
    team_df.groupby('team')['date']
    .diff()
    .dt.days
)

# ── EWA momentum features (NO LEAKAGE) ──────────────────────────────────────
team_df['ewa_win_rate_momentum'] = (
    team_df.groupby('team')['ewa_win_rate']
    .transform(lambda x: x - x.shift(5))
)

team_df['ewa_goal_diff_momentum'] = (
    team_df.groupby('team')['ewa_goal_diff']
    .transform(lambda x: x - x.shift(5))
)
# Instead of filtering to WC2026_TEAMS, keep all teams
# that appear in the data with sufficient history
team_df = team_df[
    team_df.groupby("team")["team"]
    .transform("count") >= 10  # at least 10 matches in dataset
]

team_df.to_csv('filtered_matches.csv', index=False)