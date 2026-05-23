import pandas as pd

df = pd.read_csv('all_matches.csv')

df['date'] = pd.to_datetime(df['date'])

df = df[df['date'] >= '2010-01-01']

team_rows = []

for _, row in df.iterrows():

    # HOME TEAM PERSPECTIVE
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
        'neutral': row['neutral']
    })

    # AWAY TEAM PERSPECTIVE
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
        'neutral': row['neutral']
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
team_df = team_df[(team_df['team'].isin(top_teams) & team_df['opponent'].isin(top_teams))]
team_df = team_df.sort_values(
    by=['team', 'date']
)
team_df['rolling_goal_diff_5'] = (
    team_df
    .groupby('team')['goal_diff']
    .transform(
        lambda x: x.shift().rolling(5).mean()
    )
)

team_df['win'] = (
    team_df['result'] == 1
).astype(int)

team_df['rolling_win_rate_5'] = (
    team_df
    .groupby('team')['win']
    .transform(
        lambda x: x.shift().rolling(5).mean()
    )
)
weights = {
    'World Cup': 5,
    'World Cup qualifier': 4,
    'European Championship qual': 3,
    'Copa America': 3,
    'African Nations Cup': 3,
    'Friendly': 1
}

team_df['tournament_weight'] = (
    team_df['tournament']
    .map(weights)
    .fillna(2)
)

team_df = team_df.dropna()

elo_df = pd.read_csv("eloratings.csv")
elo_df['date'] = pd.to_datetime(
    elo_df['date'],
    format = 'mixed'
)
team_df = team_df.sort_values('date')
elo_df = elo_df.sort_values('date')

team_df = pd.merge_asof(
    team_df,
    elo_df[['date', 'team', 'rating']],
    on='date',
    by='team',
    direction='backward'
)
team_df = team_df.rename(
    columns={'rating': 'team_elo'}
)

# Create temporary opponent column in elo_df
elo_opponent = elo_df.rename(
    columns={
        'team': 'opponent',
        'rating': 'opponent_elo'
    }
)


team_df = pd.merge_asof(
    team_df.sort_values('date'),
    elo_opponent[['date', 'opponent', 'opponent_elo']],
    on='date',
    by='opponent',
    direction='backward'
)

# Elo difference
team_df['elo_diff'] = (
    team_df['team_elo'] -
    team_df['opponent_elo']
)

team_df = team_df.dropna()
team_df.to_csv('filtered_matches.csv', index=False)
print(team_df[['team_elo', 'opponent_elo']].isna().sum())