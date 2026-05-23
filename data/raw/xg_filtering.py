import json
import os
import pandas as pd

# LOAD FILTERED DATASET
filtered_matches = pd.read_csv(
    'filtered_matches.csv'
)

filtered_matches['date'] = pd.to_datetime(
    filtered_matches['date']
)

# KEEP ONLY ONE ROW PER MATCH
filtered_unique = filtered_matches[[
    'date',
    'team',
    'opponent',
    'goals_for',
    'goals_against',
    'tournament'
]].copy()

# NORMALIZE TEAM ORDER
filtered_unique['team_1'] = filtered_unique[
    ['team', 'opponent']
].min(axis=1)

filtered_unique['team_2'] = filtered_unique[
    ['team', 'opponent']
].max(axis=1)

filtered_unique['score_1'] = filtered_unique[
    ['goals_for', 'goals_against']
].min(axis=1)

filtered_unique['score_2'] = filtered_unique[
    ['goals_for', 'goals_against']
].max(axis=1)

# REMOVE DUPLICATES
filtered_unique = filtered_unique.drop_duplicates(
    subset=[
        'date',
        'team_1',
        'team_2',
        'score_1',
        'score_2'
    ]
)

# STATSBOMB COMPETITIONS
competitions = ['43', '55', '223', '1267']

all_matches = []

for comp in competitions:

    folder_path = f'matches/{comp}'

    for file_name in os.listdir(folder_path):

        if not file_name.endswith('.json'):
            continue

        file_path = os.path.join(
            folder_path,
            file_name
        )

        with open(file_path, 'r', encoding='utf-8') as f:

            matches = json.load(f)

        for match in matches:

            home_team = match[
                'home_team'
            ]['home_team_name']

            away_team = match[
                'away_team'
            ]['away_team_name']

            home_score = match['home_score']
            away_score = match['away_score']

            all_matches.append({

                'competition_id': comp,

                'match_id': match['match_id'],

                'date': pd.to_datetime(
                    match['match_date']
                ),

                'home_team': home_team,

                'away_team': away_team,

                'home_score': home_score,

                'away_score': away_score,

                'team_1': min(
                    home_team,
                    away_team
                ),

                'team_2': max(
                    home_team,
                    away_team
                ),

                'score_1': min(
                    home_score,
                    away_score
                ),

                'score_2': max(
                    home_score,
                    away_score
                )

            })

matches_df = pd.DataFrame(all_matches)

# MATCH AGAINST FILTERED DATASET
matches_df = matches_df.merge(

    filtered_unique,

    on=[
        'date',
        'team_1',
        'team_2',
        'score_1',
        'score_2'
    ],

    how='inner'
)

# SAVE
matches_df.to_csv(
    'match_ids.csv',
    index=False
)

print(matches_df['tournament'].value_counts())
print(matches_df.shape)