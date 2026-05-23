import json
import os
import pandas as pd

competitions = ['43', '55', '223', '1267']

all_matches = []

for comp in competitions:

    folder_path = f'matches/{comp}'

    for file_name in os.listdir(folder_path):

        file_path = os.path.join(folder_path, file_name)

        with open(file_path, 'r', encoding='utf-8') as f:
            matches = json.load(f)

        for match in matches:

            all_matches.append({

                'competition_id': comp,

                'match_id': match['match_id'],

                'match_date': match['match_date'],

                'home_team': match['home_team']['home_team_name'],

                'away_team': match['away_team']['away_team_name'],

                'home_score': match['home_score'],

                'away_score': match['away_score']

            })

matches_df = pd.DataFrame(all_matches)

matches_df.to_csv(
    'match_ids.csv',
    index=False
)

print(matches_df.head())
print(matches_df.shape)