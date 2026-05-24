import soccerdata as sd

fb = sd.FBref(

    leagues=[
        "INT-World Cup",
        "INT-European Championship"
    ],

    seasons=[
        2012,
        2014,
        2016,
        2018,
        2020,
        2022,
        2024
    ]
)

schedule = fb.read_schedule()

print(schedule.head())
print(schedule.columns)
team_stats = fb.read_team_match_stats()

print(team_stats.columns)
print(team_stats.head())