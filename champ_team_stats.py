import pandas as pd
import numpy as np
import os

teams_dir = '../FPL_ML_2020/champ_promoted_stats'

# For all the players listed in the data/year directory, train the model...
Records = []
heads = ["Player", "Pos", "MP", "Min", "Gls",
         "Ast", "PK", "PKatt", "CrdY", "CrdR", "Starts"]
for subdir, dirs, files in os.walk(teams_dir):
    for file in files:
        if file == "stats.csv":
            team_data = pd.read_csv(subdir+"/stats.csv", sep=",")
            team_data["Player"] = [i.split(str("\\"), 1) for i in team_data["Player"]]
            team_data["Player"] = [str(i[0]) for i in team_data["Player"]]
            print(team_data.head())




breakpoint()
# TODO automate the selected team with the master function
# Clean_sheets = pd.DataFrame(data={"Team":["Leeds_United", "West_Brom","Fulham"],
#                                   "CS":[22, 14, 17]})
# a = Clean_sheets["Team"][0].loc



# league_heads = ["CS"]


time_played60 = 1
time_played60_plus = 2
goal_def_gk = 6
goal_mid = 5
goal_fwd = 4
assist = 3
cs_def_gk = 4
cs_mid = 1
goal_a_gk_def = -1
yellow_card = -1
red_card = -3
penalty_miss = -2


def get_pos(player_data):
    index = 1
    if player_data[index][0] == "G":
        return 1
    elif player_data[index][0] == "D":
        return 2
    elif player_data[index][0] == "M":
        return 3
    elif player_data[index][0] == "F":
        return 4
    else:
        return 0


def clean_sheet_points(pos):
    if pos == 1 or pos == 2:
        return cs_def_gk * cs
    elif pos == 3:
        return cs_mid * cs
    else:
        return 0


def goal_points(goals, pos):
    if pos == 1 or pos == 2:
        return goals * goal_def_gk
    elif pos == 3:
        return goals * goal_mid
    elif pos == 4:
        return goals * goal_fwd
    else:
        return 0


def time_points(time, games_played, starts, pos):
    if games_played > 0:
        avg_mins = (time / games_played)
        if avg_mins < 70:
            return games_played
        else:
            # clean this up brah
            p = (games_played * 2) + clean_sheet_points(pos)
            return p
    else:
        return 0


def total_points(player_data):
    total = 0
    # Get points scored for playing
    time = player_data[3]
    games_played = player_data[2]
    starts = player_data[10]
    pos = get_pos(player_data)
    total += time_points(time, games_played, starts, pos)
    # Get points scored for goals
    goals_scored = player_data[4]
    total += goal_points(goals_scored, pos)
    # Get points scored for assists
    assists = player_data[5]
    total += assists * assist
    # Deduct penalty misses
    total -= (player_data[7]-player_data[6]) * penalty_miss
    # Deduct points for yellow and red cards
    total -= player_data[8] * yellow_card
    total -= player_data[9] * red_card
    # Add points for clean sheets
    if total != total:
        total = 0
    return player_data[0], round(total * (38/46))



# ["Player", "Pos", "Starts", "Min", "Gls", "Ast", "PK", "PKatt", "CrdY", "CrdR"]
# ['Ben White' 'DF'    46     4140.0  1.0    2.0    0.0    0.0     6.0     0.0]

# points system used in fpl

