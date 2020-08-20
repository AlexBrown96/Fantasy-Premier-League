import pandas as pd
import numpy as np
import os

teams_dir = '../Fantasy-Premier-League/champ_promoted_stats'
Prem_players_list = pd.read_csv("../Fantasy-Premier-League/data/2020-21/cleaned_players.csv")
# For all the players listed in the data/year directory, train the model...

# Find champ players in prem players list for next season
Prem_players_list["name"] = ["{} {}".format(x, y) for x,y in list(zip(Prem_players_list["first_name"], Prem_players_list["second_name"]))]


def get_price(name):
    temp_data = (np.array(Prem_players_list))[:, -1]
    if name in temp_data:
        player_index = (np.nonzero(Prem_players_list["name"] == name)[0][0])
        player_price = (np.array(Prem_players_list))[player_index][17]
        return player_price
    else:
        return 40

heads = ["Player", "Pos", "MP", "Min", "Gls",
         "Ast", "PK", "PKatt", "CrdY", "CrdR", "Starts", "MOTM"]


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
        return cs_def_gk * clean_sheets
    elif pos == 3:
        return cs_mid * clean_sheets
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
            p = (games_played * 2)
            return p
    else:
        return 0


def total_points(player_data, team_name):
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
    total += clean_sheet_points(pos)
    # Bonus points 3 for MOTM
    # Forumla from Man City data
    MOTM = player_data[11]
    BP = -0.2355*(MOTM**2) + 5.6108*MOTM + 2.2519
    total += BP
    if total != total:
        total = 0
    # Rough price estimate
    if games_played > 10:
        total = total / games_played
    else:
        total = 0
    player_price = get_price(player_data[0])
    if games_played > 15:
        playing_chance = games_played/46
    else:
        playing_chance = 0

    return [player_data[0], round(total), round(total), 1, player_price, playing_chance, 'nan', round(total), player_data[1], team_name]

Records = []
for subdir, dirs, files in os.walk(teams_dir):
    league_data = np.array(pd.read_csv(teams_dir+"/team_stats.csv"))
    for file in files:
        team_name = subdir.replace(teams_dir+"\\", '')
        team_name = team_name.replace("_", ' ')
        if file == "stats.csv":
            team_data = pd.read_csv(subdir+"/stats.csv")
            team_data["Player"] = [i.split(str("\\"), 1) for i in team_data["Player"]]
            team_data["Player"] = [str(i[0]) for i in team_data["Player"]]
            temp_data = league_data[:, 0]
            team_index = (np.nonzero(temp_data == team_name)[0][0])
            clean_sheets = league_data[team_index][12]
            team_data = np.array(team_data[heads])
            for i in range(len(team_data)):
                Records.append(total_points(team_data[i], int(str(team_index)+"01")))

df = pd.DataFrame([i for i in Records],
                  columns=['name', 'predicted_points','recent_points',
                           'accuracy', 'player_recent_value', 'chance_playing_next_round',
                           'news', 'points_per_game', 'position', 'team_code'])
df.to_csv('Champ_player_predictions.csv', index=False)
