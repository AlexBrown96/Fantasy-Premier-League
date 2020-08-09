import os
import pandas as pd
from Predicted_points import predicted_points
import numpy as np

# Obtain raw player data
pd_in = pd.read_csv("../FPL_ML_2020/data/2019-20/players_raw.csv")
player_raw_data = np.array(pd_in)
# Column names for the selected stats from players_raw.csv
heads = ["web_name", "chance_of_playing_next_round", "news", "points_per_game", "element_type", "team", "now_cost"]


def selected_stats(row_index):
    return pd_in[heads].loc[row_index]


def stats_raw(id):
    # Get the row of player data associated with the ID
    player_id_data = player_raw_data[:,26]
    player_index = (np.nonzero(player_id_data == id)[0][0])
    # Obtain player data from the selected column heading
    extra_data = selected_stats(player_index)
    '''
    Could be a better way of doing this method but would require rewrite
    '''
    chance_playing_next_round = extra_data["chance_of_playing_next_round"]
    news = extra_data["news"]
    points_per_game = extra_data["points_per_game"]
    # position = get_position(extra_data["element_type"])
    position = extra_data["element_type"]
    team_code = extra_data["team"]
    web_name = extra_data["web_name"]
    cost = extra_data["now_cost"]
    return web_name, chance_playing_next_round, news, points_per_game, position, team_code, cost


players_dir = '../FPL_ML_2020/data/2019-20/players'

# For all the players listed in the data/year directory, train the model...
Records = []
for subdir, dirs, files in os.walk(players_dir):
    for file in files:
        if file == "gw.csv":
            training_counts = 5000
            n = 12

            data = pd.read_csv(subdir+"/gw.csv", sep=",")
            if len(data["round"]) > 12:
                player_id = (''.join(filter(lambda i: i.isdigit(), subdir))).replace('2020201920', '')
                # From raw data
                web_name, chance_playing_next_round, news, points_per_game, position, team_code, cost = stats_raw(int(player_id))
                # From training data
                if chance_playing_next_round != str(0):
                    points, n_points, acc = predicted_points(team_code, data, training_counts, n)
                    print("player {} has been trained".format(web_name), "Accuracy = {} %".format(acc*100))
                    Records.append([web_name, points, n_points, acc, cost, chance_playing_next_round, news,
                                    points_per_game, position, team_code])
                else:
                    print("player {}'s chance of playing too low".format(web_name))

df = pd.DataFrame([i for i in Records],
                            columns=['name', 'predicted_points','recent_points', 'accuracy', 'player_recent_value', 'chance_playing_next_round', 'news', 'points_per_game', 'position', 'team_code'])
df.to_csv('Player_predictions.csv', index=False)


def get_position(element_type):
    if element_type == 1:
        return "GK"
    elif element_type == 2:
        return "DEF"
    elif element_type == 3:
        return "MID"
    elif element_type == 4:
        return "FWD"
    else:
        return "None"

