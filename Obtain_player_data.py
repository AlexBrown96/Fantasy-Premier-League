import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
from Predicted_points import predicted_points
import numpy as np
# TODO remove relegated teams
# Obtain raw player data
pd_in = pd.read_csv("../Fantasy-Premier-League/data/2019-20/players_raw.csv")
player_raw_data = np.array(pd_in)
current_player_data = pd.read_csv("../Fantasy-Premier-League/data/2020-21/players_raw.csv")
current_player_data["name"] = [(''.join(filter(lambda j: j.isalpha(), "{}{}".format(x,y)))) for x,y in
                               list(zip(current_player_data["first_name"], current_player_data["second_name"]))]
cp = np.array(current_player_data)

# Column names for the selected stats from players_raw.csv
heads = ["web_name", "chance_of_playing_next_round", "news", "points_per_game", "element_type", "team", "now_cost"]


def selected_stats(row_index, df_in):
    return df_in[heads].loc[row_index]


def stats_raw(id):
    # Get the row of player data associated with the ID
    if id in player_raw_data[:,26]:
        player_id_data = player_raw_data[:,26]
        player_index = (np.nonzero(player_id_data == id)[0][0])
        extra_data = selected_stats(player_index, pd_in)
        team_code = extra_data["team"]
        past_position = extra_data["element_type"]
    else:
        print("player not found")
        return None
    return team_code, past_position


players_dir = '../Fantasy-Premier-League/data/2019-20/players'

# For all the players listed in the data/year directory, train the model...
Records = []
for subdir, dirs, files in os.walk(players_dir):
    for file in files:
        if file == "gw.csv":
            training_counts = 250
            # Minimum games played
            min_games = 5

            data = pd.read_csv(subdir+"/gw.csv", sep=",")
            player_id = (''.join(filter(lambda i: i.isdigit(), subdir))).replace('201920', '')
            if len(data["round"]) >= min_games:
                # From raw data
                team_code, past_position = stats_raw(int(player_id))
                if team_code == None:
                    print("player {}'s team not found".format(subdir.replace(players_dir,"")))
                else:
                    web_name = subdir.replace(players_dir, "")
                    web_name = (''.join(filter(lambda i: i.isalpha(), web_name)))
                    if web_name in cp:
                        player_id_data = current_player_data["name"]
                        player_index = (np.nonzero(player_id_data == web_name)[0][0])
                        chance_playing = np.array(current_player_data["chance_of_playing_next_round"])[player_index]
                        if chance_playing == "None":
                            points, acc, value, pos = predicted_points(team_code, data, training_counts, web_name, past_position)
                            if acc > 0.65:
                                print("player {} has been trained. Points = {}".format(web_name, points), "Accuracy = {} %".format(acc*100))
                                Records.append([web_name, points, acc, value, pos, team_code])
                            else:
                                print("player {}'s accuracy is too low".format(web_name))
                        else: print("player {} has low chance of playing next round".format(web_name))
                    else:
                        print("player {} is not playing this season".format(web_name))
            else:
                print("player {} not found".format(subdir.replace(players_dir,"")))


df = pd.DataFrame([i for i in Records],
                            columns=["web_name", "points", "acc", "value", "pos", "team_code"])
df.to_csv('Player_predictions.csv', index=False)
