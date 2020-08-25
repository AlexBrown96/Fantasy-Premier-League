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
player_ids = player_raw_data[:,26]
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
    if id in player_id[:]:

        #player_index = (np.nonzero(player_ids[:] == str(id))[0][0])
        player_index = np.where(player_ids[:] == np.int64(id))[0][0]
        extra_data = selected_stats(player_index, pd_in)
        team_code = extra_data["team"]
        past_position = extra_data["element_type"]
    else:
        print("player not found")
        breakpoint()
        return None, None
    return team_code, past_position


gw_dir = '../Fantasy-Premier-League/data/2019-20/gws'
num_players = []
for subdir2, dirs2, files2 in os.walk(gw_dir):
    for file in files2:
        if file != "merged_gw.csv":
            gw_data = pd.read_csv(subdir2+"/"+str(file), sep=",")
            num_players.append((gw_data["selected"].sum())/15)
num_players = [i for i in num_players if i > 0]

players_dir = '../Fantasy-Premier-League/data/2019-20/players'



# For all the players listed in the data/year directory, train the model...
Records = []
for subdir, dirs, files in os.walk(players_dir):
    for file in files:
        if file == "gw.csv":
            training_counts = 10
            # Minimum games played
            min_games = 14

            data = pd.read_csv(subdir+"/gw.csv", sep=",")
            data["selected_by_percent"] = [100*(val/num_players[key]) for key, val in enumerate(data["selected"])]
            player_id = (''.join(filter(lambda i: i.isdigit(), subdir))).replace('201920', '')
            if len(data["round"]) >= min_games:
                # From raw data
                team_code, past_position = stats_raw(player_id)
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
                            if acc > 0.80:
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
