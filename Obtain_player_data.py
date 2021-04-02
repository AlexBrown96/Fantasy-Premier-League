import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np
from Position_player_model import feature_prediction


pd_in = pd.read_csv("../Fantasy-Premier-League/data/2020-21/players_raw.csv")
player_raw_data = np.array(pd_in)
player_ids = player_raw_data[:,30]
current_player_data = pd.read_csv("../Fantasy-Premier-League/data/2020-21/players_raw.csv")
current_player_data["name"] = [(''.join(filter(lambda j: j.isalpha(), "{}{}".format(x,y)))) for x,y in
                               list(zip(current_player_data["first_name"], current_player_data["second_name"]))]
cp = np.array(current_player_data)

# Column names for the selected stats from players_raw.csv
heads = ["web_name", "chance_of_playing_next_round", "news", "points_per_game", "element_type", "team", "now_cost", "team_code"]


def selected_stats(row_index, df_in):
    return df_in[heads].loc[row_index]


def stats_raw(id):
    # Get the row of player data associated with the ID
    if id in player_id[:]:
        #player_index = (np.nonzero(player_ids[:] == str(id))[0][0])
        player_index = np.where(player_ids[:] == np.int64(id))[0][0]
        extra_data = selected_stats(player_index, pd_in)
        team_code = extra_data["team_code"]
        past_position = extra_data["element_type"]
    else:
        print("player not found")
        breakpoint()
        return None, None
    return team_code, past_position


gw_dir = '../Fantasy-Premier-League/data/2020-21/gws'
num_players = []
for subdir2, dirs2, files2 in os.walk(gw_dir):
    for file in files2:
        if file != "merged_gw.csv":
            gw_data = pd.read_csv(subdir2+"/"+str(file), sep=",")
            num_players.append((gw_data["selected"].sum())/15)
num_players = [i for i in num_players if i > 0]
players_dir = '../Fantasy-Premier-League/data/2020-21/players'



# For all the players listed in the data/year directory, train the model...
Records = []
for subdir, dirs, files in os.walk(players_dir):
    for file in files:
        if file == "gw.csv":
            # Minimum games played
            min_games = 5
            data = pd.read_csv(subdir+"/gw.csv", sep=",")
            player_id = (''.join(filter(lambda i: i.isdigit(), subdir))).replace('202021', '')
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
                        # current_player_data["chance_of_playing_next_round"]
                        current_player_data.loc[current_player_data["chance_of_playing_next_round"] == "None", "chance_of_playing_next_round"] = "100"
                        chance_playing = np.array(current_player_data["chance_of_playing_next_round"])[player_index]
                        if chance_playing == "100":
                            points, value, pos = feature_prediction(data, web_name, team_code, player_id)
                            print("player {} has been trainined. Expected points: {}".format(web_name, points))
                            Records.append([web_name, points, value, pos, team_code, player_id])
                        else:
                            points, value, pos = feature_prediction(data, web_name, team_code)
                            Records.append([web_name, points*(4/5), value, pos, team_code, player_id])
                            print("player {} has low chance of playing next round, Expected score: {}".format(web_name, points*(4/5)))
                    else:
                        print("player {} is not playing this season".format(web_name))
            else:
                print("player {} not found".format(subdir.replace(players_dir,"")))


df = pd.DataFrame([i for i in Records],
                            columns=["web_name", "points", "value", "pos", "team_code", "id"])
df.to_csv('Player_predictions.csv', index=False)
