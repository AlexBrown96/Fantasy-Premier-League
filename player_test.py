import pickle
import pandas as pd
import numpy as np
import sklearn
import os

pd_in = pd.read_csv("../Fantasy-Premier-League/data/2019-20/players_raw.csv")
player_raw_data = np.array(pd_in)

model = pickle.load(open("general_model.p", "rb"))

players_dir = '../Fantasy-Premier-League/data/2019-20/players'

# For all the players listed in the data/year directory, train the model...
Records = []
for subdir, dirs, files in os.walk(players_dir):
    for file in files:
        if file == "gw.csv":
            data = pd.read_csv(subdir+"/gw.csv", sep=",")
            if len(data["round"]) > 12:
                player_id = (''.join(filter(lambda i: i.isdigit(), subdir))).replace('201920', '')
                # From raw data

                heads = ["total_points", "pos", "minutes", "team", "now_cost", "was_home", "ict_index"]


def predict_points(player_data):
    points = (model.predict(np.array([player_data])))
    return points


def stats_raw(id):
    # Get the row of player data associated with the ID
    player_id_data = player_raw_data[:,26]
    player_index = (np.nonzero(player_id_data == id)[0][0])
    # Obtain player data from the selected column heading
    extra_data = selected_stats(player_index)
    '''
    Could be a better way of doing this method but would require rewrite
    '''
    position = extra_data["element_type"]
    team_code = extra_data["team"]
    cost = extra_data["now_cost"]
    return position, team_code, cost


def selected_stats(row_index):
    return pd_in[heads].loc[row_index]