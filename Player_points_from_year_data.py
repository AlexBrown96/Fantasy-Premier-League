# TODO create generic ML model to predict points based on position, team, was_home then predict clean sheets ect
import sklearn
from sklearn import linear_model
import pandas as pd
import numpy as np
import pickle
import time
import Fixture_difficulty as fd

start_time = time.time()

pd.set_option("display.max_columns", None)
pd_in = pd.read_csv("../Fantasy-Premier-League/data/2019-20/players_raw.csv")
us_in = pd.read_csv("../Fantasy-Premier-League/data/2019-20/understat/understat_player.csv")
fd_in = pd.read_csv("../Fantasy-Premier-League/data/2019-20/fixtures.csv")
us_in["player_name"] = [(''.join(filter(lambda j: j.isalpha(), i))) for i in us_in["player_name"]]
player_raw_data = np.array(pd_in)
understat_raw_data = np.array(us_in)
pd_heads = ["element_type", "team", "now_cost"]
us_heads = ["games", "xG", "xA"]

data_in = pd.read_csv("../Fantasy-Premier-League/data/2019-20/gws/merged_gw.csv")


def selected_stats(data, heads, row_index):
    return data[heads].loc[row_index]


def organise_data(merged_gw_data):
    pos_list = []
    team_list = []
    cost_list = []
    xG_list = []
    xA_list = []
    games_list = []
    # headers pos, min, team, was_home, xCleansheet, xG, xA
    '''
    name,assists,bonus,bps,clean_sheets,
    creativity,element,fixture,goals_conceded,
    goals_scored,ict_index,influence,
    kickoff_time,minutes,opponent_team,own_goals,
    penalties_missed,penalties_saved,red_cards,
    round,saves,selected,team_a_score,team_h_score,
    threat,total_points,transfers_balance,transfers_in,
    transfers_out,value,was_home,yellow_cards,GW
    '''
    #merged_gw_data = merged_gw_data[:200]

    id = merged_gw_data["name"]
    merged_gw_data["id"] = [(''.join(filter(lambda j: j.isdigit(), i))) for i in merged_gw_data["name"]]
    merged_gw_data["name"] = [(''.join(filter(lambda j: j.isalpha(), i))) for i in merged_gw_data["name"]]

    # First get the player's team, position and fixture difficulties
    for row, val in enumerate(merged_gw_data["id"]):
        player_id_data = player_raw_data[:, 26]
        player_index = (np.nonzero(player_id_data == int(val))[0][0])
        # Obtain player data from the selected column heading
        extra_data = selected_stats(pd_in, pd_heads, player_index)
        pos_list.append(extra_data["element_type"])
        team_list.append(extra_data["team"])
        cost_list.append(extra_data["now_cost"])
    for row, val in enumerate(merged_gw_data["name"]):
        # Understat data
        val.replace(" ", "")
        player_id_data_us = understat_raw_data[:, 1]
        try:
            player_index_us = (np.nonzero(player_id_data_us == val)[0][0])
        except:
            extra_data_us = selected_stats(us_in, us_heads, player_index_us)
            xG_list.append(0)
            xA_list.append(0)
            games_list.append(0)
        else:
            extra_data_us = selected_stats(us_in, us_heads, player_index_us)
            xG_list.append(extra_data_us["xG"])
            xA_list.append(extra_data_us["xA"])
            games_list.append(extra_data_us["games"])
    xG_list = [np.true_divide(x, y) if y != 0 else 0 for x, y in np.array(list(zip(xG_list, games_list)))]
    xA_list = [np.true_divide(x, y) if y != 0 else 0 for x, y in np.array(list(zip(xA_list, games_list)))]
    merged_gw_data["pos"] = pos_list
    merged_gw_data["now_cost"] = cost_list
    merged_gw_data["team"] = team_list
    merged_gw_data["xG"] = xG_list
    merged_gw_data["xA"] = xA_list
    merged_gw_data["games"] = games_list
    # Modify the input data based on the selected features
    heads = ["total_points", "pos", "minutes", "team", "now_cost", "was_home", "ict_index", "xG", "xA"]
    player_data = merged_gw_data[heads]
    # Drop the predicted points label to produce x and y
    x = np.array(player_data.drop(["total_points"], 1))
    y = np.array(player_data)
    print("Organising time of {} rows of player data: {}".format(len(merged_gw_data["name"]), time.time()-start_time))
    return x, y


training_counts = 50


def train_model(x_data, y_data, n=1):
    best_acc = 0
    models = [[], []]
    for counts in range(n):
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x_data, y_data, test_size=0.15)

        linear = linear_model.LinearRegression()
        linear.fit(x_train, y_train)

        acc = linear.score(x_test, y_test)
        # acc = linear.score(x_train, y_train)
        models[0].append(acc)
        models[1].append(linear)
    best_acc = max(models[0])
    best_linear = models[1][models[0].index(best_acc)]
    return best_linear, best_acc


# Pass data set into organise function
# Organised_data = organise_data(___.csv)
x, y = organise_data(data_in)
# Train the model based on this data
model = train_model(x,y)
pickle.dump(model, open("general_model.p", "wb"))





