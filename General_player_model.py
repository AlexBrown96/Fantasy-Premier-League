import sklearn
from sklearn import linear_model
import warnings
warnings.simplefilter(action='ignore', category=UnboundLocalError)
import pandas as pd
import numpy as np
import pickle
import time
import Fixture_difficulty as fd

start_time = time.time()

pd.set_option("display.max_columns", None)
pd_in = pd.read_csv("../Fantasy-Premier-League/data/2019-20/players_raw.csv")
us_in = pd.read_csv("../Fantasy-Premier-League/data/2019-20/understat/understat_player.csv", encoding='latin-1')
fd_in = pd.read_csv("../Fantasy-Premier-League/data/2019-20/fixtures.csv")
us_in["player_name"] = [(''.join(filter(lambda j: j.isalpha(), i))) for i in us_in["player_name"]]
player_raw_data = np.array(pd_in)
understat_raw_data = np.array(us_in)
teams_in = pd.read_csv("../Fantasy-Premier-League/data/2019-20/teams.csv")
teams_raw = np.array(teams_in)
pd_heads = ["element_type", "team", "now_cost"]
us_heads = ["games", "xG", "xA"]

def selected_stats(data, heads, row_index):
    return data[heads].loc[row_index]


def Organise_data_set(season_data):
    '''
    Function takes merged gw data and organises stats
    for each player from the season.

    Returns the following features as an npy array:

    "total_points", "pos", "minutes", "now_cost",
    "was_home", "ict_index", "xG", "xA", "bonus",
    "clean_sheets", "strength", "saves"

    '''
    pos_list = []
    team_list = []
    cost_list = []
    xG_list = []
    xA_list = []
    games_list = []
    # headers pos, min, team, was_home, xCleansheet, xG, xA
    season_data["id"] = [(''.join(filter(lambda j: j.isdigit(), i))) for i in season_data["name"]]
    season_data["name"] = [(''.join(filter(lambda j: j.isalpha(), i))) for i in season_data["name"]]
    
    # First get the player's team, position and fixture difficulties
    for row, val in enumerate(season_data["id"]):
        player_id_data = player_raw_data[:, 26]
        player_index = (np.nonzero(player_id_data == int(val))[0][0])
        # Obtain player data from the selected column heading
        extra_data = selected_stats(pd_in, pd_heads, player_index)
        pos_list.append(extra_data["element_type"])
        team_list.append(extra_data["team"])
        cost_list.append(extra_data["now_cost"])
    print("finished appending data from players")
    for row, val in enumerate(season_data["name"]):
        # Understat data
        val.replace(" ", "")
        player_id_data_us = us_in["player_name"]
        if val in list(us_in["player_name"]):
            player_index_us = (np.nonzero(player_id_data_us == val)[0][0])
            extra_data_us = selected_stats(us_in, us_heads, player_index_us)
            xG_list.append(extra_data_us["xG"])
            xA_list.append(extra_data_us["xA"])
            games_list.append(extra_data_us["games"])
        else:
            xG_list.append(0)
            xA_list.append(0)
            games_list.append(0)

    xG_list = [np.true_divide(x, y) if y != 0 else 0 for x, y in np.array(list(zip(xG_list, games_list)))]
    xA_list = [np.true_divide(x, y) if y != 0 else 0 for x, y in np.array(list(zip(xA_list, games_list)))]

    season_data["pos"] = pos_list
    season_data["now_cost"] = cost_list
    season_data["team"] = team_list
    season_data["xG"] = xG_list
    season_data["xA"] = xA_list
    season_data["games"] = games_list
    print("finished appending season data")
    # Fixture difficulty
    strength = []
    for team in season_data["team"]:
        team_index = (np.nonzero(teams_raw == team)[0][0])
        extra_data_teams = selected_stats(teams_in, ["strength"], team_index)
        strength.append(extra_data_teams["strength"])
    season_data["strength"] = strength
    # Modify the input data based on the selected features

    heads = ["total_points", "pos", "minutes", "now_cost", "was_home", "ict_index", "xG", "xA", "bonus", "clean_sheets", "strength", "saves"]
    player_data = season_data[heads]
    # Drop the predicted points label to produce x and y
    x = np.array(player_data.drop(["total_points"], 1))
    y = np.array(player_data["total_points"])
    print("Organising time of {} rows of player data: {} seconds".format(len(season_data["name"]), time.time()-start_time))
    return x, y

########################################################################################################################

data_in = pd.read_csv("../Fantasy-Premier-League/data/2019-20/gws/merged_gw.csv")
x, y = None, None

# x, y = Organise_data_set(data_in)

# Save organised data

# if x is not None:
#     with open('x.p', "wb") as x_data:
#         pickle.dump(x, x_data)
# if y is not None:
#     with open('y.p', "wb") as y_data:
#         pickle.dump(y, y_data)

########################################################################################################################

def train_model(x_data, y_data, training_counts=1):
    best_acc = 0
    models = [[], []]
    print("Training model, with {} training counts".format(training_counts))
    for counts in range(training_counts):
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x_data, y_data, test_size=0.1)

        linear = linear_model.LinearRegression()
        linear.fit(x_train, y_train)

        acc = linear.score(x_test, y_test)
        # acc = linear.score(x_train, y_train)
        models[0].append(acc)
        models[1].append(linear)
    best_acc = max(models[0])
    best_linear = models[1][models[0].index(best_acc)]
    print("Accuracy: ", best_acc)
    return best_linear, best_acc

# with open('x.p', 'rb') as x:
#     x = pickle.load(x)
#
# with open('y.p', 'rb') as y:
#     y = pickle.load(y)

model = None
# model, acc = train_model(x, y, 1000)
#
# if model is not None:
#     with open('General_player_linear_model.p', "wb") as m:
#         pickle.dump(model, m)

########################################################################################################################
us_in = pd.read_csv("../Fantasy-Premier-League/data/2019-20/understat/understat_player.csv", encoding='latin-1')
us_in["player_name"] = [(''.join(filter(lambda j: j.isalpha(), i))) for i in us_in["player_name"]]
current_player_data = pd.read_csv("../Fantasy-Premier-League/data/2020-21/players_raw.csv")
current_player_data["name"] = [(''.join(filter(lambda j: j.isalpha(), "{}{}".format(x,y)))) for x,y in
                               list(zip(current_player_data["first_name"], current_player_data["second_name"]))]
current_teams = pd.read_csv("../Fantasy-Premier-League/data/2020-21/teams.csv")

gameweek = 0


def feature_prediction(linear, data, player_name, team_code):
    heads = ["total_points", "pos", "minutes", "team", "now_cost",
             "was_home", "ict_index", "xG", "xA", "bonus", "clean_sheets", "strength"]
    # Get last seasons data
    # Assists and goals scored from understat
    player_id_data_us = understat_raw_data[:, 1]
    try:
        player_index_us = (np.nonzero(player_id_data_us == player_name)[0][0])
    except:
        # TODO if xG and xA not found and player is mid/fwd take the team xG and xA
        xG = 0
        xA = 0
    else:
        extra_data_us = selected_stats(us_in, us_heads, player_index_us)
        games = np.array(extra_data_us["games"])
        xG = np.array(extra_data_us["xG"]) / games
        xA = np.array(extra_data_us["xA"]) / games
    games = len(data["saves"])
    saves = np.sum(np.array(data["saves"])) / games
    cs = np.sum(np.array(data["clean_sheets"])) / games
    # Get gameweek

    # Values from current data
    temp = np.array(current_player_data)
    player_index_cp = (np.nonzero(np.array(current_player_data["name"]) == player_name)[0][0])
    extra_data = selected_stats(current_player_data,
                                ["now_cost", "element_type", "team", "ict_index",
                                 "selected_by_percent", "points_per_game", "minutes", "bonus"], player_index_cp)
    value = extra_data["now_cost"]
    pos = extra_data["element_type"]
    team = extra_data["team"]
    bonus = extra_data["bonus"] / games
    avg_mins = extra_data["minutes"] / games
    ict = extra_data["ict_index"] / games

    # Get fixture data
    fixture_data = pd.read_csv("../Fantasy-Premier-League/data/2020-21/fixtures.csv")
    #fixture_dif = (fd.fixture_dif_data(team, fixture_data))[0][gameweek]
    # Get new team code
    gameweek = 1
    gw_matches = list(np.where(np.array(fixture_data["event"]) == gameweek)[0])
    team_a = np.array(fixture_data["team_a"])[gw_matches[0]:gw_matches[-1]]
    team_h = np.array(fixture_data["team_h"])[gw_matches[0]:gw_matches[-1]]
    if np.int64(team) not in team_a and team not in team_h:
        return 0, value, pos
    else:
        if team in team_a:
            was_home = False
        else:
            was_home = True

    # Get team strength for upcoming fixture
    c_t = np.array(current_teams["code"])
    if team_code in c_t:
        team_index = (np.nonzero(c_t == team_code)[0][0])
        strength = selected_stats(current_teams, ["strength"], team_index)[0]
    else:
        strength = 2
    # Predictions
    # TODO could this be done for multiple future gameweeks eg 3 gws
    predictions = np.array([pos, avg_mins, value, was_home, ict, xG, xA, bonus, cs, strength, saves])
    points = float(linear.predict([predictions]))
    return points, value, pos


