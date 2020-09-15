from statistics import mean
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import Fixture_difficulty as fd
from gameweek import get_recent_gameweek_id
gameweek = get_recent_gameweek_id()
fixture_data = pd.read_csv("../Fantasy-Premier-League/data/2019-20/fixtures.csv")


def predicted_points(team_code, data, training_counts=10, player_name="", past_position="2"):
    # Features used to train the model
    # TODO current team code
    headers = ["total_points", "assists", "clean_sheets",
               "goals_scored", "was_home", "saves", "ict_index", "value", "minutes", "bonus"]
    # Sort out selected to a rough percentage
    # Work out the fixture difficulty rating so that it can be added to the model
    team_dif_data = fd.fixture_dif_data(team_code, fixture_data)
    fd_temp = []
    for k, v in enumerate(data["round"]):
        # Get the indexes of rounds played
        if v in team_dif_data[1]:
            idx = team_dif_data[1].index(v)
            fd_temp.append(team_dif_data[0][idx])
    # Add position data
    pos_temp = []
    for i in range(len(data["round"])):
        pos_temp.append(past_position)
    #
    player_data = data[headers]
    team_dif_data = pd.DataFrame(fd_temp, columns=["fixture_difficulty"])
    headers.append("fixture_difficulty")
    player_data = pd.concat([player_data, team_dif_data], axis=1)
    #
    pos_list_data = pd.DataFrame(pos_temp, columns=["position"])
    headers.append("position")
    player_data = pd.concat([player_data, pos_list_data], axis=1)
    x = np.array(player_data.drop(["total_points"], 1))
    # <class 'list'>: [0, 0.02631578947368421, 0, True, 0.0, 0.6815789473684211, 55.0, 3, 3.0, 0.1]
    # Array of labels
    y = np.array(player_data["total_points"])
    linear, acc = train_model(x, y, training_counts)
    pred, value, pos = feature_prediction(linear, data, team_code, player_name)
    return pred, acc, value, pos


def train_model(x_data, y_data, n=1):
    best_acc = 0
    models = [[],[]]
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


def selected_stats(data, heads, row_index):
    return data[heads].loc[row_index]


us_in = pd.read_csv("../Fantasy-Premier-League/data/2020-21/understat/understat_player.csv", encoding='latin-1')
us_in["player_name"] = [(''.join(filter(lambda j: j.isalpha(), i))) for i in us_in["player_name"]]
understat_raw_data = np.array(us_in)
current_player_data = pd.read_csv("../Fantasy-Premier-League/data/2020-21/players_raw.csv")
current_player_data["name"] = [(''.join(filter(lambda j: j.isalpha(), "{}{}".format(x,y)))) for x,y in
                               list(zip(current_player_data["first_name"], current_player_data["second_name"]))]



def feature_prediction(linear, data, team_code, player_name):
    # work out if the player is actually playing this season
    headers = ["assists", "clean_sheets",
               "goals_scored", "was_home", "saves", "value", "selected"]
    # Get last seasons data

    # Assists and goals scored from understat
    us_heads = ["xG", "xA", "games"]
    pd_heads = ["clean_sheets", "saves"]
    # player_name.replace(" ", "")
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
    team_code = extra_data["team"]
    bonus = extra_data["bonus"] / games
    #selected = extra_data["selected_by_percent"]
    avg_mins = extra_data["minutes"] / games
    # TODO find out why this is very high for some players
    ict = extra_data["ict_index"] / games
    fixture_data = pd.read_csv("../Fantasy-Premier-League/data/2020-21/fixtures.csv")
    fixture_dif = (fd.fixture_dif_data(team_code, fixture_data))[0][gameweek]
    # Get new team code
    team_a = np.array(fixture_data["team_a"])[gameweek*20:(gameweek*20)+20]
    team_h = np.array(fixture_data["team_h"])[gameweek*20:(gameweek*20)+20]
    team_a_index = (np.nonzero(team_a == team_code)[0][0])
    team_h_index = (np.nonzero(team_h == team_code)[0][0])

    if team_a_index > team_h_index:
        was_home = True
    else:
        was_home = False

    # Predictions
    # TODO could this be done for multiple future gameweeks eg 3 gws

    predictions = [xA, cs, xG, was_home, saves, ict, value, avg_mins, bonus, fixture_dif, pos]
    points = float(linear.predict(np.array([predictions])))
    return points, value, pos
