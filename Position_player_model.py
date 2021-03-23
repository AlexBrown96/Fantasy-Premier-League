import sklearn
from sklearn import linear_model
import warnings
import pandas as pd
import numpy as np
import pickle
import time
import gameweek
import current_week_fixtures
warnings.simplefilter(action='ignore', category=UnboundLocalError)
gameweek = gameweek.get_recent_gameweek_id()

start_time = time.time()

pd.set_option("display.max_columns", None)
pd_in = pd.read_csv("../Fantasy-Premier-League/data/2020-21/players_raw.csv")
us_in = pd.read_csv("../Fantasy-Premier-League/data/2020-21/understat/understat_player.csv", encoding='latin-1')
fd_in = pd.read_csv("../Fantasy-Premier-League/data/2020-21/fixtures.csv")
us_in["player_name"] = [(''.join(filter(lambda j: j.isalpha(), i))) for i in us_in["player_name"]]
player_raw_data = np.array(pd_in)
understat_raw_data = np.array(us_in)
teams_in = pd.read_csv("../Fantasy-Premier-League/data/2020-21/teams.csv")
teams_raw = np.array(teams_in)
pd_heads = ["element_type", "team", "now_cost"]
us_heads = ["games", "xG", "xA", "npxG"]

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
    season_data["id"] = season_data["element"]
    season_data["name"] = [(''.join(filter(lambda j: j.isalpha(), i))) for i in season_data["name"]]

    # First get the player's team, position and fixture difficulties
    for row, val in enumerate(season_data["id"]):
        player_id_data = player_raw_data[:, 30]
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


    # Get opponent_team_strength
    str_a_h = []
    str_a_a = []
    str_d_h = []
    str_d_a = []
    for team in season_data["opponent_team"]:
        opp_index = (np.nonzero(teams_raw == team))[0][0]
        extra_data_teams = selected_stats(teams_in, ["strength_attack_home", "strength_attack_away", "strength_defence_home", "strength_defence_away"], opp_index)
        str_a_h.append(extra_data_teams["strength_attack_home"])
        str_a_a.append(extra_data_teams["strength_attack_away"])
        str_d_h.append(extra_data_teams["strength_defence_home"])
        str_d_a.append(extra_data_teams["strength_defence_away"])
    # Drop the predicted points label to produce x and y
    season_data["strength_attack_home"] = str_a_h
    season_data["strength_attack_away"] = str_a_a
    season_data["strength_defence_home"] = str_d_h
    season_data["strength_defence_away"] = str_d_a
    heads = ["name", "total_points", "pos", "minutes", "now_cost", "was_home", "ict_index", "xG", "xA", "clean_sheets", "strength", "saves", "opponent_team", "strength_attack_home", "strength_attack_away", "strength_defence_home", "strength_defence_away"]
    player_data = season_data[heads]
    player_data.to_csv('temp.csv', index=False)
    # x = np.array(player_data.drop(["total_points"], 1))
    # y = np.array(player_data["total_points"])
    print("Organising time of {} rows of player data: {} seconds".format(len(season_data["name"]), time.time()-start_time))
    return player_data

########################################################################################################################

data_in = pd.read_csv("../Fantasy-Premier-League/data/2020-21/gws/merged_gw.csv")
x = None

#x = Organise_data_set(data_in)

#Save organised data

if x is not None:
    with open('x.p', "wb") as x_data:
        pickle.dump(x, x_data)

########################################################################################################################

gk_heads = ["total_points", "now_cost", "was_home", "ict_index", "clean_sheets", "saves", "strength_defence_home", "strength_defence_away"]
def_heads = ["total_points", "now_cost", "was_home", "ict_index", "clean_sheets", "xG", "xA", "strength_defence_home", "strength_defence_away"]
mid_heads = ["total_points", "now_cost", "was_home", "ict_index", "xG", "xA", "strength_attack_home", "strength_attack_away", "strength_defence_home", "strength_defence_away"]
fwd_heads = ["total_points", "now_cost", "was_home", "ict_index", "xG", "xA", "strength_attack_home", "strength_attack_away", ]


def train(x_data, heads, pos_n=1, training_counts=1):
    models = [[], []]
    print("Training model, with {} training counts".format(training_counts))
    temp = x_data[pd.DataFrame(x_data).pos == pos_n][heads]
    #temp["opponent_team"] = temp["opponent_team"].astype(str)
    x_data = np.array(temp.drop(["total_points"], 1))
    y_data = np.array(temp["total_points"])
    for counts in range(training_counts):
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x_data, y_data, test_size=0.1)

        linear = linear_model.LinearRegression()
        linear.fit(x_train, y_train)

        acc = linear.score(x_test, y_test)
        models[0].append(acc)
        models[1].append(linear)
    best_acc = max(models[0])
    best_linear = models[1][models[0].index(best_acc)]
    print("Accuracy: ", best_acc)
    return best_linear


# with open('x.p', 'rb') as x:
#     x = pickle.load(x)
# with open('gk_model.p', "wb") as m:
#     pickle.dump(train(x, gk_heads, 1, 3000), m)
# with open('def_model.p', "wb") as m:
#     pickle.dump(train(x, def_heads, 2, 3000), m)
# with open('mid_model.p', "wb") as m:
#     pickle.dump(train(x, mid_heads, 3, 3000), m)
# with open('fwd_model.p', "wb") as m:
#     pickle.dump(train(x, fwd_heads, 4, 3000), m)


########################################################################################################################
us_in = pd.read_csv("../Fantasy-Premier-League/data/2020-21/understat/understat_player.csv", encoding='latin-1')
us_in["player_name"] = [(''.join(filter(lambda j: j.isalpha(), i))) for i in us_in["player_name"]]
current_player_data = pd.read_csv("../Fantasy-Premier-League/data/2020-21/players_raw.csv")
current_player_data["name"] = [(''.join(filter(lambda j: j.isalpha(), "{}{}".format(x,y)))) for x,y in
                               list(zip(current_player_data["first_name"], current_player_data["second_name"]))]
current_teams = pd.read_csv("../Fantasy-Premier-League/data/2020-21/teams.csv")


def feature_prediction(data, player_name, team_code):
    '''
    ["total_points", "pos", "minutes", "team", "now_cost",
             "was_home", "ict_index", "xG", "xA", "clean_sheets", "strength"]
    '''
    # Get last seasons data
    # Assists and goals scored from understat
    player_id_data_us = understat_raw_data[:, 1]
    try:
        player_index_us = (np.nonzero(player_id_data_us == player_name)[0][0])
    except:
        xG = 0
        xA = 0
    else:
        extra_data_us = selected_stats(us_in, us_heads, player_index_us)
        games = np.array(extra_data_us["games"])
        xG = np.array(extra_data_us["xG"]) / games
        xA = np.array(extra_data_us["xA"]) / games
    games = len(data["saves"][:])
    saves = np.sum(np.array(data["saves"])[:]) / games
    cs = np.sum(np.array(data["clean_sheets"])[:]) / games
    # Get gameweek

    # Values from current data
    player_index_cp = (np.nonzero(np.array(current_player_data["name"]) == player_name)[0][0])
    extra_data = selected_stats(current_player_data,
                                ["now_cost", "element_type", "team_code", "ict_index",
                                 "selected_by_percent", "points_per_game", "minutes", "chance_of_playing_next_round"], player_index_cp)
    value = extra_data["now_cost"]
    pos = extra_data["element_type"]
    team = extra_data["team_code"]
    ict = extra_data["ict_index"] / games
    cpnr = extra_data["chance_of_playing_next_round"]
    # Get fixtures for the team that the player is playing for
    current_fixtures = current_week_fixtures.get_next_n_fixtures(4)
    team_a_n = list(current_fixtures["team_a_code"]).count(team)
    team_a = list(current_week_fixtures.get_current_week_fixtures()["team_a_code"])
    team_h_n = list(current_fixtures["team_h_code"]).count(team)
    team_h = list(current_week_fixtures.get_current_week_fixtures()["team_h_code"])
    multiplyer = (team_a_n + team_h_n) / 4
    # Chance of playing next round multiplyer
    if cpnr == "None":
        cpnr = "100"
    multiplyer *= float(cpnr)/100
    if team in team_a:
        was_home = False
    else:
        was_home = True

    # Get team strength for upcoming fixtures
    opp_index = (np.nonzero(teams_raw == team_code))[0][0]
    extra_data_teams = selected_stats(teams_in, ["strength_attack_home", "strength_attack_away", "strength_defence_home", "strength_defence_away"], opp_index)
    str_a_h = (extra_data_teams["strength_attack_home"])
    str_a_a = (extra_data_teams["strength_attack_away"])
    str_d_h = (extra_data_teams["strength_defence_home"])
    str_d_a = (extra_data_teams["strength_defence_away"])

    # Predictions
    if pos == 1:
        with open('gk_model.p', "rb") as l:
            linear = pickle.load(l)
        vals = [value, was_home, ict, cs, saves, str_d_h, str_d_a]
    elif pos == 2:
        with open('def_model.p', "rb") as l:
            linear = pickle.load(l)
        vals = [value, was_home, ict, cs, xG, xA, str_d_h, str_d_a]
    elif pos == 3:
        with open('mid_model.p', "rb") as l:
            linear = pickle.load(l)
        vals = [value, was_home, ict, xG, xA, str_d_h, str_d_a, str_a_h, str_a_a]
    elif pos == 4:
        with open('gk_model.p', "rb") as l:
            linear = pickle.load(l)
        vals = [value, was_home, ict, xG, xA, str_a_h, str_a_a]
    # TODO could this be done for multiple future gameweeks eg 3 gws
    predictions = np.array(vals)
    predictions = linear.predict([predictions])
    #   ["total_points", "pos", "minutes", "now_cost", "was_home", "ict_index", "xG", "xA", "clean_sheets", "strength", "saves"]
    return multiplyer*predictions[0], value, pos