import sklearn
from sklearn import linear_model
import warnings
import pandas as pd
import numpy as np
import pickle
import gameweek
import current_week_fixtures
from tqdm import tqdm


warnings.simplefilter(action='ignore', category=UnboundLocalError)
gameweek = gameweek.get_recent_gameweek_id()
pd.set_option("display.max_columns", None)

# Import data sets
pd_in = pd.read_csv("../Fantasy-Premier-League/data/2020-21/players_raw.csv")
us_in = pd.read_csv("../Fantasy-Premier-League/data/2020-21/understat/understat_player.csv")
fd_in = pd.read_csv("../Fantasy-Premier-League/data/2020-21/fixtures.csv")
#
us_in["player_name"] = [(''.join(filter(lambda j: j.isalpha(), i))) for i in us_in["player_name"]]
player_raw_data = np.array(pd_in)
understat_raw_data = np.array(us_in)
teams_in = pd.read_csv("../Fantasy-Premier-League/data/2020-21/teams.csv")
teams_raw = np.array(teams_in)

# Headers
heads = ["total_points", "now_cost", "was_home", "team_strength",
             "opp_strength", "xA_dif", "xG_dif", "clean_sheets",
             "ict_index", "minutes", "big_six", "big_six_opp",
             "last_points", "gk", "def", "mid", "fwd"]

def Organise_season_data(data_set):
    '''
    data_set: Merged_gws from data/year/gws
    '''
    data_set = data_set.rename(columns={"id": "element"})
    # map positions to numerical values
    data_set["position"] = data_set["position"].map({"FWD": 4, "MID": 3, "DEF": 2, "GK": 1})
    # remove anyone who did not play so the dataset is smaller
    data_set = data_set[data_set.minutes != "0"]
    # gather team data
    data_set["team_code"] = data_set["team"].map(dict(zip(teams_in.name, teams_in.code)))
    data_set["big_six"] = data_set["team_code"].map({3: True, 8: True, 14: True, 43: True, 1: True, 6: True}).fillna(False)
    data_set["big_six_opp"] = data_set["opponent_team"].map({1: True, 5: True, 11: True, 12: True, 13: True, 17: True}).fillna(False)
    # gather game strength
    data_set["team_strength"] = data_set["team_code"].map(dict(zip(teams_in.code, teams_in.strength)))
    data_set["opp_strength"] = data_set["opponent_team"].map(dict(zip(teams_in.id, teams_in.strength)))
    # gather further player info
    data_set["now_cost"] = data_set["element"].map(dict(zip(pd_in.id, pd_in.now_cost)))
    # understat data
    # understat names
    w_names = pd.read_csv("../Fantasy-Premier-League/data/2020-21/understat/players/weird_names.csv", names=["read_name", "us_name"])
    w_name_dict = dict(zip(w_names.read_name, w_names.us_name))
    data_set["name"] = [(''.join(filter(lambda j: j.isalpha(), i))) for i in data_set["name"]]
    data_set["name"] = data_set["name"].replace(w_name_dict)
    data_set["name"] = [(''.join(filter(lambda j: j.isalpha(), i))) for i in data_set["name"]]
    data_set["us_id"] = data_set["name"].map(dict(zip(us_in.player_name, us_in.id)))
    # Trim the date of the match
    data_set["time"] = [i[:10] for i in data_set["kickoff_time"]]
    data_set = data_set.sort_values(["us_id", "round"])
    with tqdm(total=len(data_set["us_id"].unique()), position=0, leave=True, desc="Merging fpl and understat data") as pbar:
        for n in data_set["us_id"].unique():
            stats = ["npxG", "xA", "npg"]
            pbar.update()
            for stat in stats:
                try:
                    us_player_data = pd.read_csv("../Fantasy-Premier-League/data/2020-21/understat/players/{}.csv".format(str(int(n))))
                    temp = data_set.loc[data_set["us_id"] == n]["time"].map(dict(zip(us_player_data.date, us_player_data[stat])))
                    data_set.loc[data_set.us_id == n, stat] = temp


                except (ValueError, IOError):
                    print("player {} not found in files...".format(n))
            data_set.loc[data_set.us_id == n, "last_points"] = data_set.loc[data_set["us_id"] == n]["total_points"].shift(periods=1)
    data_set = data_set.dropna(subset=["last_points"])
    print("finished_organising_data")
    data_set["npxG"] = data_set["npxG"].fillna(0)
    data_set["npg"] = data_set["npg"].fillna(0)
    data_set["xA"] = data_set["xA"].fillna(0)
    data_set["xG_dif"] = data_set["npxG"] - data_set["npg"]
    data_set["xA_dif"] = data_set["xA"] - data_set["assists"]
    # Position Classifier
    for pos in ["gk", "def", "mid", "fwd"]:
        id = {"gk": 1, "def": 2, "mid": 3, "fwd": 4}.get(pos)
        data_set[pos] = data_set["position"].map({id: True}).fillna(False)
    data_set.to_csv("temp_data_set.csv")
    return data_set


def train_model(data, heads, pos_n, training_counts=100, model="General model"):
    # TODO maybe train with players who have played more than x games
    data = data[data.minutes != "0"]
    models = [[], []]
    #data = data[data.position == pos_n][heads]
    data = data[heads]
    print("Training model with {} counts".format(training_counts))
    x_data = np.array(data.drop(["total_points"], 1))
    y_data = np.array(data["total_points"])
    for count in range(training_counts):
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x_data, y_data, test_size=0.05)
        linear = linear_model.LinearRegression()
        linear.fit(x_train, y_train)
        acc = linear.score(x_test, y_test)
        models[0].append(acc)
        models[1].append(linear)

    best_acc = max(models[0])
    best_linear = models[1][models[0].index(best_acc)]
    print(model, ":", "Accuracy : ", best_acc)
    return best_linear


def feature_prediction(team_code, player_id):
    '''
    :param team_code: team code in teams.csv
    :param player_id: fpl player id
    :return: predicted_points, value, pos, captain points
    '''
    # TODO maybe look at fixture diff next game and look at previous?
    # TODO maybe goal difference to games played
    merged_data = pd.read_csv("temp_data_set.csv")
    if player_id not in list(merged_data["element"]):
        return 0, 0, 0, 0
    merged_data = merged_data[merged_data["element"] == player_id][-8:]
    games_played = len(merged_data["GW"])

    value, pos, big_six, last_points = merged_data["value"].iloc[-1], \
                                       merged_data["position"].iloc[-1], \
                                       merged_data["big_six"].iloc[0], \
                                       merged_data["last_points"].iloc[0]
    minutes = sum(merged_data["minutes"])/games_played#[-3:]) / 3
    # if minutes <= 45:
    #     return 0, value, pos, 0
    # if games_played == 0:
    #     return 0, value, pos, 0
    xG, xA, cs, ict = sum(merged_data["xG_dif"]) / games_played, \
                      sum(merged_data["xA_dif"]) / games_played, \
                      sum(merged_data["clean_sheets"]) / games_played, \
                      sum(merged_data["ict_index"]) / games_played
    # Fixtures
    future_fixtures_n = 4
    current_fixtures = current_week_fixtures.get_next_n_fixtures(future_fixtures_n)
    team_a_n, team_h_n = list(current_fixtures["team_a_code"]).count(team_code), \
                         list(current_fixtures["team_h_code"]).count(team_code)
    team_a = list(current_week_fixtures.get_current_week_fixtures()["team_a_code"])
    team_count = list(current_week_fixtures.get_current_week_fixtures()["team_h_code"]) + team_a
    multiplier, captain_multiplier = (team_a_n + team_h_n) / future_fixtures_n, \
                                     team_count.count(team_code)
    opp_list = list(current_fixtures[current_fixtures["team_a_code"] == team_code]["team_h_code"]) + \
               list(current_fixtures[current_fixtures["team_h_code"] == team_code]["team_a_code"])
    if opp_list[0] in [3, 8, 14, 43, 1, 6]:
        big_six_opp = True
    else:
        big_six_opp = False
    if team_code in team_a:
        was_home = False
    else:
        was_home = True
    tot_str = 0
    for opp in opp_list:
        tot_str += teams_in[teams_in["code"] == opp]["strength"].iloc[0]
    opp_str = tot_str / len(opp_list)
    strength = teams_in[teams_in["code"] == team_code]["strength"].iloc[0]
    gk = defe = mid = fwd = False
    if pos == 1: gk = True
    elif pos == 2: defe = True
    elif pos == 3: mid = True
    elif pos == 4: fwd = True
    # Open the model from file
    #model_type = {1:'gk_model.p', 2: 'def_model.p', 3: 'mid_model.p', 4: 'fwd_model.p'}.get(pos)
    with open('general_model.p', "rb") as saved_model:
        linear = pickle.load(saved_model)
    vals = [value, was_home, strength, opp_str, xA, xG, cs, ict, minutes, big_six, big_six_opp, last_points, gk, defe, mid, fwd]
    predictions = float(linear.predict([np.array(vals)])[0])
    pd.set_option('display.width', 200)
    df = pd.DataFrame([linear.coef_, vals, linear.coef_ * vals], index=["coef", "input_val", "sum"],
                      columns=["now_cost", "was_home", "team_strength",
                               "opp_strength", "xA_dif", "xG_dif", "clean_sheets",
                               "ict_index", "minutes", "big_six", "big_six_opp",
                               "last_points", "gk", "def", "mid", "fwd"])
    print(df)
    breakpoint()
    return multiplier*predictions, value, pos, captain_multiplier*predictions


def main():
    # data_in = pd.read_csv("../Fantasy-Premier-League/data/2020-21/gws/merged_gw.csv")
    # x = Organise_season_data(data_in.set_index("GW"))
    # #x = pd.read_csv("temp_data_set.csv")
    # for model in range(1, 5, 1):
    #     model_type = {1: 'gk_model.p', 2: 'def_model.p', 3: 'mid_model.p', 4: 'fwd_model.p'}.get(model)
    #     with open(model_type, "wb") as m:
    #         pickle.dump(train_model(x, heads, 1, 1000, model_type), m)
    # with open("general_model.p", "wb") as m:
    #     pickle.dump(train_model(x, heads, 1, 1000, "General_model.p"), m)
    print(feature_prediction(35, 481))

if __name__ == "__main__":
    main()