import sklearn
from sklearn import linear_model
import warnings
import pandas as pd
import numpy as np
import pickle
import time
import gameweek
import current_week_fixtures
from understat import parse_player_data



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
pd_heads = ["element", "team", "now_cost"]
us_heads = ["games", "xG", "xA", "npxG"]
gk_heads = ["total_points", "now_cost", "was_home", "team_strength", "opp_strength", "clean_sheets"]
def_heads = ["total_points", "now_cost", "was_home", "team_strength", "opp_strength", "xA_dif", "xG_dif", "clean_sheets"]
mid_heads = ["total_points", "now_cost", "was_home", "team_strength", "opp_strength", "xA_dif", "xG_dif"]
fwd_heads = ["total_points", "now_cost", "was_home", "team_strength", "opp_strength", "xA_dif", "xG_dif"]


def Organise_season_data(data_set):
    #
    data_set = data_set.rename(columns={"id": "element"})
    # map positions to numerical values
    data_set["position"] = data_set["position"].map({"FWD": 4, "MID": 3, "DEF": 2, "GK": 1})
    # remove anyone who did not play so the dataset is smaller
    data_set = data_set[data_set.minutes > 0]
    # gather team data
    data_set["team_code"] = data_set["team"].map(dict(zip(teams_in.name, teams_in.code)))
    # gather game strength
    data_set["team_strength"] = data_set["team_code"].map(dict(zip(teams_in.code, teams_in.strength)))
    data_set["opp_strength"] = data_set["opponent_team"].map(dict(zip(teams_in.id, teams_in.strength)))
    # gather further player info
    data_set["now_cost"] = data_set["element"].map(dict(zip(pd_in.id, pd_in.now_cost)))
    # understat data
    # understat names
    w_names = pd.read_csv("../Fantasy-Premier-League/data/2020-21/understat/players/weird_names.csv", names=["read_name", "us_name"])
    w_name_dict = dict(zip(w_names.read_name, w_names.us_name))
    #na_names = data_set.loc[data_set["us_id"].isna()]["name"].unique()
    data_set["name"] = [(''.join(filter(lambda j: j.isalpha(), i))) for i in data_set["name"]]
    data_set["name"] = data_set["name"].replace(w_name_dict)
    data_set["name"] = [(''.join(filter(lambda j: j.isalpha(), i))) for i in data_set["name"]]
    data_set["us_id"] = data_set["name"].map(dict(zip(us_in.player_name, us_in.id)))
    data_set["time"] = [i[:10] for i in data_set["kickoff_time"]]

    data_set = data_set.sort_values(["us_id", "GW"])

    for n in data_set["us_id"].unique():
        stats = ["xG", "xA"]
        for stat in stats:
            try:
                us_player_data = pd.read_csv("../Fantasy-Premier-League/data/2020-21/understat/players/{}.csv".format(str(int(n))))
                temp = data_set.loc[data_set["us_id"] == n]["time"].map(dict(zip(us_player_data.date, us_player_data[stat])))
                data_set.loc[data_set.us_id == n, stat] = temp
                print(round(100*np.where(data_set["us_id"].unique()==n)[0][0]/len(data_set["us_id"].unique()),3), "% Done")
            except (FileNotFoundError, IOError):
                print("player {} not found in files...".format(n))

    print("finished_organising_data")
    data_set["xG"] = data_set["xG"].fillna(0)
    data_set["xA"] = data_set["xA"].fillna(0)
    data_set["xG_dif"] = data_set["goals_scored"] - data_set["xG"]
    data_set["xA_dif"] = data_set["assists"] - data_set["xA"]
    data_set.to_csv("temp_data_set.csv")
    return data_set


def train_model(data, heads, pos_n, training_counts=100, model="General model"):
    # TODO maybe train with players who have played more than x games
    models = [[],[]]
    data = data[data.position == pos_n][heads]
    print("Training model with {} counts".format(training_counts))
    x_data = np.array(data.drop(["total_points"], 1))
    y_data = np.array(data["total_points"])
    for count in range(training_counts):
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x_data, y_data, test_size=0.05)
        linear = linear_model.LinearRegression()
        a1 = np.isnan(np.sum(x_train))
        a2 = np.isnan(np.sum(y_train))
        linear.fit(x_train, y_train)
        acc = linear.score(x_test, y_test)
        models[0].append(acc)
        models[1].append(linear)

    best_acc = max(models[0])
    best_linear = models[1][models[0].index(best_acc)]
    print(model, ":", "Accuracy : ", best_acc)
    return best_linear


def feature_prediction(data, player_name, team_code, player_id):
    '''
    gk_heads = ["total_points", "now_cost", "was_home", "team_strength", "opp_strength", "clean_sheets"]
    def_heads = ["total_points", "now_cost", "was_home", "team_strength", "opp_strength", "xA_dif", "xG_dif", "clean_sheets"]
    mid_heads = ["total_points", "now_cost", "was_home", "team_strength", "opp_strength", "xA_dif", "xG_dif"]
    fwd_heads = ["total_points", "now_cost", "was_home", "team_strength", "opp_strength", "xA_dif", "xG_dif"]
    '''
    # TODO maybe look at fixture diff next game and look at previous?
    # TODO maybe look if player is from a "big 6" or "new promoted" team?
    # TODO maybe goal difference to games played
    data = data[data["element"] == player_id]
    merged_data = pd.read_csv("temp_data_set.csv")
    merged_data = merged_data[merged_data["element"] == player_id]
    games_played = len(merged_data["GW"])
    xG = sum(merged_data["xG_dif"]) / games_played
    xA = sum(merged_data["xA_dif"]) / games_played
    value = merged_data["value"].iloc[-1]
    cs = merged_data["clean_sheets"].sum() / games_played
    pos = merged_data["position"].iloc[-1]
    # Fixtures
    future_fixtures_n = 4
    current_fixtures = current_week_fixtures.get_next_n_fixtures(future_fixtures_n)
    team_a_n = list(current_fixtures["team_a_code"]).count(team_code)
    team_a = list(current_week_fixtures.get_current_week_fixtures()["team_a_code"])
    team_h_n = list(current_fixtures["team_h_code"]).count(team_code)
    team_h = list(current_week_fixtures.get_current_week_fixtures()["team_h_code"])
    multiplyer = (team_a_n + team_h_n) / future_fixtures_n
    # if data["chance_of_playing_next_round"] == "None":
    #     cpnr = "100"
    # multiplyer *= float(cpnr)/100
    # TODO this may fail for bgw/dgw and postponed fixtures
    games = dict(zip(team_a, team_h))
    if team_code in games.keys():
        was_home = False
    else:
        was_home = True
    str_opp = fd_in[fd_in["team_code"] == team_h[0]]["strength"]
    str_t = merged_data["team_strength"].iloc[-1]
    breakpoint()
    # Predictions
    if pos == 1:
        with open('gk_model.p', "rb") as l:
            linear = pickle.load(l)
        vals = [value, was_home, str_t, str_opp, cs]
    elif pos == 2:
        with open('def_model.p', "rb") as l:
            linear = pickle.load(l)
        vals = [value, was_home, str_t, str_opp, xA, xG, cs]
    elif pos == 3:
        with open('mid_model.p', "rb") as l:
            linear = pickle.load(l)
        vals = [value, was_home, str_t, str_opp, xA, xG]
    elif pos == 4:
        with open('gk_model.p', "rb") as l:
            linear = pickle.load(l)
        vals = [value, was_home, str_t, str_opp, xA, xG]
    breakpoint()


def main():
    # data_in = pd.read_csv("../Fantasy-Premier-League/data/2020-21/gws/merged_gw.csv")
    # x = Organise_season_data(data_in.set_index("GW"))
    x = pd.read_csv("../Fantasy-Premier-League/data/2020-21/players/Timo_Werner_117/gw.csv")
    feature_prediction(x, "TimoWerner", 8, 117)
    # with open('gk_model.p', "wb") as m:
    #     pickle.dump(train_model(x, gk_heads, 1, 4000, "gk model"), m)
    # with open('def_model.p', "wb") as m:
    #     pickle.dump(train_model(x, def_heads, 2, 5000, "def model"), m)
    # with open('mid_model.p', "wb") as m:
    #     pickle.dump(train_model(x, mid_heads, 3, 10000, "mid model"), m)
    # with open('fwd_model.p', "wb") as m:
    #     pickle.dump(train_model(x, fwd_heads, 4, 10000, "fwd model"), m)


if __name__ == "__main__":
    main()