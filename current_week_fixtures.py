import pandas as pd
import numpy as np
from gameweek import get_recent_gameweek_id

gameweek_num = get_recent_gameweek_id() +1
# Get next the fixtures for next week
def get_current_week_fixtures():
    fixture_data = pd.read_csv("../Fantasy-Premier-League/data/2020-21/fixtures.csv")
    f_headers = ["event", "team_a", "team_h", "team_h_difficulty", "team_a_difficulty"]
    current_fix = fixture_data[f_headers].loc[fixture_data["event"] == gameweek_num]

    # Convert team ids to names

    team_data = pd.read_csv("../Fantasy-Premier-League/data/2020-21/teams.csv")
    t_headers = ["id", "code"]
    tn1, tn2 = [], []
    for a in current_fix["team_a"]:
        temp = team_data[t_headers].loc[team_data["id"] == a]
        tn1.append(list(temp["code"])[0])
    for h in current_fix["team_h"]:
        temp = team_data[t_headers].loc[team_data["id"] == h]
        tn2.append(list(temp["code"])[0])

    current_fix["team_a_code"] = tn1
    current_fix["team_h_code"] = tn2
    return current_fix
#print(get_current_week_fixtures())


def get_next_n_fixtures(n=1):
    fixture_data = pd.read_csv("../Fantasy-Premier-League/data/2020-21/fixtures.csv")
    f_headers = ["event", "team_a", "team_h", "team_h_difficulty", "team_a_difficulty"]
    current_fix = fixture_data[f_headers][(fixture_data["event"] >= gameweek_num) &
                                          (fixture_data["event"] < gameweek_num+n)]
    # Convert team ids to names

    team_data = pd.read_csv("../Fantasy-Premier-League/data/2020-21/teams.csv")
    t_headers = ["id", "code"]
    tn1, tn2 = [], []
    for a in current_fix["team_a"]:
        temp = team_data[t_headers].loc[team_data["id"] == a]
        tn1.append(list(temp["code"])[0])
    for h in current_fix["team_h"]:
        temp = team_data[t_headers].loc[team_data["id"] == h]
        tn2.append(list(temp["code"])[0])

    current_fix["team_a_code"] = tn1
    current_fix["team_h_code"] = tn2
    return current_fix
