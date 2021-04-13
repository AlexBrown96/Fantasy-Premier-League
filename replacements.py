import pandas as pd
from gameweek import get_recent_gameweek_id as gameweek_id

machine_data = pd.read_csv("../Fantasy-Premier-League/data/2020-21/selections/player_predictions_week_{}.csv".format(gameweek_id()))
actual_team = pd.read_csv("../Fantasy-Premier-League/team_162673_data20_21/picks_{}.csv".format(gameweek_id()))


def convert_picks():
    raw_data = pd.read_csv("../Fantasy-Premier-League/data/2020-21/players_raw.csv")
    r = pd.DataFrame(columns=raw_data.columns)
    for id in actual_team["element"]:
        t = raw_data[raw_data["id"] == id]
        r = r.append(t)
    r["predicted_points"] = r["id"].map(dict(zip(machine_data.id, machine_data.points)))
    return r

def print_team():
    a = convert_picks()
    print(a.sort_values(by=["predicted_points"]))
    print(a[a["element_type"]==1][["web_name", "predicted_points"]])
    print(a[a["element_type"]==2][["web_name", "predicted_points"]])
    print(a[a["element_type"]==3][["web_name", "predicted_points"]])
    print(a[a["element_type"]==4][["web_name", "predicted_points"]])



print_team()
# df = convert_picks()
# df = df.sort_values(by=["predicted_points"])
breakpoint()

# def gk_replacement():
#     data = convert_picks()
#     gks = machine_data[machine_data["pos"] == 1]
#     my_gks = data[data["element_type"] == 1]
#     gk_list = gks[gks["points"] > 0]
#     avg = sum(gks["points"]) / len(gks["points"])
#     breakpoint()
#
# defs = machine_data[machine_data["pos"] == 2]
# mids = machine_data[machine_data["pos"] == 3]
# fwds = machine_data[machine_data["pos"] == 4]
#
#
# gk_replacement()