import pandas as pd

current_team = pd.read_csv("../Fantasy-Premier-League/team_162673_data20_21/picks_1.csv")
current_players = pd.read_csv("Player_predictions.csv")
def conv_team(data):
    ids = list(data["id"])
    points = list(data["points"])
    names = list(data["web_name"])

    for key, val in enumerate(ids):
        if val not in list(current_team["element"]):
            points[key] = -100
    data["points"] = pd.Series(points)
    return data
