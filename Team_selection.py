import pandas as pd
import pulp
import numpy as np
import gameweek
trained_player_data = pd.read_csv("Player_predictions.csv")
#trained_player_data = pd.read_csv("../Fantasy-Premier-League/data/2020-21/players_raw.csv")
from Transfer_selection import TransferOptimiser

#trained_player_data2 = pd.read_csv("champ_player_predictions_temp.csv")
#trained_player_data = pd.concat([trained_player_data, trained_player_data2], axis=0, ignore_index=True)

# Code taken from https://github.com/nuebar/forecasting-fantasy-football
# nuebar's code:


def select_team(expected_scores, prices, positions, clubs, total_budget=104.8, sub_factor=0.1):
    num_players = len(expected_scores)
    model = pulp.LpProblem("Constrained value maximisation", pulp.LpMaximize)
    decisions = [
        pulp.LpVariable("x{}".format(i), lowBound=0, upBound=1, cat='Integer')
        for i in range(num_players)
    ]
    captain_decisions = [
        pulp.LpVariable("y{}".format(i), lowBound=0, upBound=1, cat='Integer')
        for i in range(num_players)
    ]
    sub_decisions = [
        pulp.LpVariable("z{}".format(i), lowBound=0, upBound=1, cat='Integer')
        for i in range(num_players)
    ]


    # objective function:
    model += sum((captain_decisions[i] + decisions[i] + sub_decisions[i]*sub_factor) * expected_scores[i]
                 for i in range(num_players)), "Objective"

    # cost constraint
    model += sum((decisions[i] + sub_decisions[i]) * prices[i] for i in range(num_players)) <= total_budget  # total cost

    # position constraints
    # 1 starting goalkeeper
    model += sum(decisions[i] for i in range(num_players) if positions[i] == 1) == 1
    # 2 total goalkeepers
    model += sum(decisions[i] + sub_decisions[i] for i in range(num_players) if positions[i] == 1) == 2

    # 3-5 starting defenders
    model += sum(decisions[i] for i in range(num_players) if positions[i] == 2) >= 3
    model += sum(decisions[i] for i in range(num_players) if positions[i] == 2) <= 5
    # 5 total defenders
    model += sum(decisions[i] + sub_decisions[i] for i in range(num_players) if positions[i] == 2) == 5

    # 3-5 starting midfielders
    model += sum(decisions[i] for i in range(num_players) if positions[i] == 3) >= 3
    model += sum(decisions[i] for i in range(num_players) if positions[i] == 3) <= 5
    # 5 total midfielders
    model += sum(decisions[i] + sub_decisions[i] for i in range(num_players) if positions[i] == 3) == 5

    # 1-3 starting attackers
    model += sum(decisions[i] for i in range(num_players) if positions[i] == 4) >= 1
    model += sum(decisions[i] for i in range(num_players) if positions[i] == 4) <= 3
    # 3 total attackers
    model += sum(decisions[i] + sub_decisions[i] for i in range(num_players) if positions[i] == 4) == 3

    # club constraint
    for club_id in np.unique(clubs):
        model += sum(decisions[i] + sub_decisions[i] for i in range(num_players) if clubs[i] == club_id) <= 3  # max 3 players

    model += sum(decisions) == 11  # total team size
    model += sum(captain_decisions) == 1  # 1 captain

    for i in range(num_players):
        model += (decisions[i] - captain_decisions[i]) >= 0  # captain must also be on team
        model += (decisions[i] + sub_decisions[i]) <= 1  # subs must not be on team

    model.solve()
    print("Total expected score = {}".format(model.objective.value()))

    return decisions, captain_decisions, sub_decisions


expected_scores = trained_player_data["points"]
trained_player_data["value"] = [i/10 for i in trained_player_data["value"]]
prices = trained_player_data["value"]
positions = trained_player_data["pos"]
names = trained_player_data["web_name"]
clubs = trained_player_data["team_code"]


decisions, captain_decisions, sub_decisions= select_team(expected_scores.values,
                                           prices.values,
                                           positions.values,
                                           clubs.values)


temp = []

for i in range(trained_player_data.shape[0]):
    if decisions[i].value() != 0:
        temp.append([names[i], expected_scores[i], prices[i], positions[i], clubs[i], "player"])

for i in range(trained_player_data.shape[0]):
    if sub_decisions[i].value() == 1:
        temp.append([names[i], expected_scores[i], prices[i], positions[i], clubs[i], "sub"])

# for i in range(trained_player_data.shape[0]):
#     if captain_decisions[i].value() == 1:
#         temp.append([names[i], expected_scores[i], prices[i], positions[i], clubs[i], "captain"])

team_selection = pd.DataFrame([i for i in temp], columns=["name", "points", "value", "pos", "team_code", "type"])
#print(team_selection)
team_selection.to_csv('team_selection_week{}.csv'.format(gameweek.get_recent_gameweek_id()), index=False)

# Transfers

###############################################################
# Ensure current team is picked

current_team = pd.read_csv("../Fantasy-Premier-League/team_162673_data20_21/picks_{}.csv".format(gameweek.get_recent_gameweek_id()))


def conv_team(data):
    ids = list(data["id"])
    points = list(data["points"])
    names = list(data["web_name"])

    for key, val in enumerate(ids):
        if val not in list(current_team["element"]):
            points[key] = -100
    data["points"] = pd.Series(points)
    return data

df = conv_team(pd.read_csv("Player_predictions.csv"))
expected_scores = df["points"]
prices = pd.Series([i/10 for i in df["value"]])
positions = df["pos"]
clubs = df["team_code"]
names = df["web_name"]
################################################################
decisions, captain_decisions, sub_decisions = select_team(expected_scores.values, prices.values, positions.values, clubs.values)
################################################################
player_indices = []
print()
print()
print()
print()
print("First Team:")
for i in range(len(decisions)):
    if decisions[i].value() == 1:
        print("{}{}".format(names[i], "*" if captain_decisions[i].value() == 1 else ""))
        player_indices.append(i)
print()
print("Subs:")
for i in range(len(sub_decisions)):
    if sub_decisions[i].value() == 1:
        print(names[i])
        player_indices.append(i)
score_forecast = trained_player_data["points"]

opt = TransferOptimiser(score_forecast.values, prices.values, prices.values, positions.values, clubs.values)
transfer_in_decisions, transfer_out_decisions, starters, sub_decisions, captain_decisions = opt.solve(player_indices, budget_now=1.0, sub_factor=0.2)

dec = [],[]
for i in range(len(transfer_in_decisions)):
    if transfer_in_decisions[i].value() == 1:
        dec[0].append([names[i], prices[i], score_forecast[i], positions[i]])
    if transfer_out_decisions[i].value() == 1:
        dec[1].append([names[i], prices[i], score_forecast[i],  positions[i]])
print("Transfers in")
print(pd.DataFrame(dec[0][:], columns=["name", "price", "points", "pos"]))
print("Transfers out")
print(pd.DataFrame(dec[1][:], columns=["name", "price", "points", "pos"]))