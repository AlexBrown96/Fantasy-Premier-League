import pandas as pd
import pulp
import numpy as np
trained_player_data = pd.read_csv("Player_predictions.csv")
#trained_player_data2 = pd.read_csv("champ_player_predictions_temp.csv")
#trained_player_data = pd.concat([trained_player_data, trained_player_data2], axis=0, ignore_index=True)

# Code taken from https://github.com/nuebar/forecasting-fantasy-football
# nuebar's code:


def select_team(expected_scores, prices, positions, clubs, total_budget=100, sub_factor=0.2):
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



for i in range(trained_player_data.shape[0]):
    if decisions[i].value() != 0:
        print("**{}** Points = {}, Price = {}, position = {}".format
              (names[i], expected_scores[i], prices[i], positions[i]))
print()
print("Subs:")
# print results
for i in range(trained_player_data.shape[0]):
    if sub_decisions[i].value() == 1:
        print("**{}** Points = {}, Price = {}, position = {}".format
              (names[i], expected_scores[i], prices[i], positions[i]))

print()
print("Captain:")
# print results
for i in range(trained_player_data.shape[0]):
    if captain_decisions[i].value() == 1:
        print("**{}** Points = {}, Price = {}, position = {}".format
              (names[i], expected_scores[i], prices[i], positions[i]))