import pandas as pd
import numpy as np

# Get the next upcoming fixture in the gw for the team
fixture_data = pd.read_csv("../Fantasy-Premier-League/data/2019-20/fixtures.csv")
team_data = pd.read_csv("../Fantasy-Premier-League/data/2019-20/teams.csv")
np_team_data = np.array(team_data)


heads_fixtures = ["team_a", "team_h", "team_a_score", "team_h_score", "team_a_difficulty", "team_h_difficulty"]
heads_team_data = ["name", "strength_defence_home", "strength_defence_away", "strength_attack_home", "strength_attack_away"
                   ,"strength_overall_home", "strength_overall_away"]

team_a = fixture_data["team_a"]
team_h = fixture_data["team_h"]
gameweek = fixture_data["event"]
team_h_difficulty = fixture_data["team_h_difficulty"]
team_a_difficulty = fixture_data["team_a_difficulty"]


def id_to_team_data(id):
    team_id_data = np_team_data[:,3]
    team_index = (np.nonzero(team_id_data == id)[0][0])
    output = team_data[heads_team_data].loc[team_index]
    return output

# # Not sure when this is update so may have to change in later game weeks
# Get a generic gameweek (2019-2020 gw 1)
def fixture_dif_data(team_code, fixture_data):
    dif_home = [[],[]]
    # fixture_data = pd.read_csv("../FPL_ML_2020/data/2019-20/fixtures.csv")
    team_h = fixture_data["team_h"]
    for i,v in enumerate(team_h):
        if team_code == v:
            dif_home[1].append(gameweek[i])
            dif_home[0].append(fixture_data["team_h_difficulty"][i])
        if team_code == team_a[i]:
            dif_home[1].append(gameweek[i])
            dif_home[0].append(fixture_data["team_a_difficulty"][i])
    return dif_home

def player_strength(team_id, position, fixture_home):
    output = id_to_team_data(team_id)
    if fixture_home:
        if position == 1 or position == 2:
            return output["strength_defence_home"]
        else:
            return output["strength_attack_home"]
    else:
        if position == 3 or position == 4:
            return output["strength_defence_away"]
        else:
            return output["strength_attack_away"]


def player_opposition(team_id, gameweek_number=1):
   pass

def main():
    for week in gameweek:
        player_opposition(1, week)

if __name__ == "__main__":
    main()

