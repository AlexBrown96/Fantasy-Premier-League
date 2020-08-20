import pandas as pd
import numpy as np
import os

teams_dir = '../Fantasy-Premier-League/champ_promoted_stats'
Prem_players_list = pd.read_csv("../Fantasy-Premier-League/data/2020-21/cleaned_players.csv")
# For all the players listed in the data/year directory, train the model...

# Find champ players in prem players list for next season
Prem_players_list["name"] = ["{} {}".format(x, y) for x,y in list(zip(Prem_players_list["first_name"], Prem_players_list["second_name"]))]



def get_price(name):
    temp_data = (np.array(Prem_players_list))[:, 0]
    if name in Prem_players_list["name"]:
        player_index = (np.nonzero(Prem_players_list["name"] == name)[0][0])
        player_price = (np.array(Prem_players_list))[player_index][17]
        return player_price
    else:
        return 4
print(get_price("Tom Cairne"))