import os
import pandas as pd
import numpy as np

gw_dir = '../Fantasy-Premier-League/data/2020-21/gws'
num_players = []
for subdir2, dirs2, files2 in os.walk(gw_dir):
    for file in files2:
        if file != "merged_gw.csv":
            # Read in every gw_data (so far)
            gw_data = pd.read_csv(subdir2+"/"+str(file), sep=",")
            names = [(''.join(filter(lambda j: j.isalpha(), i))) for i in gw_data["name"]]
            gw_data["name"] = pd.Series(names)
            # Read in the corresponding predictions
            ts_data = pd.read_csv("team_selection_week"+(''.join(filter(lambda i: i.isdigit(), file))).replace('gw', '')+".csv")
            a_points = []
            for player in ts_data["name"]:
                try:
                    a_points.append(gw_data["total_points"][gw_data.index[gw_data["name"]==player][0]])
                except:
                    a_points.append("0")
            ts_data["actual_points"] = a_points
            ts_data.to_csv("team_selection_week"+(''.join(filter(lambda i: i.isdigit(), file))).replace('gw', '')+".csv", index=False)
