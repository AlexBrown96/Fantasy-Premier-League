import os
import pandas as pd
import numpy as np
gw_dir = '../Fantasy-Premier-League/data/2020-21/gws'
num_players = []
temp_df = pd.DataFrame(columns=["name","points","value","pos","team_code","type","actual_points"])
for subdir2, dirs2, files2 in os.walk(gw_dir, topdown=False):
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
                    a_points.append(int(gw_data["total_points"][gw_data.index[gw_data["name"]==player][0]]))
                except:
                    a_points.append(0)
            #a_points[a_points.index(max(a_points))] = 2*max(a_points)
            ts_data["actual_points"] = a_points
            ts_data["total_points_gw"] = 15*[ts_data["actual_points"].sum()]
            ts_data["gameweek"] = 15*[(''.join(filter(lambda i: i.isdigit(), file)))]
            ts_data.to_csv("team_selection_week"+(''.join(filter(lambda i: i.isdigit(), file))).replace('gw', '')+".csv", index=False)
            temp_df = pd.concat([temp_df, ts_data], ignore_index=True)

temp_df.to_csv("merged_team_selections.csv")