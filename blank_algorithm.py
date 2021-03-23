import pandas as pd
import numpy as np
import os
import statistics
import matplotlib.pyplot as plt
import seaborn as sns

players_dir = '../Fantasy-Premier-League/data/2020-21/players'
heads = ["web_name", "chance_of_playing_next_round", "news", "points_per_game", "element_type", "team", "now_cost", "team_code"]
num_games = 8

def blank_prediction(total_points, games):
    if games > len(total_points):
        games = len(total_points)
    if len(total_points) < 2:
        return 100, 0, 0
    total_points = total_points[-games:]
    blanks = total_points[total_points < 5]
    chance_of_blank = int(100 * len(blanks) / len(total_points))
    sd_points = statistics.stdev(total_points)
    m = statistics.mean(total_points)
    return chance_of_blank, m, sd_points

def selected_stats(row_index, df_in):
    return df_in[heads].loc[row_index]

blank_data = []
for subdir, dirs, files in os.walk(players_dir):
    for file in files:
        if file == "gw.csv":
            data = pd.read_csv(subdir+"/gw.csv", sep=",")
            name = (''.join(filter(lambda i: i.isalpha(), subdir.replace(players_dir, ""))))
            player_id = (''.join(filter(lambda i: i.isdigit(), subdir))).replace('202021', '')
            #av_points = data["total_points"][-num_games:].sum()/num_games
            chance_of_blank, m, sd_points = blank_prediction(data["total_points"],num_games)
            point_set = data["total_points"][-num_games:]
            blank_data.append([player_id, name, chance_of_blank, m, sd_points, list(point_set), sd_points*2+m])
            print([player_id, name, chance_of_blank, m, sd_points, list(point_set)])


df = pd.DataFrame([i for i in blank_data],
                  columns=["id", "name", "chance_of_blank", "av_points", "STD", "point_set", "95% conf"])
df = df.sort_values(by=["chance_of_blank"]).head(100)
df.to_csv('blank_predictions.csv', index=False)