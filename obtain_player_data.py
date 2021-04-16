import os
import pandas as pd
from Position_player_model import feature_prediction
from gameweek import get_recent_gameweek_id
from tqdm import tqdm


def obtain_player_data():
    players_dir = '../Fantasy-Premier-League/data/2020-21/players'
    current_player_data = pd.read_csv("../Fantasy-Premier-League/data/2020-21/players_raw.csv")
    current_player_data.loc[current_player_data["chance_of_playing_next_round"] == "None", "chance_of_playing_next_round"] = "100"
    records = []
    with tqdm(total=len(current_player_data["web_name"]), position=0, leave=True, desc="Obtaining player data and predicting points") as pbar:
        for subdir, dirs, files in os.walk(players_dir):
            for file in files:
                if file == "gw.csv":
                    data = pd.read_csv(subdir+"/gw.csv", sep=",")
                    web_name = subdir.replace(players_dir, "")
                    web_name = (''.join(filter(lambda i: i.isalpha(), web_name)))
                    if sum(data[-5:]["minutes"])/len(data[-5:]) >= 0:
                        player_id = int(data["element"][0])
                        player_data = current_player_data[current_player_data["id"] == player_id]
                        if player_data.empty:
                            print("player data for {} empty".format(web_name))
                        else:
                            team_code = int(player_data["team_code"].values[0])
                            points, value, pos, cap_points = feature_prediction(team_code, player_id)
                            chance_playing = int(player_data["chance_of_playing_next_round"].values[0])
                            multiplyer = points * chance_playing/100
                            records.append([web_name, points, value, pos, team_code, player_id, cap_points])
                            #print("{} has been trained, expected points: {}".format(web_name, points))
                    pbar.update()
    df = pd.DataFrame([i for i in records],
                                columns=["web_name", "points", "value", "pos", "team_code", "id", "cap_points"])

    df.to_csv(r"../Fantasy-Premier-League/data/2020-21/selections/player_predictions_week_{}.csv".format(get_recent_gameweek_id()), index=False)

def main():
    obtain_player_data()


if __name__ == "__main__":
    main()