from global_scraper import parse_data
from teams_scraper import store_data
from obtain_player_data import obtain_player_data
from Team_selection import team_selection
from gameweek import get_recent_gameweek_id
from pathlib import Path
from understat import parse_player_data
import pandas as pd
from tqdm import tqdm
from Position_player_model import Organise_season_data
from collector import merge_all_gws

predict = True
select_team = True
gather_understat = False
organise_data = False
get_team = False

def main():
    file = Path("../Fantasy-Premier-League/data/2020-21/gws/gw{}.csv".format(get_recent_gameweek_id()))
    ids = pd.read_csv("../Fantasy-Premier-League/data/2020-21/understat/understat_player.csv")
    if gather_understat:
        print("Gathering player understat data from webpages...this make take some time")
        for i in tqdm(ids.id):
            parse_player_data(i)

    if not file.is_file():
        print("Gathering fpl data from webpages")
        parse_data()
        merge_all_gws(get_recent_gameweek_id(), "../Fantasy-Premier-League/data/2020-21/gws")
    if get_team:
        team_id = 162673
        print("Gathering data for team {} from webpages".format(team_id))
        store_data(team_id, 'team_{}_data20_21'.format(team_id))
    if organise_data:
        data_in = pd.read_csv("../Fantasy-Premier-League/data/2020-21/gws/merged_gw.csv")
        Organise_season_data(data_in.set_index("GW"))
    if predict:
        print("Predicting player data based off trained models")
        obtain_player_data()
    if select_team:
        print("Selecting team based on predictions")
        team_selection(102.4, 0.2)


if __name__ == "__main__":
    main()