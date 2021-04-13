from global_scraper import parse_data
from teams_scraper import store_data
from obtain_player_data import obtain_player_data
from Team_selection import team_selection
from gameweek import get_recent_gameweek_id
from pathlib import Path
predict = True
select_team = True

def main():
    file = Path("../Fantasy-Premier-League/data/2020-21/gws/gw{}.csv".format(get_recent_gameweek_id()))
    if not file.is_file():
        parse_data()
    team_id = 162673
    store_data(team_id, 'team_{}_data20_21'.format(team_id))
    if predict:
        obtain_player_data()
    if select_team:
        team_selection(102.4, 0.4)


if __name__ == "__main__":
    main()