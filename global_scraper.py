from parsers import *
from cleaners import *
from getters import *
from collector import collect_gw, merge_gw
from understat import parse_epl_data
import csv
from tqdm import tqdm



def parse_data():
    pbar = tqdm(total=10, desc="Parsing data...", leave=False, position=0)
    """ Parse and store all the data
    """
    season = '2020-21'
    base_filename = 'data/' + season + '/'
    pbar.set_description("Getting data",refresh=True)
    data = get_data()
    pbar.update(1)
    pbar.set_description("Parsing summary data",refresh=True)
    parse_players(data["elements"], base_filename)
    xPoints = []
    for e in data["elements"]:
        xPoint = {}
        xPoint['id'] = e['id']
        xPoint['xP'] = e['ep_this']
        xPoints += [xPoint]
    gw_num = 0
    events = data["events"]
    for event in events:
        if event["is_current"] == True:
            gw_num = event["id"]
    pbar.update(1)
    pbar.set_description("Cleaning summary data",refresh=True)
    clean_players(base_filename + 'players_raw.csv', base_filename)
    pbar.update(1)
    pbar.set_description("Getting fixtures data",refresh=True)
    fixtures(base_filename)
    pbar.update(1)
    pbar.set_description("Getting teams data",refresh=True)
    parse_team_data(data["teams"], base_filename)
    pbar.update(1)
    pbar.set_description("Extracting player ids",refresh=True)
    id_players(base_filename + 'players_raw.csv', base_filename)
    player_ids = get_player_ids(base_filename)
    num_players = len(data["elements"])
    player_base_filename = base_filename + 'players/'
    gw_base_filename = base_filename + 'gws/'
    pbar.update(1)
    pbar.set_description("Extracting player specific data",refresh=True)
    for i,name in tqdm(player_ids.items(), leave=False, position=0):
        player_data = get_individual_player_data(i)
        parse_player_history(player_data["history_past"], player_base_filename, name, i)
        parse_player_gw_history(player_data["history"], player_base_filename, name, i)
    if gw_num > 0:
        pbar.update(1)
        pbar.set_description("Writing expected points",refresh=True)
        with open(os.path.join(gw_base_filename, 'xP' + str(gw_num) + '.csv'), 'w+') as outf:
            w = csv.DictWriter(outf, ['id', 'xP'])
            w.writeheader()
            for xp in xPoints:
                w.writerow(xp)
        pbar.update(1)
        pbar.set_description("Collecting gw scores",refresh=True)
        collect_gw(gw_num, player_base_filename, gw_base_filename)
        pbar.update(1)
        pbar.set_description("Merging gw scores",refresh=True)
        merge_gw(gw_num, gw_base_filename)
    understat_filename = base_filename + 'understat'
    parse_epl_data(understat_filename)
    pbar.update(1)
    pbar.set_description("Done!",refresh=True)


def fixtures(base_filename):
    data = get_fixtures_data()
    parse_fixtures(data, base_filename)

def main():
    parse_data()

if __name__ == "__main__":
    main()