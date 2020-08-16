# TODO create generic ML model to predict points based on position, team, was_home then predict clean sheets ect
import sklearn
from sklearn import linear_model
import pandas as pd
import numpy as np

def organise_data(merged_gw_data):
    # headers pos, min, team, was_home, x_cleansheet, xG, xA
    '''
    name,assists,bonus,bps,clean_sheets,
    creativity,element,fixture,goals_conceded,
    goals_scored,ict_index,influence,
    kickoff_time,minutes,opponent_team,own_goals,
    penalties_missed,penalties_saved,red_cards,
    round,saves,selected,team_a_score,team_h_score,
    threat,total_points,transfers_balance,transfers_in,
    transfers_out,value,was_home,yellow_cards,GW
    '''

    # First get the player's team, position and fixture difficulties

    # Modify the input data based on the selected features
    heads = ["total_points", "minutes", "was_home"]
    player_data = merged_gw_data[heads]
    # Drop the predicted points label to produce x and y
    x = np.array(player_data.drop(["total_points"], 1))
    y = np.array(player_data)
    return x, y


training_counts = 1


def train_model(x ,y):
    best_acc = 0
    for counts in range(training_counts):
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

        linear = linear_model.LinearRegression()
        linear.fit(x_train, y_train)
        acc = linear.score(x_test, y_test)

        if best_acc <= acc:
            best_acc = acc
    predicted_data_set = [90, True]
    points = float(linear.predict(np.array([predicted_data_set])))
    return points


# Pass data set into organise function
# Organised_data = organise_data(___.csv)
data_in = pd.read_csv("../Fantasy-Premier-League/data/2019-20/gws/merged_gw.csv")
data_in = organise_data(data_in)
# Train the model based on this data
print(train_model(data_in))
# trained_model_output = train_model(Organised_data)

# Use next game week predictions to predict total points
# points = float(linear.predict(np.array([new data])))


