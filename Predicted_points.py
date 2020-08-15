from statistics import mean
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import Fixture_difficulty as fd
from sklearn import preprocessing


def predicted_points(team_code, data, training_counts = 10, n=3):
    # Features used to train the model
    headers = ["total_points", "assists", "clean_sheets",
               "goals_conceded", "goals_scored", "minutes", "team_a_score", "team_h_score",
               "was_home", "saves", "round"]
    # Work out the fixture difficulty rating so that it can be added to the model
    team_dif_data = fd.fixture_dif_data(team_code)
    temp = []
    for k, v in enumerate(data["round"]):
        # Get the indexes of rounds played
        if v in team_dif_data[1]:
            idx = team_dif_data[1].index(v)
            temp.append(team_dif_data[0][idx])
    player_data = data[headers]
    team_dif_data = pd.DataFrame(temp, columns=["fixture_difficulty"])
    headers.append("fixture_difficulty")
    predicted = "total_points"
    player_data = pd.concat([player_data, team_dif_data], axis=1)
    # print(player_data.head())
    #for col in player_data[headers]:
    #   player_data[col] = sklearn.preprocessing.robust_scale(player_data[col])
    x = np.array(player_data.drop([predicted], 1))
    # Array of labels
    y = np.array(player_data[predicted])

    def train_model():
        best_acc = 0
        for counts in range(training_counts):
            x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

            linear = linear_model.LinearRegression()
            linear.fit(x_train, y_train)

            acc = linear.score(x_test, y_test)
            # acc = linear.score(x_train, y_train)
            if best_acc <= acc:
                best_acc = acc

        # with open("Jack_Grelish_model.pickle", "wb") as f:
        #     pickle.dump(linear, f)
        # print(best_acc)
        # print(best_acc)
        return linear, acc

    def best_fit_slope_and_intercept(xs, ys):
        m = (((mean(xs)*mean(ys)) - mean(xs*ys)) /
             ((mean(xs)*mean(xs)) - mean(xs*xs)))

        b = mean(ys) - m*mean(xs)

        return m, b
    # TODO For team_a and team_h you could use the Fixture difficulty
    def predicted_item_point(plot_item, length):
        # Create x and y values for the item to extrapolate the next point
        ys = np.array(player_data[plot_item][:length], dtype=np.float64)
        xs = np.array((range(length)), dtype=np.float64)
        ## plt.plot(xs, ys)
        # Get the gradient and intercept values of the best fit line
        m, b = best_fit_slope_and_intercept(xs, ys)
        ## Create points for the best fit line
        # regression_line = [(m*i)+b for i in xs]
        # Plot the best fit line
        # plt.plot(xs, regression_line)
        # Make prediction based off best fit line creating another point at the end of the data set
        predict_x = xs[-1]+1
        predict_y = (m*predict_x)+b
        return predict_y
        # plt.scatter(predict_x, predict_y, color='r')
        # plt.show()

    predicted_data_set = []
    n_predicted_data_set = []
    headers.remove(predicted)
    for i in headers:
        # Predictions for the whole data set
        predict_y = predicted_item_point(i, len(y))
        predicted_data_set.append(predict_y)
        # Predictions for the nth data set (Recent games)
        predict_y = predicted_item_point(i, n)
        n_predicted_data_set.append(predict_y)


    linear, acc = train_model()
    # print(linear.predict(np.array([predicted_data_set])))
    if acc > -1 and acc <= 1:
        points = float(linear.predict(np.array([predicted_data_set])))
        n_points = float(linear.predict(np.array([n_predicted_data_set])))
    else:
        points = 0
        n_points = 0

    return points, n_points, acc

# TODO create generic ML model to predict points based on position, team, was_home then predict clean sheets ect


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
    headers = []
    # Drop the predicted points label to produce x and y
    x = np.array(1)
    y = np.array(1)
    return Organised_data


training_counts = 1


def train_model(data):
    best_acc = 0
    for counts in range(training_counts):
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

        linear = linear_model.LinearRegression()
        linear.fit(x_train, y_train)
        acc = linear.score(x_test, y_test)

        if best_acc <= acc:
            best_acc = acc
    return linear, acc

# Pass data set into organise function
# Organised_data = organise_data(___.csv)

# Train the model based on this data
# trained_model_output = train_model(Organised_data)

# Use next game week predictions to predict total points
# points = float(linear.predict(np.array([new data])))