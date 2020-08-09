from statistics import mean
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import Fixture_difficulty as fd
from sklearn.linear_model import LinearRegression


def predicted_points(team_code, data, training_counts = 10, n=3):

    #headers = ["total_points", "assists", "clean_sheets", "creativity",
    #"goals_conceded", "goals_scored", "ict_index", "influence",
    #"minutes", "team_a_score", "team_h_score", "threat", "value", "was_home",
    #"yellow_cards", "saves","round"]

    # headers = ["total_points", "assists", "clean_sheets", "creativity",
    #            "goals_conceded", "goals_scored", "ict_index", "influence",
    #            "minutes", "team_a_score", "team_h_score", "threat",
    #            "value", "was_home", "saves", "round"]
    headers = ["total_points", "assists", "clean_sheets", "creativity",
               "goals_conceded", "goals_scored", "ict_index", "influence",
               "minutes", "team_a_score", "team_h_score", "threat",
               "value", "was_home", "saves", "round"]
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
    else:
        points = 0
    n_points = float(linear.predict(np.array([n_predicted_data_set])))


    return points, n_points, acc


