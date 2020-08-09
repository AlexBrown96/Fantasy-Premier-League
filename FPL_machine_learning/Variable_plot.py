import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import pickle
import matplotlib.pyplot as pyplot
from matplotlib import style


data = pd.read_csv("../FPL_Machine_learning/Fantasy-Premier-League-master/data/2019-20/players/Kevin_De Bruyne_215/gw.csv", sep=",")
data = data[["total_points", "assists", "clean_sheets", "creativity",
             "goals_conceded", "goals_scored", "ict_index", "influence",
             "minutes", "team_a_score", "team_h_score", "threat", "value", "was_home",
             "yellow_cards", "saves"]]
predicted = "total_points"
def plot_player_data(predicted="assists"):
    style.use("ggplot")
    pyplot.scatter(range(len(data["total_points"])), data[predicted])
    pyplot.xlabel("gameweek")
    pyplot.ylabel(predicted)
    pyplot.show()

plot_player_data(predicted)