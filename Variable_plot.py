import pandas as pd
import matplotlib.pyplot as pyplot
from matplotlib import style


data = pd.read_csv("../FPL_Machine_learning/Fantasy-Premier-League-master/data/2019-20/players/Kevin_De Bruyne_215/gw.csv", sep=",")
heads = ["total_points", "assists", "clean_sheets",
               "goals_conceded", "goals_scored", "minutes", "was_home", "saves", "round", "opponent_team"]
data = data[heads]
predicted = "total_points"
def plot_player_data(predicted="assists"):
    style.use("ggplot")
    pyplot.scatter(range(len(data["total_points"])), data[predicted])
    pyplot.xlabel("gameweek")
    pyplot.ylabel(predicted)
    pyplot.show()


values = data.values
i = 1
for head in heads:
    pyplot.subplot(len(heads), 1, i)
    pyplot.plot(values[:, heads.index(head)])
    pyplot.title(head, y=0.5, loc="right")
    i += 1
pyplot.show()