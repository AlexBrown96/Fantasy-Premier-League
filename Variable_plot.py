import pandas as pd
import matplotlib.pyplot as pyplot
from matplotlib import style


data = pd.read_csv("C:/Users/Alext/PycharmProjects/Fantasy-Premier-League/data/2018-19/gws/merged_gw.csv", encoding='latin-1')
heads = ["total_points", "assists", "clean_sheets",
               "goals_conceded", "goals_scored", "minutes", "was_home", "saves", "round"]
data = data[heads][:1000]
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