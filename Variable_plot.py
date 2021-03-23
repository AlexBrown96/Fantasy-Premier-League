import pandas as pd
import matplotlib.pyplot as pyplot
from matplotlib import style


data = pd.read_csv("../Fantasy-Premier-League/data/2020-21/gws/merged_gw.csv", encoding='latin-1')
heads = ["total_points", "assists", "clean_sheets",
               "goals_conceded", "goals_scored","ict_index"]
data = data[heads]
predicted = "clean_sheets"
def plot_player_data(predicted="assists"):
    style.use("ggplot")
    pyplot.scatter(data["total_points"], data[predicted])
    pyplot.xlabel("total_points")
    pyplot.ylabel(predicted)
    pyplot.show()

plot_player_data(predicted)

data = data[heads][-600:]
values = data.values
i = 1
for head in heads:
    pyplot.subplot(len(heads), 1, i)
    pyplot.plot(values[:, heads.index(head)])
    pyplot.title(head, y=0.5, loc="right")
    i += 1
pyplot.show()