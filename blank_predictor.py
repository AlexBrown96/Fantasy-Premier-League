import pandas as pd
import numpy as np
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing, neighbors

player_data = pd.read_csv("../Fantasy-Premier-League/data/2019-20/gws/merged_gw.csv")
player_data["blank"] = [0 if i > 4 else 1 for i in player_data["total_points"]]
heads = ["minutes", "was_home", "ict_index", "goals_scored", "assists", "selected",
         "saves", "blank"]

X = np.array(player_data[heads].drop(["blank"], 1))
Y = np.array(player_data["blank"])

def train_model(X, Y):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.005)
    models = [[], []]
    for n in range(1, 10, 1):
        knn = KNeighborsClassifier(n_neighbors=n)
        knn.fit(x_train,y_train)
        acc = knn.score(x_test, y_test)
        models[0].append(acc)
        models[1].append(knn)
    best_acc = max(models[0])
    print("Accuracy: ", best_acc)
    return models[1][models[0].index(best_acc)]

train_model(X, Y)