import pandas as pd
import numpy as np
import os
import statistics
import matplotlib.pyplot as plt
import seaborn as sns
import json
from scipy.stats import truncnorm


def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


data = pd.read_csv("../Fantasy-Premier-League/blank_predictions.csv")

for index, row in data.iterrows():
    frame = [row["name"],
             row["chance_of_blank"],
             row["av_points"],
             row["STD"]]
    reps = 5000
    sims = 1000
    #pct_to_target = np.random.normal(frame[2], frame[3], reps).round(0)
    pct_to_target = get_truncated_normal(frame[2], frame[3], 0, 25)
    fig, ax = plt.subplots(1, sharex=True)
    ax.hist(pct_to_target.rvs(10000), density=True)
    print(pct_to_target.rvs(10000))
    #plt.hist(json.loads(row["point_set"]))
    plt.show()