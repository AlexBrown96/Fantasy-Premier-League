import pickle
import pandas as pd
vals = ["now_cost", "was_home", "team_strength",
        "opp_strength", "xA_dif", "xG_dif", "clean_sheets",
        "ict_index", "minutes", "big_six", "big_six_opp",
        "last_points", "gk", "def", "mid", "fwd"]

with open('general_model.p', "rb") as saved_model:
    linear = pickle.load(saved_model)

temp = pd.DataFrame([linear.coef_], columns=vals)
breakpoint()