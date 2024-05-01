# checks which attributes are important in the trained model

import xgboost as xgb
from joblib import load
from parse_data import load_tracks
import matplotlib.pyplot as plt

def check_attrib_importance():
    model = load("src/models/xgboost.joblib")
    _,_,columns = load_tracks()
    model.get_booster().feature_names = list(columns)
    xgb.plot_importance(model)
    plt.show()


if __name__ == "__main__":
    check_attrib_importance()