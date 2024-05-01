# trains and saves model to models/xgboost.joblib

import xgboost as xgb
import logging
import sys

from training import train
from parse_data import load_tracks
from tune_hyperparameters import tune



def main():
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.log(logging.INFO, "Loading dataset")
    features, targets,columns = load_tracks()
    #tune(features, targets) #use this to find out best parameters
    logging.log(logging.INFO, "Fitting model")
    model = train(xgb.XGBRegressor(colsample_bytree=0.7,gamma=0.2,min_child_weight=1, max_depth=3, learning_rate=0.15), features, targets, True, "./src/models/xgboost.joblib")
if __name__ == "__main__":
    main()
