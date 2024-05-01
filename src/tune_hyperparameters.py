# tune xgboost for the data we are working with
from sklearn.model_selection import RandomizedSearchCV
import xgboost
from parse_data import load_tracks

def tune(features, classes):
    xgb = xgboost.XGBRegressor()

    # source: https://jayant017.medium.com/hyperparameter-tuning-in-xgboost-using-randomizedsearchcv-88fcb5b58a73
    params = {
        "learning_rate" : [0.05,0.10,0.15,0.20,0.25,0.30],
        "max_depth" : [ 3, 4, 5, 6, 8, 10, 12, 15],
        "min_child_weight" : [ 1, 3, 5, 7 ],
        "gamma": [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
        "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]
    }   

    model = RandomizedSearchCV(xgb,param_distributions=params,scoring='roc_auc',verbose=True)
    model.fit(features, classes)
    print(model.best_estimator_)