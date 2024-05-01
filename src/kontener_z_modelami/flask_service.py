#!/bin/env python

import json
from flask import Flask, request
from joblib import load
import numpy as np
import csv
import datetime

API_PREFIX = "/api/v1/"
app = Flask(__name__)

base_model = load("models/logistic_regression.joblib")
xgboost = load("models/xgboost.joblib")

XGB_THRESHOLD = 0.635
LOG_THRESHOLD = 0.5


@app.route(API_PREFIX + "base_model", methods=["POST"])
def predict_base():
    values = np.array(json.loads(request.data)["features"])
    result = base_model.predict_proba(values)
    res = []
    for _, prob_1 in result.tolist():
        if prob_1 >= LOG_THRESHOLD:
            res.append(1)
        else:
            res.append(0)
    return res


@app.route(API_PREFIX + "xgboost", methods=["POST"])
def predict_xgboost():
    values = np.array(json.loads(request.data)["features"])
    result = xgboost.predict_proba(values)
    print(result)
    res = []
    for _, prob_1 in result.tolist():
        if prob_1 >= XGB_THRESHOLD:
            res.append(1)
        else:
            res.append(0)
    return res


@app.route(API_PREFIX + "ab", methods=["POST"])
def ab():
    id = json.loads(request.data)["user_id"]
    values = np.array([json.loads(request.data)["features"]])
    model_to_use = id % 2
    threshold,model = (LOG_THRESHOLD,base_model) if model_to_use else (XGB_THRESHOLD,xgboost)
    result = model.predict_proba(values)
    pred = 0 if result[0][0] >= threshold else 1
    model_name = "base model" if model_to_use == 1 else "xgboost"

    with open("db.csv", "a+", newline="") as fh:
        writer = csv.writer(fh, delimiter=",")
        writer.writerow(
            [
                datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
                pred,
                model_name,
                id,
            ]
        )
    return [pred]


if __name__ == "__main__":

    app.run(port=9999)
