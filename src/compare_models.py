# creates both models, trains them on same data and checks results
import logging
import sys
from parse_data import load_tracks
from sklearn.model_selection import train_test_split
from training import train
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    RocCurveDisplay,
    ConfusionMatrixDisplay,
)
import numpy as np
import matplotlib.pyplot as plt

#colsample_bytree=0.7,gamma=0.2,min_child_weight=1, max_depth=3, learning_rate=0.15

def main():
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.log(logging.INFO, "Loading dataset")
    features, targets,_ = load_tracks()
    logging.log(logging.INFO, "Fitting models")

    for seed in [1, 2, 3, 4, 5]:
        trainX, testX, trainY, testY = train_test_split(
            features, targets, test_size=0.2, random_state=seed
        )

        # params found from tuning
        xgbr = xgb.XGBRegressor(colsample_bytree=0.7,gamma=0.2,min_child_weight=1, max_depth=3, learning_rate=0.15)
        xgbr = train(xgbr, trainX, trainY, True, "./src/models/xgboost.joblib")

        logistic = LogisticRegression(class_weight="balanced")
        logistic = train(logistic, trainX, trainY, True, "./src/models/xgboost.joblib")

        xgbr_result = xgbr.predict(testX)
        logistic_result = logistic.predict_proba(testX)[:, 1]
        #print(logistic.classes_)

        # get ROC for both models
        for model, result in [("xgbr", xgbr_result), ("logistic", logistic_result)]:
            fpr, tpr, thresholds = roc_curve(testY, result)
            auc = roc_auc_score(testY, result)
            rcd = RocCurveDisplay(fpr=fpr, tpr=tpr)
            rcd.plot()
            plt.plot([0, 1], [0, 1], linestyle="--", label="auc=0.5")
            plt.title(f"{model} AUC={auc}")
            plt.show()

            best_point = np.argmax(tpr - fpr)
            threshold = thresholds[best_point]
            print(threshold)
            # now get classes that are the best according to ROC curve
            classes = result > threshold
            cmat = confusion_matrix(testY, classes)
            d = ConfusionMatrixDisplay(cmat)
            d.plot()
            plt.title("Confusion matrix for " + model)
            plt.show()


if __name__ == "__main__":
    main()
